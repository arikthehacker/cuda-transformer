"""Token files (data/train.bin, data/chat/pretrain_train.bin ...) and batches from them.

A small train.bin (stages 1 to 5, about 5 MB) gets uploaded to the GPU once
and every batch is sliced there. A big one (stage 6 pretraining, about 1B ids)
stays in CPU memory as uint16 and only each batch goes to the GPU: as int64 on
the GPU it would be 8 GB. No DataLoader.

For SFT, both helpers also take the -1 targets and the allowed starts from
reference/chat.py: x = tokens[s : s + T], y = targets[s : s + T].
"""
from __future__ import annotations

import os
import struct

import numpy as np
import torch

MAGIC = 0x31455042          # "BPE1", same as tokenizer/main.cpp
VERSION = 1
HEADER_BYTES = 12
GPU_TOKENS_MAX = 1 << 26    # token files longer than this (64M ids) stay in CPU memory


def load_tokens(path: str) -> np.ndarray:
    """Check magic, version and count, then return the uint16 ids after the
    12 byte header. Raises ValueError if anything does not match. Reads the
    ids straight into one array, so a 2 GB file needs 2 GB, not 4."""
    size = os.path.getsize(path)
    if size < HEADER_BYTES:
        raise ValueError(f"{path}: {size} bytes is too short for the {HEADER_BYTES} byte header")
    with open(path, "rb") as f:
        magic, version, count = struct.unpack("<III", f.read(HEADER_BYTES))
        if magic != MAGIC:
            raise ValueError(f"{path}: magic is 0x{magic:08x}, expected 0x{MAGIC:08x} (\"BPE1\")")
        if version != VERSION:
            raise ValueError(f"{path}: version {version}, expected {VERSION}")
        expected = HEADER_BYTES + 2 * count
        if size != expected:
            raise ValueError(f"{path}: header says {count:,} tokens ({expected:,} bytes) but the file is {size:,} bytes")
        return np.fromfile(f, dtype="<u2", count=count)


def to_device(tokens: np.ndarray | torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """uint16 ids -> int64 tensor on `device`, shape (N,)."""
    if torch.is_tensor(tokens):
        return tokens.to(device=device, dtype=torch.int64)
    return torch.from_numpy(tokens.astype(np.int64)).to(device)


def gather(tokens: np.ndarray, starts: np.ndarray, n_ctx: int,
           targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(x, y), each (B, T) int64, from CPU arrays. Without targets
    y = tokens[s + 1 : s + T + 1], with them y = targets[s : s + T]."""
    starts = np.asarray(starts, dtype=np.int64)
    if targets is None:
        chunk = tokens[starts[:, None] + np.arange(n_ctx + 1)].astype(np.int64)  # (B, T+1)
        return chunk[:, :-1], chunk[:, 1:]
    idx = starts[:, None] + np.arange(n_ctx)                                    # (B, T)
    return tokens[idx].astype(np.int64), targets[idx].astype(np.int64)


def _cpu_array(a: np.ndarray | torch.Tensor) -> np.ndarray:
    return a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


class Batches:
    """Random (x, y) training batches.
    x = tokens[i : i + T], y = tokens[i + 1 : i + T + 1], or with SFT targets
    y = targets[i : i + T] and i drawn only from `starts`.
    Uses its own torch.Generator seeded from TrainConfig.seed, so runs repeat."""

    def __init__(self, tokens: np.ndarray | torch.Tensor, batch_size: int, n_ctx: int,
                 device: torch.device | str, seed: int, targets: np.ndarray | None = None,
                 starts: np.ndarray | None = None) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        n = len(tokens)
        if n < n_ctx + 1:
            raise ValueError(f"need at least {n_ctx + 1} tokens, got {n}")
        if (targets is None) != (starts is None):
            raise ValueError("SFT batches need both targets and starts")
        if targets is not None and len(targets) != n:
            raise ValueError(f"{len(targets):,} targets for {n:,} tokens")
        self.on_gpu = n <= GPU_TOKENS_MAX
        if self.on_gpu:
            self.tokens = to_device(tokens, self.device)                    # (N,)
            self.targets = None if targets is None else to_device(targets, self.device)
        else:
            self.tokens = _cpu_array(tokens)                                # (N,) uint16, CPU
            self.targets = None if targets is None else _cpu_array(targets)
        if starts is None:
            self.starts = None
            self.num_starts = n - n_ctx                                     # i + T < N, so y fits
        else:
            if len(starts) == 0:
                raise ValueError("no conversation starts to draw batches from")
            self.starts = torch.as_tensor(np.asarray(starts, dtype=np.int64), device=self.device)
            self.num_starts = len(starts)
        self.offsets = torch.arange(n_ctx + 1, device=self.device)          # (T+1,)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(x, y), each (B, T) int64."""
        pick = torch.randint(0, self.num_starts, (self.batch_size,),
                             generator=self.gen, device=self.device)        # (B,)
        starts = pick if self.starts is None else self.starts[pick]
        self.last_starts = starts                                           # train.py --starts writes these
        if not self.on_gpu:
            x, y = gather(self.tokens, starts.cpu().numpy(), self.n_ctx, self.targets)
            return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        if self.targets is None:
            chunk = self.tokens[starts[:, None] + self.offsets[None, :]]    # (B, T+1)
            return chunk[:, :-1], chunk[:, 1:]                              # (B, T), (B, T)
        idx = starts[:, None] + self.offsets[None, :-1]                     # (B, T)
        return self.tokens[idx], self.targets[idx]

    def get_state(self) -> torch.Tensor:
        """Generator state, for checkpoints."""
        return self.gen.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.gen.set_state(state)


def fixed_starts(num: int, total: int, n_tokens: int, n_ctx: int,
                 starts: np.ndarray | None = None) -> np.ndarray:
    """`total` evenly spaced batch starts: over every start a row fits at, or
    over the given SFT starts (picked evenly from the list). int64 (total,)."""
    if starts is None:
        last = n_tokens - n_ctx - 1                                         # largest valid start
        if last < 0:
            raise ValueError(f"need at least {n_ctx + 1} tokens, got {n_tokens}")
        return np.array([0 if total == 1 else (i * last) // (total - 1) for i in range(total)], dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("no conversation starts to take val batches from")
    pick = [0 if total == 1 else (i * (starts.size - 1)) // (total - 1) for i in range(total)]
    return starts[pick]


def fixed_batches(tokens: np.ndarray | torch.Tensor, batch_size: int, n_ctx: int, count: int,
                  device: torch.device | str, targets: np.ndarray | None = None,
                  starts: np.ndarray | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The same evenly spaced batches every time, so val loss compares
    fairly across steps and across runs. Returns `count` pairs of (B, T).
    Gathered on the CPU, so only the batches go to the GPU."""
    data = _cpu_array(tokens)
    tgt = None if targets is None else _cpu_array(targets)
    s = fixed_starts(count, count * batch_size, data.size, n_ctx, starts).reshape(count, batch_size)
    out = []
    for c in range(count):
        x, y = gather(data, s[c], n_ctx, tgt)
        out.append((torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)))   # (B, T), (B, T)
    return out
