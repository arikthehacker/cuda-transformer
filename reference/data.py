"""Token files from stage 1 (data/train.bin, data/val.bin) and batches from them.

The whole train.bin is about 5 MB, so the ids get uploaded to the GPU once and
every batch is sliced there. No DataLoader.
"""
from __future__ import annotations

import struct

import numpy as np
import torch

MAGIC = 0x31455042          # "BPE1", same as tokenizer/main.cpp
VERSION = 1
HEADER_BYTES = 12


def load_tokens(path: str) -> np.ndarray:
    """Check magic, version and count, then return the uint16 ids after the
    12 byte header. Raises ValueError if anything does not match."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: {len(raw)} bytes is too short for the {HEADER_BYTES} byte header")

    magic, version, count = struct.unpack_from("<III", raw, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: magic is 0x{magic:08x}, expected 0x{MAGIC:08x} (\"BPE1\")")
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")
    expected = HEADER_BYTES + 2 * count
    if len(raw) != expected:
        raise ValueError(f"{path}: header says {count:,} tokens ({expected:,} bytes) but the file is {len(raw):,} bytes")

    return np.frombuffer(raw, dtype="<u2", count=count, offset=HEADER_BYTES).copy()


def to_device(tokens: np.ndarray | torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """uint16 ids -> int64 tensor on `device`, shape (N,)."""
    if torch.is_tensor(tokens):
        return tokens.to(device=device, dtype=torch.int64)
    return torch.from_numpy(tokens.astype(np.int64)).to(device)


class Batches:
    """Random (x, y) training batches, kept on the GPU.
    x = tokens[i : i + T], y = tokens[i + 1 : i + T + 1].
    Uses its own torch.Generator seeded from TrainConfig.seed, so runs repeat."""

    def __init__(self, tokens: np.ndarray | torch.Tensor, batch_size: int, n_ctx: int,
                 device: torch.device | str, seed: int) -> None:
        self.device = torch.device(device)
        self.tokens = to_device(tokens, self.device)                    # (N,)
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        n = self.tokens.numel()
        if n < n_ctx + 1:
            raise ValueError(f"need at least {n_ctx + 1} tokens, got {n}")
        self.num_starts = n - n_ctx                                     # i + T < N, so y fits
        self.offsets = torch.arange(n_ctx + 1, device=self.device)      # (T+1,)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(x, y), each (B, T) int64."""
        starts = torch.randint(0, self.num_starts, (self.batch_size,),
                               generator=self.gen, device=self.device)  # (B,)
        chunk = self.tokens[starts[:, None] + self.offsets[None, :]]    # (B, T+1)
        return chunk[:, :-1], chunk[:, 1:]                              # (B, T), (B, T)

    def get_state(self) -> torch.Tensor:
        """Generator state, for checkpoints."""
        return self.gen.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.gen.set_state(state)


def fixed_batches(tokens: np.ndarray | torch.Tensor, batch_size: int, n_ctx: int, count: int,
                  device: torch.device | str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The same evenly spaced batches every time, so val loss compares
    fairly across steps and across runs. Returns `count` pairs of (B, T)."""
    data = to_device(tokens, device)                                    # (N,)
    total = count * batch_size
    last = data.numel() - n_ctx - 1                                     # largest valid start
    if last < 0:
        raise ValueError(f"need at least {n_ctx + 1} tokens, got {data.numel()}")

    starts = [0 if total == 1 else (i * last) // (total - 1) for i in range(total)]
    starts_t = torch.tensor(starts, dtype=torch.int64, device=device).view(count, batch_size)  # (count, B)
    offsets = torch.arange(n_ctx + 1, device=device)                    # (T+1,)

    out = []
    for c in range(count):
        chunk = data[starts_t[c][:, None] + offsets[None, :]]           # (B, T+1)
        out.append((chunk[:, :-1], chunk[:, 1:]))                       # (B, T), (B, T)
    return out
