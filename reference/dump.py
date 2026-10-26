"""The answer key: one forward pass, backward pass and AdamW step of a tiny GPT,
with every value written to data/ref/debug.bin for the stage 3 CUDA tests.

    python -m reference.dump                          tiny model -> data/ref/debug.bin
    python -m reference.dump --full                   default size -> data/ref/debug_full.bin
    python -m reference.dump --read data/ref/debug.bin            list what is inside
    python -m reference.dump --compare data/ref/debug.bin mine.bin   error table

Tiny model: L 2, H 4, C 64, n_ctx 64, B 4, T 64, V from tokenizer.bpe padded
to 64s. Exact mode, fixed seed, the first evenly spaced batch of val.bin.
No gradient clipping before the AdamW step.

debug.bin, little endian:
  char[8]   "GPTREF1\\0"
  uint32    V, Vp, n_ctx, L, H, C, B, T
  float32   lr, beta1, beta2, eps, weight_decay
  uint32    number of records
  each record:
    uint32  name length, then the name bytes (no terminator)
    uint32  dtype: 0 = float32, 1 = int32
    uint32  ndim, then ndim uint32 dims
    data    row-major

Records, in this order:
  config     config.ln_eps                    (LayerNorm eps, a float32 scalar)
  params     wte, wpe, h.0.ln1.w, h.0.ln1.b, h.0.attn.c_attn.w, ..., ln_f.w, ln_f.b
  batch      tokens (B, T) int32, targets (B, T) int32
  forward    emb, h.{l}.ln1, h.{l}.ln1.mean, h.{l}.ln1.rstd, h.{l}.qkv,
             h.{l}.scores, h.{l}.probs, h.{l}.atty, h.{l}.attn_out, h.{l}.res1,
             h.{l}.ln2, h.{l}.ln2.mean, h.{l}.ln2.rstd, h.{l}.fc, h.{l}.gelu,
             h.{l}.mlp_out, h.{l}.res2, ln_f, ln_f.mean, ln_f.rstd, logits, loss
  backward   grad.<every param>, grad.<every forward value>, grad_norm
  optimizer  step1.<every param>              (each param after one AdamW step)

scores holds -inf in the masked (future) spots, and its gradient is 0 there.
grad.loss is 1. grad.*.mean and grad.*.rstd are there for completeness: a
CUDA LayerNorm backward goes straight from dout to dx and can skip them.

Testing the AdamW kernel: feed it the dumped params and grad.* values, not
gradients your own CUDA backward made. At step 1 the update is about
lr * g / (|g| + eps), so a gradient near 0 that differs by fp32 noise can flip
sign and land 2 * lr away, which looks like a bug but isn't.

The key bias is the clearest case. Columns C .. 2C-1 of h.{l}.attn.c_attn.b add
the same vector to every key, which shifts each row of scores by the same
amount, and softmax ignores that. So the true gradient of that slice is exactly
0, and grad.h.{l}.attn.c_attn.b[C:2C] holds only rounding noise (about 1e-11).
Its step1 values come out near lr * noise / (|noise| + eps), about 5e-7, and
differ between any two machines (a CPU dump and a GPU dump already disagree).
Compare that slice with an absolute tolerance of about 1e-6, or skip it.
"""
from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import torch

from .config import DUMP_BATCH, REF_DIR, TOKENIZER_PATH, VAL_PATH, ModelConfig, TrainConfig, tiny_model_config
from .data import fixed_batches, load_tokens
from .model import GPT, Recorder
from .tokenizer import load_bpe
from .train import exact_mode

MAGIC = b"GPTREF1\0"
FLOAT32 = 0
INT32 = 1


def build_records(cfg: ModelConfig, tc: TrainConfig, tokens: np.ndarray, batch_size: int,
                  device: str = "cpu") -> tuple[dict, dict[str, torch.Tensor]]:
    """Run the answer key and return (header, records), records in file order."""
    exact_mode(tc.seed)
    model = GPT(cfg).to(device)
    T = cfg.n_ctx
    x, y = fixed_batches(tokens, batch_size, T, 1, device)[0]                   # (B, T), (B, T)

    records: dict[str, torch.Tensor] = {}
    records["config.ln_eps"] = torch.tensor(cfg.ln_eps, dtype=torch.float32)
    for name, p in model.named_parameters():
        records[name] = p.detach().clone()
    records["tokens"] = x.to(torch.int32)
    records["targets"] = y.to(torch.int32)

    rec = Recorder()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y, rec)
    loss.backward()

    for name, t in rec.values.items():
        records[name] = t.detach().clone()
    for name, p in model.named_parameters():
        records["grad." + name] = p.grad.detach().clone()
    for name, t in rec.values.items():
        if t.grad is None:
            raise RuntimeError(f"{name} has no gradient, so it is not on the path to the loss")
        records["grad." + name] = t.grad.detach().clone()
    squares = sum(p.grad.double().pow(2).sum() for p in model.parameters())
    records["grad_norm"] = squares.sqrt().float()                               # () global L2 norm

    opt = model.optimizer(tc)                       # lr = tc.lr, no warmup, no clipping
    opt.step()
    for name, p in model.named_parameters():
        records["step1." + name] = p.detach().clone()

    header = {
        "V": cfg.vocab_size, "Vp": cfg.padded_vocab, "n_ctx": cfg.n_ctx, "L": cfg.n_layer,
        "H": cfg.n_head, "C": cfg.n_embd, "B": batch_size, "T": T,
        "lr": tc.lr, "beta1": tc.beta1, "beta2": tc.beta2, "eps": tc.eps, "weight_decay": tc.weight_decay,
    }
    return header, records


HEADER_INTS = ("V", "Vp", "n_ctx", "L", "H", "C", "B", "T")
HEADER_FLOATS = ("lr", "beta1", "beta2", "eps", "weight_decay")


def write_dump(path: str, header: dict, records: dict[str, torch.Tensor]) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8I", *(header[k] for k in HEADER_INTS)))
        f.write(struct.pack("<5f", *(header[k] for k in HEADER_FLOATS)))
        f.write(struct.pack("<I", len(records)))
        for name, t in records.items():
            arr = t.detach().cpu().contiguous().numpy()
            if arr.dtype == np.float32:
                code, data = FLOAT32, arr.astype("<f4", copy=False)
            elif arr.dtype == np.int32:
                code, data = INT32, arr.astype("<i4", copy=False)
            else:
                raise TypeError(f"{name}: dtype {arr.dtype} is not float32 or int32")
            raw_name = name.encode("utf-8")
            f.write(struct.pack("<I", len(raw_name)))
            f.write(raw_name)
            f.write(struct.pack("<II", code, arr.ndim))
            if arr.ndim:
                f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(np.ascontiguousarray(data).tobytes())


def read_dump(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """The matching reader: (header, {name: array}) in file order."""
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: not a GPTREF1 file")
    ints = struct.unpack_from("<8I", buf, 8)
    floats = struct.unpack_from("<5f", buf, 40)
    (count,) = struct.unpack_from("<I", buf, 60)
    header = {**dict(zip(HEADER_INTS, ints)), **dict(zip(HEADER_FLOATS, floats))}

    pos = 64
    records: dict[str, np.ndarray] = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<I", buf, pos)
        name = buf[pos + 4: pos + 4 + name_len].decode("utf-8")
        pos += 4 + name_len
        code, ndim = struct.unpack_from("<II", buf, pos)
        pos += 8
        dims = struct.unpack_from(f"<{ndim}I", buf, pos) if ndim else ()
        pos += 4 * ndim
        n = int(np.prod(dims)) if ndim else 1
        dtype = {FLOAT32: "<f4", INT32: "<i4"}[code]
        records[name] = np.frombuffer(buf, dtype=dtype, count=n, offset=pos).reshape(dims).copy()
        pos += 4 * n
    if pos != len(buf):
        raise ValueError(f"{path}: {len(buf) - pos} bytes left over after {count} records")
    return header, records


def error_stats(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    """(max absolute error, max error relative to the largest |ref| value).
    Equal values count as 0 error, including two matching -inf."""
    a = ref.astype(np.float64)
    b = got.astype(np.float64)
    with np.errstate(invalid="ignore"):
        diff = np.where(a == b, 0.0, np.abs(a - b))
    diff = np.where(np.isnan(diff), np.inf, diff)           # a NaN on either side is never a match
    max_abs = float(diff.max()) if diff.size else 0.0
    finite = np.abs(a[np.isfinite(a)])
    scale = float(finite.max()) if finite.size else 0.0
    if scale > 0:
        return max_abs, max_abs / scale
    return max_abs, (0.0 if max_abs == 0 else float("inf"))


def compare(ref: dict[str, np.ndarray], got: dict[str, np.ndarray], tol: float = 1e-4) -> int:
    """Print max abs and max rel error per tensor, in record order. Marks the
    first tensor over `tol` (relative), because that is where a bug starts.
    Returns how many tensors were over tol or missing."""
    print(f"{'name':34s} {'shape':18s} {'max abs':>10s} {'max rel':>10s}")
    bad = 0
    for name, a in ref.items():
        shape = str(tuple(a.shape))
        if name not in got:
            print(f"{name:34s} {shape:18s} {'missing':>10s}")
            bad += 1
            continue
        b = got[name]
        if b.shape != a.shape:
            print(f"{name:34s} {shape:18s} shape {tuple(b.shape)} does not match")
            bad += 1
            continue
        max_abs, max_rel = error_stats(a, b)
        flag = ""
        if not max_rel <= tol:                              # also catches NaN
            flag = "   <-- first mismatch" if bad == 0 else "   !"
            bad += 1
        print(f"{name:34s} {shape:18s} {max_abs:10.3e} {max_rel:10.3e}{flag}")
    return bad


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write, read or compare the GPTREF1 answer key. Run from the repo root.")
    p.add_argument("--full", action="store_true", help="default model size instead of the tiny one")
    p.add_argument("--out", default=None, help="output file (default data/ref/debug.bin or debug_full.bin)")
    p.add_argument("--device", default="cpu", help="cpu (default, same numbers on any machine) or cuda")
    p.add_argument("--read", metavar="FILE", help="list the header and records of FILE")
    p.add_argument("--compare", nargs=2, metavar=("REF", "OTHER"), help="error table of OTHER against REF")
    p.add_argument("--tol", type=float, default=1e-4, help="relative error that counts as a mismatch (default 1e-4)")
    args = p.parse_args(argv)

    if args.read:
        header, records = read_dump(args.read)
        print(" ".join(f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}" for k, v in header.items()))
        for name, arr in records.items():
            print(f"{name:34s} {str(arr.dtype):8s} {tuple(arr.shape)}")
        return
    if args.compare:
        _, ref = read_dump(args.compare[0])
        _, got = read_dump(args.compare[1])
        bad = compare(ref, got, args.tol)
        print(f"\n{bad} tensor(s) over tolerance" if bad else "\nevery tensor within tolerance")
        raise SystemExit(1 if bad else 0)

    vocab, _ = load_bpe(TOKENIZER_PATH)
    cfg = ModelConfig.for_vocab(len(vocab)) if args.full else tiny_model_config(len(vocab))
    out = args.out or os.path.join(REF_DIR, "debug_full.bin" if args.full else "debug.bin")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    header, records = build_records(cfg, TrainConfig(), load_tokens(VAL_PATH), DUMP_BATCH, args.device)
    write_dump(out, header, records)

    _, back = read_dump(out)                        # read it straight back
    same = list(back) == list(records) and all(
        np.array_equal(records[k].cpu().numpy(), back[k]) for k in records)
    if not same:
        raise SystemExit(f"{out}: reading the file back did not give the same tensors")
    print(f"wrote {out}: {len(records)} records, {os.path.getsize(out) / 1e6:.1f} MB, "
          f"loss {records['loss'].item():.4f} (ln V = {np.log(cfg.vocab_size):.4f}), read back identical")


if __name__ == "__main__":
    main()
