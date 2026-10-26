"""A checkpoint -> weights.bin, the one weight file both sides read.

    python -m reference.export                                  data/ref/best.pt -> data/ref/weights.bin
    python -m reference.export --ckpt data/ref/last.pt --out data/ref/last.bin
    python -m reference.export --fp16                           half the size (generation, the browser demo)
    python -m reference.export --read data/ref/weights.bin      print the header and every tensor's shape
    python -m reference.export --from-dump data/ref/debug.bin --out data/ref/weights_debug.bin
                                                                a dump's starting weights (for the tests)

cuda/train_fast writes the same format after training, and reference/sample.py
takes a .bin in --ckpt, so a model trained by either side can be sampled by
either side.

weights.bin, little endian, a 48 byte header, then the parameters:
  uint32   magic 0x57545047 ("GPTW" as bytes), version 1
  uint32   V, Vp, n_ctx, L, H, C
  float32  ln_eps
  uint32   dtype: 0 = float32, 1 = float16
  uint64   count: how many values follow
  data     every parameter in named_parameters() order (the dump order), row-major

Only numpy is needed to read and write the file. torch is imported only to
open a checkpoint.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

from .config import REF_DIR, ModelConfig

MAGIC = 0x57545047
VERSION = 1
HEADER = struct.Struct("<8I f I Q")          # magic, version, V, Vp, n_ctx, L, H, C, ln_eps, dtype, count
assert HEADER.size == 48


def param_shapes(cfg: ModelConfig) -> list[tuple[str, tuple[int, ...]]]:
    """(name, shape) of every parameter, in named_parameters() order."""
    C = cfg.n_embd
    out = [("wte", (cfg.padded_vocab, C)), ("wpe", (cfg.n_ctx, C))]
    for layer in range(cfg.n_layer):
        p = f"h.{layer}."
        out += [(p + "ln1.w", (C,)), (p + "ln1.b", (C,)),
                (p + "attn.c_attn.w", (3 * C, C)), (p + "attn.c_attn.b", (3 * C,)),
                (p + "attn.c_proj.w", (C, C)), (p + "attn.c_proj.b", (C,)),
                (p + "ln2.w", (C,)), (p + "ln2.b", (C,)),
                (p + "mlp.c_fc.w", (4 * C, C)), (p + "mlp.c_fc.b", (4 * C,)),
                (p + "mlp.c_proj.w", (C, 4 * C)), (p + "mlp.c_proj.b", (C,))]
    out += [("ln_f.w", (C,)), ("ln_f.b", (C,))]
    return out


def write_weights(path: str, cfg: ModelConfig, params: dict[str, np.ndarray], fp16: bool = False) -> int:
    """Write params (name -> array) to path. Returns the number of values."""
    dtype = np.float16 if fp16 else np.float32
    chunks = []
    for name, shape in param_shapes(cfg):
        if name not in params:
            raise ValueError(f"no parameter named {name}")
        a = np.asarray(params[name])
        if tuple(a.shape) != shape:
            raise ValueError(f"{name} is {tuple(a.shape)}, the config says {shape}")
        chunks.append(np.ascontiguousarray(a, dtype=np.float32).astype(dtype).ravel())
    data = np.concatenate(chunks)
    tmp = path + ".tmp"                      # a crash never leaves half a file under the real name
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, cfg.vocab_size, cfg.padded_vocab, cfg.n_ctx, cfg.n_layer, cfg.n_head,
                            cfg.n_embd, cfg.ln_eps, 1 if fp16 else 0, data.size))
        f.write(data.astype("<f2" if fp16 else "<f4").tobytes())
    os.replace(tmp, path)
    return int(data.size)


def read_weights(path: str) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    """(config, name -> float32 array) from a weights.bin file."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path} is too short to be a weights.bin file")
    magic, version, V, Vp, n_ctx, L, H, C, ln_eps, dtype, count = HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a weights.bin file")
    if version != VERSION or dtype not in (0, 1):
        raise ValueError(f"{path} has version {version}, dtype {dtype}; this reads version 1, dtype 0 or 1")
    cfg = ModelConfig(vocab_size=V, padded_vocab=Vp, n_ctx=n_ctx, n_layer=L, n_head=H, n_embd=C,
                      ln_eps=float(np.float32(ln_eps)))
    shapes = param_shapes(cfg)
    total = sum(int(np.prod(s)) for _, s in shapes)
    width = 2 if dtype == 1 else 4
    if count != total or len(raw) != HEADER.size + count * width:
        raise ValueError(f"{path} holds {count} values in {len(raw)} bytes, a model of its sizes has {total}")
    data = np.frombuffer(raw, dtype="<f2" if dtype == 1 else "<f4", offset=HEADER.size).astype(np.float32)
    params, at = {}, 0
    for name, shape in shapes:
        n = int(np.prod(shape))
        params[name] = data[at:at + n].reshape(shape)
        at += n
    return cfg, params


def dump_params(path: str) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    """The config and parameter records of a GPTREF1 dump (reference/dump.py), numpy only."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:8] != b"GPTREF1\0":
        raise ValueError(f"{path} is not a GPTREF1 dump")
    V, Vp, n_ctx, L, H, C, _, _ = struct.unpack_from("<8I", raw, 8)
    at = 8 + 32 + 20
    (count,) = struct.unpack_from("<I", raw, at)
    at += 4
    records = {}
    for _ in range(count):
        (n,) = struct.unpack_from("<I", raw, at)
        name = raw[at + 4:at + 4 + n].decode()
        at += 4 + n
        dtype, ndim = struct.unpack_from("<II", raw, at)
        at += 8
        dims = struct.unpack_from(f"<{ndim}I", raw, at)
        at += 4 * ndim
        size = int(np.prod(dims)) if ndim else 1
        records[name] = np.frombuffer(raw, dtype="<f4" if dtype == 0 else "<i4", count=size, offset=at).reshape(dims)
        at += 4 * size
    eps = float(records["config.ln_eps"]) if "config.ln_eps" in records else 1e-5
    cfg = ModelConfig(vocab_size=V, padded_vocab=Vp, n_ctx=n_ctx, n_layer=L, n_head=H, n_embd=C, ln_eps=eps)
    return cfg, {name: records[name] for name, _ in param_shapes(cfg)}


def load_into(path: str, fast: bool = False):
    """A reference GPT with the weights from a weights.bin file, on the CPU."""
    import torch

    from .model import GPT

    cfg, params = read_weights(path)
    model = GPT(cfg, fast=fast)
    state = {name: torch.from_numpy(a.copy()) for name, a in params.items()}
    model.load_state_dict(state, strict=True)
    return cfg, model


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write a checkpoint's weights to weights.bin. Run from the repo root.")
    p.add_argument("--ckpt", default=os.path.join(REF_DIR, "best.pt"), help="checkpoint (default data/ref/best.pt)")
    p.add_argument("--out", default=os.path.join(REF_DIR, "weights.bin"), help="output (default data/ref/weights.bin)")
    p.add_argument("--fp16", action="store_true", help="store the values as float16")
    p.add_argument("--read", metavar="FILE", help="print what is inside FILE and stop")
    p.add_argument("--from-dump", metavar="DUMP", help="take the weights from a reference/dump.py file instead")
    args = p.parse_args(argv)

    if args.from_dump:
        cfg, params = dump_params(args.from_dump)
        n = write_weights(args.out, cfg, params, args.fp16)
        print(f"{args.from_dump} -> {args.out}: {n:,} values, {'fp16' if args.fp16 else 'fp32'}")
        return

    if args.read:
        cfg, params = read_weights(args.read)
        print(f"{args.read}: L {cfg.n_layer} H {cfg.n_head} C {cfg.n_embd} n_ctx {cfg.n_ctx} "
              f"V {cfg.vocab_size} Vp {cfg.padded_vocab} ln_eps {cfg.ln_eps:g}")
        for name, a in params.items():
            print(f"  {name:24s} {str(a.shape):14s} mean {a.mean():+.4e} std {a.std():.4e}")
        print(f"  {sum(a.size for a in params.values()):,} values")
        return

    from .train import load_checkpoint

    ckpt = load_checkpoint(args.ckpt)
    cfg = ModelConfig(**ckpt["model_config"])
    params = {name: t.detach().float().cpu().numpy() for name, t in ckpt["model"].items()}
    n = write_weights(args.out, cfg, params, args.fp16)
    step = ckpt.get("step", "?")
    print(f"{args.ckpt} (step {step}) -> {args.out}: {n:,} values, {'fp16' if args.fp16 else 'fp32'}, "
          f"{os.path.getsize(args.out):,} bytes")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()
