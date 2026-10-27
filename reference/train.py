"""Train the reference GPT on data/train.bin.

    python -m reference.train                exact mode: fp32, deterministic (the answer key)
    python -m reference.train --fast         fp16 autocast + SDPA (the baseline stage 4 races)
    python -m reference.train --resume       continue from data/ref/last.pt

Stage 6, the chat model:

    python -m reference.train --fast --config chat --data data/chat/pretrain --warmup 2000 --out data/chat/ref
    python -m reference.train --fast --init data/chat/pre/weights.bin --data data/chat/sft --sft --out data/chat/ref_sft

--data PREFIX reads PREFIX_train.bin and PREFIX_val.bin (and the tokenizer.bpe
next to them). --init starts from a weights.bin, with its sizes. --sft trains
only on the answers (reference/chat.py): rows start at conversation starts,
the loss is the mean over the counted targets, and lr is 1e-4 with warmup 100
and a fresh AdamW.

Writes data/ref/best.pt (best val loss), data/ref/last.pt (every eval),
data/ref/log.csv (one row per step) and data/ref/samples.txt.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F

import numpy as np

from .config import CONFIGS, REF_DIR, SFT_TRAIN, TOKENIZER_PATH, TRAIN_PATH, VAL_PATH, ModelConfig, TrainConfig, sizes
from .data import Batches, fixed_batches, load_tokens
from .model import GPT
from .tokenizer import NEWLINE, decode, load_bpe, prompt


# ------------------------------------------------------------------ modes

def exact_mode(seed: int) -> None:
    """Seeds, torch.use_deterministic_algorithms(True), TF32 off.
    CUBLAS_WORKSPACE_CONFIG=":4096:8" has to be set before CUDA starts, so call
    this before anything touches the GPU."""
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_initialized():
        raise RuntimeError("CUDA started before exact_mode(), so CUBLAS_WORKSPACE_CONFIG comes too late")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)                     # also seeds every GPU
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def fast_mode(seed: int) -> None:
    """Seeds, but lets PyTorch pick its fastest kernels."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def make_scaler(enabled: bool):
    """GradScaler for fp16. Disabled, every call on it passes straight through."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):         # PyTorch before 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ------------------------------------------------------------------ small pure pieces

def lr_at(step: int, tc: TrainConfig) -> float:
    """Learning rate for 0-based `step`: linear warmup to lr over warmup_steps,
    then cosine down to min_lr at max_steps. With decay_steps > 0 (WSD) it stays
    at lr until the last decay_steps, then falls as min_lr + (lr - min_lr) *
    (1 - sqrt(p)), the shape Hagele et al. 2024 found best. Same numbers as
    cuda/gpt.cu lr_at() and the GPU's lr_schedule()."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step >= tc.max_steps:
        return tc.min_lr
    if tc.decay_steps > 0:
        decay_from = tc.max_steps - tc.decay_steps
        if step < decay_from:
            return tc.lr
        p = (step - decay_from) / tc.decay_steps
        return tc.min_lr + (tc.lr - tc.min_lr) * (1.0 - math.sqrt(p))
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)   # 0 -> 1
    return tc.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (tc.lr - tc.min_lr)


def token_byte_lengths(vocab: list[bytes], tags: dict[str, int], padded_vocab: int) -> list[int]:
    """Bytes of text each id stands for: len(vocab[id]) for real tokens, 0 for
    tags and padded ids (they are not text, so bits per byte leaves them out)."""
    out = [len(v) for v in vocab] + [0] * (padded_vocab - len(vocab))
    for i in tags.values():
        out[i] = 0
    return out


@torch.no_grad()
def evaluate(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]], masked: bool = False,
             token_bytes: torch.Tensor | None = None) -> tuple[float, float]:
    """(val loss, val bits per byte) over the fixed val batches.

    The loss is the mean of the batch means; masked (SFT): each batch's mean
    weighted by its counted targets, so the result is the mean over every
    counted target, the same number cuda/train_fast --sft prints.

    Bits per byte = (loss in nats summed over the counted targets that are
    text, not tags) / (ln 2 x the bytes of those target tokens). Unlike the
    loss per token it does not depend on the tokenizer. nan without token_bytes."""
    V = model.cfg.vocab_size
    batch_means, total, count = [], 0.0, 0
    nats, nbytes = 0.0, 0
    for x, y in batches:
        logits, loss = model(x, y)
        n = int((y >= 0).sum())
        batch_means.append(loss.float())
        if n:
            total += loss.float().item() * n
            count += n
        if token_bytes is not None:
            each = F.cross_entropy(logits[..., :V].float().reshape(-1, V), y.reshape(-1), ignore_index=-1,
                                   reduction="none")                                     # (B*T,), 0 where y = -1
            b = token_bytes[y.reshape(-1).clamp(min=0)] * (y.reshape(-1) >= 0)         # (B*T,) bytes, 0 for tags
            nats += each[b > 0].double().sum().item()
            nbytes += int(b.sum())
    val = total / max(1, count) if masked else torch.stack(batch_means).mean().item()
    bpb = nats / (math.log(2) * nbytes) if nbytes else float("nan")
    return val, bpb


def train_step(model: GPT, opt: torch.optim.Optimizer, scaler, x: torch.Tensor, y: torch.Tensor,
               lr: float, grad_clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One optimizer step on (x, y), each (B, T). Returns (loss, grad norm) as
    GPU tensors, so nothing here waits for the GPU to finish."""
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for group in opt.param_groups:
        group["lr"] = lr
    scaler.step(opt)
    scaler.update()
    return loss.detach(), norm.detach()


def mfu(num_params: int, tokens_per_sec: float, peak_tflops: float) -> float:
    """Fraction of the GPU's peak math in use: 6 x parameters x tokens/s / peak FLOPS."""
    return 6 * num_params * tokens_per_sec / (peak_tflops * 1e12)


# ------------------------------------------------------------------ checkpoints and samples

def save_checkpoint(path: str, model: GPT, opt, scaler, tc: TrainConfig, step: int,
                    best_val: float, best_step: int, evals_since_best: int, batches: Batches,
                    run: dict | None = None) -> None:
    """Model, optimizer, both configs, step, best val and every RNG state, and
    `run` (data prefix, tokenizer, sft) so --resume reads the same data.
    Written to a temp file first, so a crash never leaves half a checkpoint."""
    state = {
        "run": run or {},
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": dataclasses.asdict(model.cfg),
        "train_config": dataclasses.asdict(tc),
        "fast": model.fast,
        "step": step,                       # optimizer steps done
        "best_val": best_val,
        "best_step": best_step,
        "evals_since_best": evals_since_best,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if model.wte.is_cuda else [],
            "batches": batches.get_state(),
        },
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    """Everything onto the CPU first. load_state_dict then moves it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                           # PyTorch before 1.13 has no weights_only
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:              # a value weights_only does not allow
        return torch.load(path, map_location="cpu", weights_only=False)


def question_prompts(tokens: np.ndarray, tags: dict[str, int], starts: np.ndarray, n: int = 3) -> list[list[int]]:
    """The first question of n conversations spread over `starts`, each up to
    and including "<assistant>\\n": what the model answers in an SFT sample."""
    assistant = tags["<assistant>"]
    out = []
    for k in sorted({int(starts[(i * (len(starts) - 1)) // max(1, n - 1)]) for i in range(n)}):
        ids = tokens[k:k + 512].tolist()
        if assistant in ids:
            end = ids.index(assistant) + 1
            out.append(ids[:end] + [NEWLINE])
    return out


def write_samples(model: GPT, vocab: list[bytes], tags: dict[str, int], step: int,
                  tc: TrainConfig, path: str, prompts: list[list[int]] | None = None) -> None:
    """A short sample from each of the first three tags (or from each of
    `prompts`, cut at the first tag the model writes), printed and appended to
    `path`. Each sample has its own generator, so sampling never changes the
    training run."""
    device = model.wte.device
    names = ["<web>"] if "<web>" in tags else list(tags)[:3]       # the chat pretrain text only has <web>
    items = [("question", ids) for ids in prompts] if prompts else [(name, prompt(tags, name)) for name in names]
    tag_ids = set(tags.values())
    with open(path, "a", encoding="utf-8") as f:
        for name, ids in items:
            gen = torch.Generator(device=device)
            gen.manual_seed(tc.seed + step)
            idx = torch.tensor([ids], dtype=torch.long, device=device)                 # (1, t)
            out = model.generate(idx, tc.sample_tokens, 0.8, 40, generator=gen)[0].tolist()
            new = out[len(ids):]
            if prompts:
                stop = next((i for i, t in enumerate(new) if t in tag_ids), None)
                new = new if stop is None else new[:stop + 1]
            text = decode(vocab, ids + new)
            block = f"----- step {step}, {name}\n{text}\n"
            print(block)
            f.write(block + "\n")


# ------------------------------------------------------------------ main

def _e(x: float) -> str:
    """6e-04 -> 6e-4"""
    return f"{x:.0e}".replace("e-0", "e-")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    tc = TrainConfig()
    p = argparse.ArgumentParser(description="Train the reference GPT. Run from the repo root.")
    p.add_argument("--steps", type=int, default=None, help=f"optimizer steps (default {tc.max_steps})")
    p.add_argument("--batch", type=int, default=None, help=f"sequences per step (default {tc.batch_size})")
    p.add_argument("--out", default=REF_DIR, help=f"folder for checkpoints and logs (default {REF_DIR})")
    p.add_argument("--fast", action="store_true", help="fp16 autocast + scaled_dot_product_attention")
    p.add_argument("--resume", action="store_true", help="continue from OUT/last.pt")
    p.add_argument("--starts", metavar="FILE",
                   help="write every step's batch starts to FILE, one line of B numbers per step, so "
                        "cuda/train_fast -s FILE trains on the same batches (costs a little speed)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda or cpu (default: cuda when there is one)")
    p.add_argument("--config", choices=sorted(CONFIGS), default=None,
                   help="model size: small (L 6, C 384, n_ctx 256, the default) or chat (L 8, C 512, n_ctx 512)")
    p.add_argument("--data", metavar="PREFIX",
                   help="PREFIX_train.bin and PREFIX_val.bin, e.g. data/chat/pretrain (default data/train.bin, data/val.bin)")
    p.add_argument("--tokenizer", metavar="FILE",
                   help=f"default {TOKENIZER_PATH}, or tokenizer.bpe in the folder of --data")
    p.add_argument("--init", metavar="FILE", help="start from this weights.bin (and its sizes), not a fresh init")
    p.add_argument("--sft", action="store_true",
                   help=f"train on the answers only (reference/chat.py); lr {_e(SFT_TRAIN['lr'])}, "
                        f"warmup {SFT_TRAIN['warmup_steps']}")
    p.add_argument("--lr", type=float, help=f"peak learning rate (default {_e(tc.lr)}, --sft {_e(SFT_TRAIN['lr'])})")
    p.add_argument("--min-lr", type=float,
                   help=f"learning rate at the end (default {_e(tc.min_lr)}, --sft {_e(SFT_TRAIN['min_lr'])})")
    p.add_argument("--warmup", type=int, help=f"warmup steps (default {tc.warmup_steps}; 2000 for the chat pretrain)")
    p.add_argument("--decay-steps", type=int,
                   help="WSD: flat at lr, then 1 - sqrt down to min_lr over the last N steps (default 0: cosine)")
    p.add_argument("--beta2", type=float, help=f"AdamW beta2 (default {tc.beta2:g})")
    return p.parse_args(argv)


RUN_KEYS = ("data", "tokenizer", "sft")     # saved in every checkpoint, so --resume reads the same data


def data_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    """(train .bin, val .bin, tokenizer) for --data and --tokenizer."""
    if args.data:
        tok = args.tokenizer or os.path.join(os.path.dirname(args.data) or ".", "tokenizer.bpe")
        return f"{args.data}_train.bin", f"{args.data}_val.bin", os.path.normpath(tok)
    return TRAIN_PATH, VAL_PATH, os.path.normpath(args.tokenizer or TOKENIZER_PATH)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for key in ("data", "tokenizer"):           # one spelling of a path, so --resume can compare them
        if getattr(args, key):
            setattr(args, key, os.path.normpath(getattr(args, key)))
    if args.resume and args.init:
        raise SystemExit("--init starts a new run from a weights.bin; --resume continues last.pt, so leave --init out")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")    # emoji in samples on a cp1252 console

    tc = TrainConfig()
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(os.path.join(args.out, "last.pt"))
        tc = TrainConfig(**ckpt["train_config"])
        if ckpt["fast"] != args.fast:
            raise SystemExit(f"last.pt was trained with fast={ckpt['fast']}, so pass the same --fast setting")
        saved = ckpt.get("run", {})
        for key in RUN_KEYS:
            given = getattr(args, key)
            if key in saved and given not in (None, False) and given != saved[key]:
                raise SystemExit(f"last.pt was trained with --{key} {saved[key]}, not {given}")
            if key in saved:
                setattr(args, key, saved[key])
    elif args.sft:
        tc = dataclasses.replace(tc, **SFT_TRAIN)
    if args.steps is not None:
        tc = dataclasses.replace(tc, max_steps=args.steps)
    if args.batch is not None:
        tc = dataclasses.replace(tc, batch_size=args.batch)
    if args.lr is not None:
        tc = dataclasses.replace(tc, lr=args.lr)
    if args.min_lr is not None:
        tc = dataclasses.replace(tc, min_lr=args.min_lr)
    if args.warmup is not None:
        tc = dataclasses.replace(tc, warmup_steps=args.warmup)
    if args.decay_steps is not None:
        tc = dataclasses.replace(tc, decay_steps=args.decay_steps)
    if args.beta2 is not None:
        tc = dataclasses.replace(tc, beta2=args.beta2)
    if not 0 <= tc.decay_steps <= tc.max_steps:
        raise SystemExit(f"--decay-steps {tc.decay_steps} must be from 0 to the {tc.max_steps} steps")
    train_path, val_path, tok_path = data_paths(args)
    run = {"data": args.data, "tokenizer": tok_path, "sft": args.sft}

    if args.fast:
        fast_mode(tc.seed)
    else:
        exact_mode(tc.seed)                         # before anything touches the GPU
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    vocab, tags = load_bpe(tok_path)
    init = None
    if ckpt:
        cfg = ModelConfig(**ckpt["model_config"])
        if args.config and not ModelConfig.for_vocab(cfg.vocab_size, **sizes(args.config)).same_size(cfg):
            raise SystemExit(f"last.pt is L {cfg.n_layer} H {cfg.n_head} C {cfg.n_embd} n_ctx {cfg.n_ctx}, "
                             f"not the {args.config} size")
    elif args.init:
        from .export import read_weights

        cfg, init = read_weights(args.init)
        if args.config and not ModelConfig.for_vocab(cfg.vocab_size, **sizes(args.config)).same_size(cfg):
            raise SystemExit(f"{args.init} is L {cfg.n_layer} H {cfg.n_head} C {cfg.n_embd} n_ctx {cfg.n_ctx}, "
                             f"not the {args.config} size")
    else:
        cfg = ModelConfig.for_vocab(len(vocab), **sizes(args.config or "small"))
    if cfg.vocab_size != len(vocab):
        raise SystemExit(f"{tok_path} has {len(vocab)} ids but the model expects {cfg.vocab_size}")

    train_tokens = load_tokens(train_path)
    val_tokens = load_tokens(val_path)
    for name, toks in ((train_path, train_tokens), (val_path, val_tokens)):
        if int(toks.max()) >= cfg.vocab_size:
            raise SystemExit(f"{name} has id {int(toks.max())} but the tokenizer has {cfg.vocab_size} ids, "
                             "so re-run make run")

    model = GPT(cfg, fast=args.fast)                # weights made on the CPU, then moved
    if init is not None:
        model.load_state_dict({k: torch.from_numpy(a.copy()) for k, a in init.items()}, strict=True)
    model = model.to(device)
    opt = model.optimizer(tc)                       # always a fresh AdamW, --init or not
    scaler = make_scaler(args.fast and device.type == "cuda")
    prompts = None
    if args.sft:
        from .chat import prepare

        sft = {}
        for split, toks in (("train", train_tokens), ("val", val_tokens)):
            y, st, c = prepare(toks, tags, cfg.n_ctx)
            sft[split] = (y.astype(np.int32), st)
            print(f"sft    {split}: {c['conversations']:,} conversations, {c['whole']:,} whole, {c['cut']:,} cut "
                  f"at a turn boundary, {c['dropped']:,} dropped | {c['counted_targets']:,} counted targets")
        batches = Batches(train_tokens, tc.batch_size, cfg.n_ctx, device, tc.seed, *sft["train"])
        val_batches = fixed_batches(val_tokens, tc.batch_size, cfg.n_ctx, tc.eval_batches, device, *sft["val"])
        prompts = question_prompts(val_tokens, tags, sft["val"][1])
    else:
        batches = Batches(train_tokens, tc.batch_size, cfg.n_ctx, device, tc.seed)
        val_batches = fixed_batches(val_tokens, tc.batch_size, cfg.n_ctx, tc.eval_batches, device)

    start_step = 0
    best_val = float("inf")
    best_step = 0
    evals_since_best = 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if opt.defaults.get("fused"):           # PyTorch 2.0 leaves "step" on the CPU, fused wants the GPU
            for p, s in opt.state.items():
                if torch.is_tensor(s.get("step")):
                    s["step"] = s["step"].to(p.device)
        if scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if device.type == "cuda" and ckpt["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        batches.set_state(ckpt["rng"]["batches"])
        start_step, best_val, best_step = ckpt["step"], ckpt["best_val"], ckpt["best_step"]
        evals_since_best = ckpt["evals_since_best"]

    n_params = model.num_params()
    tokens_per_step = tc.batch_size * cfg.n_ctx
    mode = "fast (fp16 autocast + SDPA)" if args.fast else "exact (fp32, deterministic)"
    print(f"model  {n_params:,} parameters | L {cfg.n_layer} H {cfg.n_head} C {cfg.n_embd} "
          f"n_ctx {cfg.n_ctx} | V {cfg.vocab_size} Vp {cfg.padded_vocab}")
    print(f"data   train {len(train_tokens):,} tokens | val {len(val_tokens):,} tokens | "
          f"{tokens_per_step:,} tokens per step")
    print(f"mode   {mode} on {device}" + (f", resuming at step {start_step}" if ckpt else "")
          + (f", from {args.init}" if init is not None else "") + (", sft (answers only)" if args.sft else ""))
    token_bytes = torch.tensor(token_byte_lengths(vocab, tags, cfg.padded_vocab), dtype=torch.int64, device=device)
    val, bpb = evaluate(model, val_batches, args.sft, token_bytes)
    print(f"step {start_step:4d} | val {val:.3f} | bpb {bpb:.4f} | ln(V) = {math.log(cfg.vocab_size):.3f}")

    cuda = device.type == "cuda"
    peak = tc.peak_tflops_fp16 if args.fast else tc.peak_tflops_fp32    # MFU against the peak of the math in use

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def clock():
        """A CUDA event on the GPU (no waiting), or the wall clock on the CPU."""
        if cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def step_ms(start, end) -> float:
        return start.elapsed_time(end) if cuda else (end - start) * 1000.0

    log_path = os.path.join(args.out, "log.csv")
    log_file = open(log_path, "a" if ckpt else "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if not ckpt:
        log.writerow(["step", "train", "val", "lr", "norm", "ms", "val_bpb"])
        log.writerow([0, "", f"{val:.6f}", "", "", "", f"{bpb:.6f}"])

    rows = []                       # finished steps not yet in log.csv: (step, loss, norm, lr, start, end)
    starts_file = open(args.starts, "a" if ckpt else "w", encoding="utf-8") if args.starts else None
    pending_starts = []             # (B,) GPU tensors, written at the next eval (no sync per step)
    timed_from = start_step + tc.timing_skip
    window_start = None             # wall clock at the start of the current timing window
    window_steps = 0
    timed_steps = 0
    timed_secs = 0.0
    stop = False
    step = start_step

    while step < tc.max_steps and not stop:
        if step == timed_from:
            sync()
            window_start = time.perf_counter()
            window_steps = 0

        lr = lr_at(step, tc)
        t0 = clock()
        x, y = batches.next()                                       # (B, T), (B, T)
        if starts_file:
            pending_starts.append(batches.last_starts)
        loss, norm = train_step(model, opt, scaler, x, y, lr, tc.grad_clip)
        rows.append((step + 1, loss, norm, lr, t0, clock()))
        step += 1
        if window_start is not None:
            window_steps += 1

        if step % tc.eval_every == 0 or step == tc.max_steps:
            sync()
            tok_s = float("nan")
            if window_start is not None and window_steps > 0:
                secs = time.perf_counter() - window_start
                tok_s = window_steps * tokens_per_step / secs
                timed_steps += window_steps
                timed_secs += secs

            val, bpb = evaluate(model, val_batches, args.sft, token_bytes)
            losses = [r[1].item() for r in rows]
            for i, (s, l, n, r_lr, a, b) in enumerate(rows):
                last = i == len(rows) - 1
                log.writerow([s, f"{losses[i]:.6f}", f"{val:.6f}" if last else "", f"{r_lr:.6e}",
                              f"{n.item():.4f}", f"{step_ms(a, b):.2f}", f"{bpb:.6f}" if last else ""])
            log_file.flush()
            if starts_file:
                for s in pending_starts:
                    starts_file.write(" ".join(str(v) for v in s.tolist()) + "\n")
                starts_file.flush()
                pending_starts = []
            last_norm = rows[-1][2].item()
            rows = []

            speed = "warming up" if math.isnan(tok_s) else \
                f"{tok_s:,.0f} tok/s | mfu {100 * mfu(n_params, tok_s, peak):.1f}%"
            print(f"step {step:4d} | train {sum(losses) / len(losses):.3f} | val {val:.3f} | bpb {bpb:.4f} | "
                  f"lr {lr:.1e} | norm {last_norm:.2f} | {speed}")

            if val < best_val:
                best_val, best_step, evals_since_best = val, step, 0
                save_checkpoint(os.path.join(args.out, "best.pt"), model, opt, scaler, tc,
                                step, best_val, best_step, evals_since_best, batches, run)
            else:
                evals_since_best += 1
            save_checkpoint(os.path.join(args.out, "last.pt"), model, opt, scaler, tc,
                            step, best_val, best_step, evals_since_best, batches, run)

            if tc.sample_every and step % tc.sample_every == 0:
                write_samples(model, vocab, tags, step, tc, os.path.join(args.out, "samples.txt"), prompts)
            if tc.patience and evals_since_best >= tc.patience:
                print(f"val has not improved for {evals_since_best} evals, stopping early")
                stop = True

            if window_start is not None:            # eval and saving stay out of the timing
                sync()
                window_start = time.perf_counter()
                window_steps = 0

    log_file.close()
    if starts_file:
        starts_file.close()
    print(f"done   best val {best_val:.3f} at step {best_step} -> {os.path.join(args.out, 'best.pt')}")
    if timed_steps > 0:
        avg = timed_steps * tokens_per_step / timed_secs
        note = "" if timed_steps >= 100 else " (fewer than 100 steps, so only a rough number)"
        print(f"speed  {avg:,.0f} tok/s over {timed_steps} timed steps | "
              f"MFU {100 * mfu(n_params, avg, peak):.1f}% of {peak:g} TFLOPS{note}")


if __name__ == "__main__":
    main()
