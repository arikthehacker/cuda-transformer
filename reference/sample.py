"""Generate text from a checkpoint.

    python -m reference.sample --ckpt data/ref/best.pt --tag bible --tokens 300 \\
                               --temperature 0.8 --top_k 40 --seed 1

--tag and --seed both take several values and every pair gets printed, so
"--tag bible austen stories --seed 1 2" puts six samples side by side.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from .config import REF_DIR, TOKENIZER_PATH, ModelConfig
from .model import GPT
from .tokenizer import decode, load_bpe, prompt
from .train import load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a reference GPT checkpoint. Run from the repo root.")
    p.add_argument("--ckpt", default=os.path.join(REF_DIR, "best.pt"), help="checkpoint (default data/ref/best.pt)")
    p.add_argument("--tag", nargs="+", default=["bible"], help="tag(s) to start from, e.g. bible austen")
    p.add_argument("--tokens", type=int, default=300, help="new tokens per sample (default 300)")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = always the most likely token (default 0.8)")
    p.add_argument("--top_k", type=int, default=40, help="sample from the k most likely tokens, 0 = all (default 40)")
    p.add_argument("--seed", type=int, nargs="+", default=[1], help="seed(s) (default 1)")
    p.add_argument("--keep_going", action="store_true",
                   help="keep generating past the next tag (by default a sample ends where a new chunk would start)")
    p.add_argument("--fast", action="store_true", help="sample in fp16 autocast")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer", default=TOKENIZER_PATH, help="tokenizer file (default data/tokenizer.bpe)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")    # emoji on a cp1252 console

    ckpt = load_checkpoint(args.ckpt)
    cfg = ModelConfig(**ckpt["model_config"])
    vocab, tags = load_bpe(args.tokenizer)
    if len(vocab) != cfg.vocab_size:
        raise SystemExit(f"{args.tokenizer} has {len(vocab)} ids but {args.ckpt} was trained with {cfg.vocab_size}")

    device = torch.device(args.device)
    model = GPT(cfg, fast=args.fast)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tag_ids = set(tags.values())
    print(f"{args.ckpt}: step {ckpt['step']}, val {ckpt['best_val']:.3f}, {model.num_params():,} parameters\n")

    for tag in args.tag:
        start = prompt(tags, tag)                                               # [tag_id, 10]
        for seed in args.seed:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            idx = torch.tensor([start], dtype=torch.long, device=device)        # (1, 2)
            out = model.generate(idx, args.tokens, args.temperature, args.top_k, generator=gen)
            new = out[0, len(start):].tolist()
            ended = ""
            if not args.keep_going:
                cut = next((i for i, t in enumerate(new) if t in tag_ids), None)
                if cut is not None:
                    ended = f"\n[the model started a new chunk: {vocab[new[cut]].decode()}]"
                    new = new[:cut]
            print(f"===== {tag}, seed {seed}")
            print(decode(vocab, start + new) + ended + "\n")


if __name__ == "__main__":
    main()
