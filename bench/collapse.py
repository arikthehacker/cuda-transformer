"""How often the cross entropy kernels round the right token's gradient to 0 on the chat model's val text.

    python.exe -m bench.collapse          -> bench/results_collapse.md (bench/results.py copies it)
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from reference.chat import read_targets
from reference.data import load_tokens
from reference.export import load_into
from reference.tokenizer import decode, load_bpe

CHAT = "data/chat"
N_CTX = 512
CLIFFS = {"fp16": 2.0 ** -11, "bf16": 2.0 ** -8}


def sft_rows(tokens: np.ndarray, targets_path: str, limit: int):
    y, starts, n_ctx = read_targets(targets_path)
    ends = np.append(starts[1:], tokens.size)
    for s, e in list(zip(starts, ends))[:limit or None]:
        idx = np.arange(s, s + n_ctx)
        t = y[idx].astype(np.int64)
        t[idx >= e] = -1
        yield tokens[idx].astype(np.int64), t


def web_rows(tokens: np.ndarray, limit: int):
    count = (tokens.size - 1) // N_CTX
    for r in range(min(count, limit) if limit else count):
        chunk = tokens[r * N_CTX:(r + 1) * N_CTX + 1].astype(np.int64)
        yield chunk[:-1], chunk[1:]


def batches(rows, size: int):
    xs, ys = [], []
    for x, y in rows:
        xs.append(x)
        ys.append(y)
        if len(xs) == size:
            yield np.stack(xs), np.stack(ys)
            xs, ys = [], []
    if xs:
        yield np.stack(xs), np.stack(ys)


def measure(weights: str, rows, batch: int, chunk: int) -> tuple[Counter, Counter]:
    cfg, model = load_into(weights, fast=True)
    model = model.cuda().eval()
    V = cfg.vocab_size
    counts, hits = Counter(), Counter()
    with torch.inference_mode():
        for xb, yb in batches(rows, batch):
            x = torch.from_numpy(xb).cuda()
            y = torch.from_numpy(yb).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                h = model.hidden(x)
            mask = y >= 0
            pos = mask.nonzero().cpu().numpy()
            hs, ts = h[mask], y[mask]
            for a in range(0, ts.numel(), chunk):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = F.linear(hs[a:a + chunk], model.wte)
                L = logits[:, :V].float()
                t = ts[a:a + chunk]
                lt = L.gather(1, t[:, None]).squeeze(1)
                m = L.max(1).values
                p = torch.exp(lt - m) / torch.exp(L - m[:, None]).sum(1)
                zero = p == 1.0
                rest = torch.exp((L - lt[:, None]).clamp(max=80.0)).scatter_(1, t[:, None], 0.0).sum(1)
                counts["tokens"] += t.numel()
                counts["fp32"] += int(zero.sum())
                for name, cliff in CLIFFS.items():
                    counts[name] += int((rest < cliff).sum())
                for k in zero.nonzero().squeeze(1).tolist():
                    r, c = pos[a + k]
                    hits[(tuple(xb[r, max(0, c - 1):c + 1].tolist()), int(t[k]))] += 1
    del model
    torch.cuda.empty_cache()
    return counts, hits


def share(n: int, total: int) -> str:
    if n == 0:
        return "0"
    pct = 100 * n / total
    return f"{n:,} ({pct:.2f}%)" if pct >= 0.01 else f"{n:,} (<0.01%)"


def main() -> None:
    p = argparse.ArgumentParser(description="Count softmax collapse on the chat model's val text.")
    p.add_argument("--rows", type=int, default=0, help="only the first N rows of each val set (default all)")
    p.add_argument("--batch", type=int, default=8, help="rows per forward pass (default 8)")
    p.add_argument("--chunk", type=int, default=1024, help="targets per logits block (default 1024)")
    p.add_argument("--md", default="bench/results_collapse.md", help="the table for bench/results.py")
    args = p.parse_args()

    vocab, _ = load_bpe(f"{CHAT}/tokenizer.bpe")
    sets = [
        ("smol-smoltalk test split, answers", "SFT", f"{CHAT}/sft/weights.bin",
         lambda: sft_rows(load_tokens(f"{CHAT}/sft_val.bin"), f"{CHAT}/sft_val.tgt", args.rows)),
        ("FineWeb-Edu val", "pretrained", f"{CHAT}/pre/weights.bin",
         lambda: web_rows(load_tokens(f"{CHAT}/pretrain_val.bin"), args.rows)),
    ]
    lines = ["| validation text | model | tokens | fp32 (these kernels) | fp16 | bf16 |",
             "|---|---|---|---|---|---|"]
    collapsed = Counter()
    for text, model, weights, rows in sets:
        counts, hits = measure(weights, rows(), args.batch, args.chunk)
        n = counts["tokens"]
        print(f"{text}: {n:,} tokens, fp32 {counts['fp32']:,}, fp16 {counts['fp16']:,}, bf16 {counts['bf16']:,}")
        lines.append(f"| {text} | {model} | {n:,} | {share(counts['fp32'], n)} | "
                     f"{share(counts['fp16'], n)} | {share(counts['bf16'], n)} |")
        collapsed.update(hits)
    for (ctx, t), c in collapsed.most_common(10):
        print(f'gradient 0 in fp32: "{decode(vocab, list(ctx))}" -> "{decode(vocab, [t])}" ({c})')
    with open(args.md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {args.md}")


if __name__ == "__main__":
    main()
