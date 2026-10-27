"""The chat model's numbers for the README, read from the two train_fast logs.

    python3 bench/chat_results.py          data/chat/pre/fast_log.csv, data/chat/sft/fast_log.csv
                                           -> bench/results_chat.md (bench/results.py copies it)

Nothing here makes up a number: a log that is missing shows up as "not run yet".
"""
from __future__ import annotations

import argparse
import csv
import os

RUNS = [("pretrain (FineWeb-Edu)", "data/chat/pre/fast_log.csv"),
        ("SFT (smol-smoltalk, answers only)", "data/chat/sft/fast_log.csv")]


def read_log(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary(rows: list[dict], tokens_per_step: int, skip: int) -> dict:
    steps = [r for r in rows if r["train"]]
    vals = [(int(r["step"]), float(r["val"])) for r in rows if r["val"]]
    bpbs = [float(r["val_bpb"]) for r in rows if r.get("val_bpb") not in (None, "", "nan")]
    timed = steps[skip:] if len(steps) > skip else steps
    ms = sum(float(r["ms"]) for r in timed)
    counted = [int(r["counted"]) for r in steps if r.get("counted")]
    best_step, best = min(vals, key=lambda v: v[1])
    return {
        "steps": int(steps[-1]["step"]) if steps else 0,
        "tokens": int(steps[-1]["step"]) * tokens_per_step if steps else 0,
        "first_val": vals[0][1],
        "best_val": best,
        "best_step": best_step,
        "final_val": vals[-1][1],
        "final_bpb": bpbs[-1] if bpbs else float("nan"),
        "tok_s": len(timed) * tokens_per_step / (ms / 1000.0) if ms > 0 else float("nan"),
        "hours": sum(float(r["ms"]) for r in steps) / 3.6e6,
        "counted": sum(counted) / len(counted) if counted else 0.0,
        "skipped": sum(int(r["skipped"] or 0) for r in steps),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Write bench/results_chat.md from the chat training logs.")
    p.add_argument("--tokens-per-step", type=int, default=16 * 512, help="B x T (default 16 x 512)")
    p.add_argument("--skip", type=int, default=10, help="first steps left out of tok/s (default 10)")
    p.add_argument("--out", default="bench/results_chat.md")
    args = p.parse_args()
    lines = ["| run | steps | tokens | val at start | best val | final val | final val bpb | tok/s | GPU hours | "
             "counted per step |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, path in RUNS:
        if not os.path.exists(path):
            lines.append(f"| {name} | not run yet (`{path}`) | | | | | | | | |")
            continue
        s = summary(read_log(path), args.tokens_per_step, args.skip)
        lines.append(f"| {name} | {s['steps']:,} | {s['tokens'] / 1e6:,.0f}M | {s['first_val']:.3f} | "
                     f"{s['best_val']:.3f} (step {s['best_step']:,}) | {s['final_val']:.3f} | {s['final_bpb']:.4f} | "
                     f"{s['tok_s']:,.0f} | "
                     f"{s['hours']:.2f} | {s['counted']:,.0f} of {args.tokens_per_step:,} |")
    lines += ["", "The SFT val loss counts only the answer tokens of smol-smoltalk's test split, so it is not on the "
              "same scale as the pretraining val loss. Bits per byte leave the tags out and do not depend on the "
              "tokenizer."]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
