"""The learning rate x beta2 sweep as a table (bench/sweep_lr.sh writes the runs).

    python3 bench/sweep_summary.py         data/chat/sweep/*/fast_log.csv -> bench/results_sweep.md

For each run: the final val loss and val bits per byte, the difference from
the baseline (lr 6e-4, beta2 0.95), the first step's train loss (the same in
every run when the batches and the init are the same) and the skipped steps.
The baseline ran twice; the gap between those two runs is the noise floor, and
a difference smaller than about 2x that is a tie.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

BASE = "6e-4_0.95"
AGAIN = "6e-4_0.95_again"


def read_run(path: str) -> dict | None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = [r for r in rows if r.get("train")]
    evals = [r for r in rows if r.get("val")]
    if not steps or not evals:
        return None
    last = evals[-1]
    bpb = last.get("val_bpb") or ""
    return {
        "steps": int(steps[-1]["step"]),
        "val": float(last["val"]),
        "bpb": float(bpb) if bpb not in ("", "nan") else float("nan"),
        "first_train": float(steps[0]["train"]),
        "skipped": sum(int(r.get("skipped") or 0) for r in steps),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize the lr x beta2 sweep.")
    p.add_argument("--dir", default="data/chat/sweep", help="the sweep's runs (default data/chat/sweep)")
    p.add_argument("--out", default="bench/results_sweep.md")
    args = p.parse_args()

    runs = {}
    for log in sorted(glob.glob(os.path.join(args.dir, "*", "fast_log.csv"))):
        name = os.path.basename(os.path.dirname(log))
        r = read_run(log)
        if r:
            runs[name] = r
    if BASE not in runs:
        raise SystemExit(f"no finished baseline run {args.dir}/{BASE} yet")
    base = runs[BASE]
    noise = abs(runs[AGAIN]["bpb"] - base["bpb"]) if AGAIN in runs else float("nan")
    steps = max(r["steps"] for r in runs.values())

    lines = [f"Every run: {steps:,} steps of 16 x 512 tokens ({steps * 8192 / 1e6:,.0f}M tokens), the same batches "
             f"and init, warmup 2,000, WSD with the last 20% decaying, min lr = lr / 10.", "",
             "| lr | beta2 | final val loss | final val bpb | bpb vs baseline | first train loss | skipped steps |",
             "|---|---|---|---|---|---|---|"]
    order = sorted((n for n in runs if n != AGAIN), key=lambda n: (float(n.split("_")[0]), float(n.split("_")[1])))
    if AGAIN in runs:
        order.append(AGAIN)
    best = min((n for n in runs if n != AGAIN), key=lambda n: runs[n]["bpb"])
    for name in order:
        r = runs[name]
        lr, beta2 = name.split("_")[:2]
        diff = r["bpb"] - base["bpb"]
        tie = noise == noise and abs(diff) < 2 * noise and name not in (BASE, AGAIN)
        note = " (A/A: the baseline again)" if name == AGAIN else (" (baseline)" if name == BASE else "")
        mark = " **best**" if name == best else ""
        lines.append(f"| {lr} | {beta2} | {r['val']:.4f} | {r['bpb']:.4f}{mark} | {diff:+.4f}{' tie' if tie else ''}"
                     f"{note} | {r['first_train']:.6f} | {r['skipped']} |")
    lines += ["", f"Noise floor (the two baseline runs): {noise:.4f} bpb. A difference under about twice that is "
                  f"a tie." if noise == noise else "Noise floor: the A/A run has not finished."]
    firsts = {round(r["first_train"], 6) for r in runs.values()}
    lines.append("Every run's first step had the same train loss, so the batches and the init match."
                 if len(firsts) == 1 else
                 f"The first step's train loss differs between runs ({len(firsts)} values): check the batches and init.")
    if best.startswith("2e-3"):
        lines.append("The best lr is the largest one tried: add 3e-3 "
                     "(LRS=3e-3 AA=0 bash bench/sweep_lr.sh) before deciding.")
    lines.append(f"Best: lr {best.split('_')[0]}, beta2 {best.split('_')[1]}.")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
