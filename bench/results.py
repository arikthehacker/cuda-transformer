"""Put the measured tables into the README's results section.

    python3 bench/results.py                    print the section
    python3 bench/results.py --readme README.md write it between the results markers

It only copies what the benchmarks wrote (nothing here makes up a number):
    bench/results_train.md      bash bench/bench_train.sh
    bench/results_gemm.md       ./build/bench_gemm
    bench/results_curves.md     python3 bench/curves.py      (and bench/curves.svg)
    bench/results_generate.md   ./build/generate -b -o bench/results_generate.md
A missing file shows up as "not measured yet" in its subsection.
"""
from __future__ import annotations

import argparse
import os

START = "<!-- results start: bench/results.py writes everything up to the end marker -->"
END = "<!-- results end -->"

PARTS = [
    ("training speed", "bench/results_train.md",
     "Same model (L 6, H 6, C 384), batch 16 x 256 tokens, same data. MFU = 6 x parameters x tok/s / peak."),
    ("GEMMs against cuBLAS", "bench/results_gemm.md",
     "Every GEMM shape of one training step, fp16 in and fp32 accumulate. Each cell: TFLOPS, and speed as a "
     "percent of cuBLAS (100% = as fast as cuBLAS)."),
    ("loss curves", "bench/results_curves.md",
     "PyTorch fast mode and train_fast on the same batches (reference/train.py --starts, train_fast -s)."),
    ("generation", "bench/results_generate.md",
     "Tokens per second for one sample, prompt and every cache rollover included."),
]


def section() -> str:
    out = [START, "## results", ""]
    for title, path, about in PARTS:
        out += [f"### {title}", "", about, ""]
        if title == "loss curves" and os.path.exists("bench/curves.svg"):
            out += ["![loss curves of PyTorch fast mode and train_fast](bench/curves.svg)", ""]
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out += [f.read().rstrip(), ""]
        else:
            out += [f"not measured yet (`{path}` does not exist)", ""]
    out.append(END)
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Copy the benchmark tables into the README.")
    p.add_argument("--readme", help="README to update in place (the section goes between the markers, "
                                    "or at the end if there are none)")
    args = p.parse_args()
    text = section()
    if not args.readme:
        print(text, end="")
        return
    old = open(args.readme, encoding="utf-8").read() if os.path.exists(args.readme) else ""
    if START in old and END in old:
        head, rest = old.split(START, 1)
        tail = rest.split(END, 1)[1]
        new = head + text.rstrip("\n") + tail
    else:
        new = old.rstrip("\n") + ("\n\n" if old else "") + text
    with open(args.readme, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print(f"results section -> {args.readme}")


if __name__ == "__main__":
    main()
