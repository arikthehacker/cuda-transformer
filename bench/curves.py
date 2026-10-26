"""Loss curves of PyTorch and the CUDA trainer on one plot, as an SVG (no
plotting library needed).

    python3 bench/curves.py                         data/compare/log.csv vs data/compare/fast_log.csv
    python3 bench/curves.py --torch a/log.csv --cuda b/fast_log.csv --out bench/curves.svg

For the curves to be comparable, both runs use the same batches (and their
own folder, so the real training run in data/ref stays untouched):
    python.exe -m reference.train --fast --steps 1000 --out data/compare --starts data/compare/starts.txt
    ./build/train_fast -n 1000 -s data/compare/starts.txt -o data/compare
Both logs have the columns step,train,val,... (train_fast adds scale and skipped).
"""
from __future__ import annotations

import argparse
import csv
import math
import os


def read_log(path: str) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """(step, train loss) for every step and (step, val loss) for every eval."""
    train, val = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row.get("train"):
                train.append((step, float(row["train"])))
            if row.get("val"):
                val.append((step, float(row["val"])))
    return train, val


def smooth(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    """Moving average over the last `window` steps."""
    out, total = [], 0.0
    for i, (s, v) in enumerate(points):
        total += v
        if i >= window:
            total -= points[i - window][1]
        out.append((s, total / min(i + 1, window)))
    return out


def nice_ticks(lo: float, hi: float, count: int = 6) -> list[float]:
    span = hi - lo
    raw = span / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    ticks = []
    t = first
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def svg(torch_log: str, cuda_log: str, out: str, window: int) -> str:
    runs = [("PyTorch fast", torch_log, "#3b6fb6"), ("CUDA (train_fast)", cuda_log, "#d0693a")]
    data = [(name, *read_log(path), color) for name, path, color in runs]
    steps = [s for _, train, val, _ in data for s, _ in train + val]
    losses = [v for _, train, val, _ in data for _, v in smooth(train, window) + val]
    if not steps or not losses:
        raise SystemExit("no rows in the logs")
    x0, x1 = 0, max(steps)
    y0, y1 = min(losses), max(losses)
    pad = (y1 - y0) * 0.06 or 0.1
    y0, y1 = y0 - pad, y1 + pad

    W, H = 860, 640
    left, right, top = 70, 24, 56
    ph, gap, dh = 340, 64, 120              # loss panel, space, difference panel
    pw = W - left - right
    dtop = top + ph + gap

    def px(s: float) -> float:
        return left + (s - x0) / max(1, x1 - x0) * pw

    def py(v: float) -> float:
        return top + (y1 - v) / (y1 - y0) * ph

    font = "font-family=\"Inter, 'Segoe UI', Helvetica, Arial, sans-serif\""
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{left}" y="30" {font} font-size="17" font-weight="600" fill="#1d1d1f">'
             f'Loss, PyTorch fast mode vs the CUDA trainer (same batches)</text>']

    def x_axis(y: float) -> None:
        for t in nice_ticks(x0, x1):
            parts.append(f'<text x="{px(t):.1f}" y="{y + 20}" {font} font-size="12" fill="#6e6e73" '
                         f'text-anchor="middle">{int(t):,}</text>')

    for t in nice_ticks(y0, y1):
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="#e6e6e9"/>')
        parts.append(f'<text x="{left - 8}" y="{py(t) + 4:.1f}" {font} font-size="12" fill="#6e6e73" '
                     f'text-anchor="end">{t:g}</text>')
    x_axis(top + ph)
    parts.append(f'<text x="18" y="{top + ph / 2}" {font} font-size="13" fill="#3a3a3c" text-anchor="middle" '
                 f'transform="rotate(-90 18 {top + ph / 2})">loss</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#c7c7cc"/>')

    for k, (name, train, val, color) in enumerate(data):
        line = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in smooth(train, window))
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
                     f'stroke-linejoin="round" opacity="0.9"/>')
        for s, v in val:
            if k == 0:
                parts.append(f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="4" fill="{color}"/>')
            else:
                parts.append(f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="7" fill="none" stroke="{color}" '
                             f'stroke-width="2"/>')
        ly = top + 18 + 22 * k
        lx = W - right - 300
        parts.append(f'<line x1="{lx}" x2="{lx + 24}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 32}" y="{ly + 4}" {font} font-size="13" fill="#1d1d1f">{name}</text>')
    parts.append(f'<text x="{W - right - 300}" y="{top + 18 + 22 * len(data)}" {font} font-size="12" '
                 f'fill="#6e6e73">lines: train, {window} step average; dots and rings: val</text>')

    # the difference panel: CUDA minus PyTorch, train loss averaged the same way
    a = dict(smooth(data[0][1], window))
    b = dict(smooth(data[1][1], window))
    both = sorted(set(a) & set(b))
    parts.append(f'<text x="{left}" y="{dtop - 14}" {font} font-size="13" fill="#3a3a3c">'
                 f'difference, CUDA minus PyTorch (train, {window} step average)</text>')
    parts.append(f'<rect x="{left}" y="{dtop}" width="{pw}" height="{dh}" fill="none" stroke="#c7c7cc"/>')
    if both:
        diffs = [b[s] - a[s] for s in both]
        lim = max(0.02, max(abs(d) for d in diffs)) * 1.1

        def dy(v: float) -> float:
            return dtop + (lim - v) / (2 * lim) * dh

        for t in (-lim / 1.1, 0.0, lim / 1.1):
            parts.append(f'<line x1="{left}" x2="{W - right}" y1="{dy(t):.1f}" y2="{dy(t):.1f}" '
                         f'stroke="{"#c7c7cc" if t == 0 else "#efeff2"}"/>')
            parts.append(f'<text x="{left - 8}" y="{dy(t) + 4:.1f}" {font} font-size="12" fill="#6e6e73" '
                         f'text-anchor="end">{t:+.3f}</text>')
        line = " ".join(f"{px(s):.1f},{dy(d):.1f}" for s, d in zip(both, diffs))
        parts.append(f'<polyline points="{line}" fill="none" stroke="#5b5b60" stroke-width="1.5"/>')
    x_axis(dtop + dh)
    parts.append(f'<text x="{left + pw / 2}" y="{H - 8}" {font} font-size="13" fill="#3a3a3c" '
                 f'text-anchor="middle">step</text>')
    parts.append("</svg>")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")

    report = []
    finals = {name: val[-1] for name, _, val, _ in data if val}
    for name, (s, v) in finals.items():
        report.append(f"{name}: val {v:.4f} at step {s}")
    if len(finals) == 2:
        (a_step, a), (b_step, b) = finals.values()
        if a_step == b_step:
            report.append(f"difference at step {a_step}: {100 * (b - a) / a:+.2f}% (the bar is about 1%)")
        else:
            report.append(f"the runs end at different steps ({a_step} and {b_step}), so no difference is reported")
    return "\n".join(report)


def main() -> None:
    p = argparse.ArgumentParser(description="PyTorch vs CUDA loss curves as an SVG.")
    p.add_argument("--torch", default="data/compare/log.csv",
                   help="reference/train.py log (default data/compare/log.csv)")
    p.add_argument("--cuda", default="data/compare/fast_log.csv",
                   help="train_fast log (default data/compare/fast_log.csv)")
    p.add_argument("--out", default="bench/curves.svg", help="output (default bench/curves.svg)")
    p.add_argument("--window", type=int, default=20, help="steps in the moving average (default 20)")
    p.add_argument("--md", default="bench/results_curves.md", help="the val numbers for bench/results.py")
    args = p.parse_args()
    report = svg(args.torch, args.cuda, args.out, args.window)
    print(report)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("\n".join(f"- {line}" for line in report.splitlines()) + "\n")
    print(f"-> {args.out}, {args.md}")


if __name__ == "__main__":
    main()
