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


LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "band": "#f0efec", "s1": "#2a78d6", "s2": "#eb6834"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "band": "#262624", "s1": "#3987e5", "s2": "#d95926"}
STYLE = ("text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:12px;fill:var(--muted);"
         "font-variant-numeric:tabular-nums}"
         ".title{font-size:16px;font-weight:600;fill:var(--ink)}.sub{font-size:13px;fill:var(--ink2)}"
         ".key,.end{font-size:13px;fill:var(--ink)}.bg{fill:var(--surface)}"
         ".grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}.band{fill:var(--band)}"
         ".wide{fill:none;stroke:var(--s1);stroke-width:7;stroke-opacity:.35;stroke-linejoin:round;stroke-linecap:round}"
         ".thin{fill:none;stroke:var(--s2);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
         ".diff{fill:none;stroke:var(--ink2);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
         ".dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}.ring{fill:none;stroke:var(--s2);stroke-width:2}")


def svg(torch_log: str, cuda_log: str, out: str, window: int) -> str:
    runs = [("PyTorch fast", torch_log), ("CUDA (train_fast)", cuda_log)]
    data = [(name, *read_log(path)) for name, path in runs]
    steps = [s for _, train, val in data for s, _ in train + val]
    losses = [v for _, train, val in data for _, v in smooth(train, window) + val]
    if not steps or not losses:
        raise SystemExit("no rows in the logs")
    x0, x1 = 0, max(steps)
    y0, y1 = min(losses), max(losses)
    pad = (y1 - y0) * 0.05 or 0.1
    y0, y1 = y0 - pad, y1 + pad

    W, left, right = 860, 48, 92
    top, ph = 86, 280                       # the loss panel
    dtop, dh = top + ph + 82, 110           # the difference panel
    H = dtop + dh + 52
    pw = W - left - right

    def px(s: float) -> float:
        return left + (s - x0) / max(1, x1 - x0) * pw

    def py(v: float) -> float:
        return top + (y1 - v) / (y1 - y0) * ph

    def tokens(c: dict) -> str:
        return "".join(f"--{k}:{v};" for k, v in c.items())

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f"<style>svg{{{tokens(LIGHT)}}}@media (prefers-color-scheme: dark){{svg{{{tokens(DARK)}}}}}"
             f"{STYLE}</style>",
             f'<rect class="bg" width="{W}" height="{H}" rx="12"/>',
             f'<text class="title" x="{left}" y="34">Loss of PyTorch fast mode and train_fast on the same batches</text>']

    keys = [("wide", "PyTorch fast mode, train loss"), ("thin", "train_fast, train loss")]
    kx = left
    for cls, label in keys:
        parts.append(f'<line class="{cls}" x1="{kx}" x2="{kx + 22}" y1="58" y2="58"/>')
        parts.append(f'<text class="key" x="{kx + 30}" y="62">{label}</text>')
        kx += 54 + 7.2 * len(label)
    parts.append(f'<circle class="dot" cx="{kx + 4}" cy="58" r="4.5"/>')
    parts.append(f'<circle class="ring" cx="{kx + 4}" cy="58" r="7.5"/>')
    parts.append(f'<text class="key" x="{kx + 18}" y="62">val loss, both</text>')

    for t in nice_ticks(y0, y1, 5):
        parts.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>')
        parts.append(f'<text x="{left - 8}" y="{py(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{W - right}" y1="{top + ph}" y2="{top + ph}"/>')

    def x_axis(y: float) -> None:
        for t in nice_ticks(x0, x1):
            parts.append(f'<text x="{px(t):.1f}" y="{y + 18}" text-anchor="middle">{int(t):,}</text>')

    x_axis(top + ph)
    (_, a_train, a_val), (_, b_train, b_val) = data
    for cls, train in (("wide", a_train), ("thin", b_train)):
        line = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in smooth(train, window))
        parts.append(f'<polyline class="{cls}" points="{line}"/>')
    for s, v in a_val:
        parts.append(f'<circle class="dot" cx="{px(s):.1f}" cy="{py(v):.1f}" r="4.5"/>')
    for s, v in b_val:
        parts.append(f'<circle class="ring" cx="{px(s):.1f}" cy="{py(v):.1f}" r="7.5"/>')
    if a_val and b_val:
        (sa, va), (sb, vb) = a_val[-1], b_val[-1]
        x = px(max(sa, sb)) + 14
        if f"{va:.3f}" == f"{vb:.3f}":
            parts.append(f'<text class="end" x="{x:.1f}" y="{py(vb) + 4:.1f}">{vb:.3f}</text>')
        else:
            mid = (py(va) + py(vb)) / 2
            ya, yb = (py(va), py(vb)) if abs(py(va) - py(vb)) >= 16 else (mid - 8, mid + 8)
            for cls, r, v, y in (("dot", 3.5, va, ya), ("ring", 5, vb, yb)):
                parts.append(f'<circle class="{cls}" cx="{x + 5:.1f}" cy="{y:.1f}" r="{r}"/>')
                parts.append(f'<text class="end" x="{x + 15:.1f}" y="{y + 4:.1f}">{v:.3f}</text>')

    # the difference panel: train_fast minus PyTorch as a percent of PyTorch, against the 1% bar
    a = dict(smooth(a_train, window))
    b = dict(smooth(b_train, window))
    both = sorted(set(a) & set(b))
    parts.append(f'<text class="sub" x="{left}" y="{dtop - 18}">train_fast minus PyTorch, as a percent of the '
                 f'PyTorch loss ({window} step averages). The band is the 1% bar.</text>')
    if both:
        diffs = [100 * (b[s] - a[s]) / a[s] for s in both]
        lim = max(1.5, max(abs(d) for d in diffs) * 1.15)

        def dy(v: float) -> float:
            return dtop + (lim - v) / (2 * lim) * dh

        parts.append(f'<rect class="band" x="{left}" y="{dy(1):.1f}" width="{pw}" height="{dy(-1) - dy(1):.1f}"/>')
        for t, label in ((1, "+1%"), (0, "0"), (-1, "−1%")):
            parts.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{left}" x2="{W - right}" '
                         f'y1="{dy(t):.1f}" y2="{dy(t):.1f}"/>')
            parts.append(f'<text x="{left - 8}" y="{dy(t) + 4:.1f}" text-anchor="end">{label}</text>')
        line = " ".join(f"{px(s):.1f},{dy(d):.1f}" for s, d in zip(both, diffs))
        parts.append(f'<polyline class="diff" points="{line}"/>')
        last = round(diffs[-1], 2)
        label = f"{last:+.2f}%" if last else "0.00%"
        parts.append(f'<text class="end" x="{px(both[-1]) + 14:.1f}" y="{dy(diffs[-1]) + 4:.1f}">{label}</text>')
    x_axis(dtop + dh)
    parts.append(f'<text x="{left + pw / 2}" y="{H - 12}" text-anchor="middle">step</text>')
    parts.append("</svg>")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts) + "\n")

    report = []
    finals = {name: val[-1] for name, _, val in data if val}
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
    with open(args.md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(f"- {line}" for line in report.splitlines()) + "\n")
    print(f"-> {args.out}, {args.md}")


if __name__ == "__main__":
    main()
