"""Train the model on your own writing, every step in order: python pipeline.py --help"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from prepare_data import FILE_TYPES, LOCAL_SOURCES, LOOSE_FILES_TAG, MY_WRITING, PUBLIC, read_local_sources, tag_name

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINT = DATA / "ref" / "best.pt"
ON_WINDOWS = os.name == "nt"
CUDA_PATH = "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
MIN_CHARS = 20_000
SMALL_CHARS = 300_000
DEFAULT_STEPS = 5000


def linux_command(cmd: list[str]) -> list[str]:
    if ON_WINDOWS:
        return ["wsl", "--cd", str(ROOT), "--", "env", f"PATH={CUDA_PATH}"] + cmd
    return cmd


def vocab_for(chars: int) -> int:
    if chars >= 2_000_000:
        return 4096
    if chars >= 500_000:
        return 2048
    return 1024


def size_text(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


@dataclass
class Options:
    public: bool = True
    steps: int = DEFAULT_STEPS
    exact: bool = False


@dataclass(frozen=True)
class Source:
    tag: str
    files: int
    size: int
    where: str


class StepFailed(Exception):
    pass


class Runner:

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self.on_line = on_line
        self.proc: subprocess.Popen | None = None
        self.stopped = False

    def run(self, cmd: list[str], linux: bool = False) -> None:
        if self.stopped:
            raise StepFailed("stopped")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if ON_WINDOWS else 0
        try:
            self.proc = subprocess.Popen(linux_command(cmd) if linux else cmd, cwd=ROOT, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", creationflags=flags)
        except FileNotFoundError:
            raise StepFailed(f"could not start {'WSL' if linux and ON_WINDOWS else cmd[0]}")
        for line in self.proc.stdout:
            self.on_line(line.rstrip("\n"))
        code = self.proc.wait()
        self.proc = None
        if self.stopped:
            raise StepFailed("stopped")
        if code != 0:
            raise StepFailed(f"{' '.join(cmd[:3])} exited with code {code}")

    def stop(self) -> None:
        self.stopped = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def corpus_chars() -> int:
    path = DATA / "corpus_train.txt"
    if not path.exists():
        raise StepFailed("data/corpus_train.txt is missing, so run the mix step first")
    with open(path, encoding="utf-8") as f:
        return sum(len(line) for line in f)


def gather(runner: Runner, opts: Options) -> None:
    runner.run([sys.executable, "prepare_data.py"] + ([] if opts.public else ["--no-public"]))


def mix(runner: Runner, opts: Options) -> None:
    runner.run([sys.executable, "build_corpus.py"])


def tokenize(runner: Runner, opts: Options) -> None:
    chars = corpus_chars()
    if chars < MIN_CHARS:
        raise StepFailed(f"only {chars:,} characters of training text. Add more writing, at least {MIN_CHARS:,}.")
    if chars < SMALL_CHARS:
        runner.on_line(f"note: {chars:,} characters is not much, so expect the model to repeat your text back")
    vocab = vocab_for(chars)
    runner.run(["make", "build/bpe"], linux=True)
    runner.run(["./build/bpe", "-t", "-i", "data/corpus_train.txt", "-n", str(vocab), "-o", "data/tokenizer.bpe", "-v"],
               linux=True)
    for split in ("train", "val"):
        runner.run(["./build/bpe", "-e", "-m", "data/tokenizer.bpe", "-i", f"data/corpus_{split}.txt",
                    "-o", f"data/{split}.bin"], linux=True)


def train(runner: Runner, opts: Options) -> None:
    runner.run([sys.executable, "-m", "reference.train", "--steps", str(opts.steps)]
               + ([] if opts.exact else ["--fast"]))


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    work: Callable[[Runner, Options], None]


STEPS = [
    Step("gather", "gather your writing", gather),
    Step("mix", "mix it into training text", mix),
    Step("tokenize", "train the tokenizer", tokenize),
    Step("train", "train the model", train),
]
STEP_KEYS = [s.key for s in STEPS]
TRAIN_LINE = re.compile(r"^step\s+(\d+) \|")
MERGE_LINE = re.compile(r"^merge\s+(\d+) / (\d+)")


def progress(key: str, line: str, opts: Options) -> float | None:
    if key == "train" and (m := TRAIN_LINE.match(line)):
        return min(int(m.group(1)) / opts.steps, 1.0)
    if key == "tokenize" and (m := MERGE_LINE.match(line)):
        return int(m.group(1)) / int(m.group(2))
    return None


def build(runner: Runner, opts: Options, start: str = "gather",
          on_step: Callable[[Step, str, float], None] = lambda step, state, secs: None) -> bool:
    for step in STEPS[STEP_KEYS.index(start):]:
        on_step(step, "running", 0.0)
        began = time.perf_counter()
        try:
            step.work(runner, opts)
        except StepFailed as err:
            runner.on_line(f"error: {err}")
            on_step(step, "stopped" if runner.stopped else "failed", time.perf_counter() - began)
            return False
        on_step(step, "done", time.perf_counter() - began)
    return True


def sources() -> list[Source]:
    found = []
    if MY_WRITING.is_dir():
        loose = [p for p in MY_WRITING.iterdir() if p.is_file() and p.suffix.lower() in FILE_TYPES]
        if loose:
            found.append(Source(LOOSE_FILES_TAG, len(loose), sum(p.stat().st_size for p in loose), "my_writing/"))
        for folder in sorted(p for p in MY_WRITING.iterdir() if p.is_dir()):
            files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in FILE_TYPES]
            found.append(Source(tag_name(folder.name), len(files), sum(p.stat().st_size for p in files),
                                f"my_writing/{folder.name}/"))
    for name, (entries, skip) in read_local_sources().items():
        files = [f for f in listed_files(entries) if f.name not in skip]
        online = any(e.startswith("wattpad:") for e in entries)
        found.append(Source(name, len(files), sum(f.stat().st_size for f in files),
                            LOCAL_SOURCES.name + (" (and Wattpad)" if online else "")))
    return found


def listed_files(entries: list[str]) -> list[Path]:
    files = []
    for entry in entries:
        if entry.startswith("wattpad:"):
            continue
        path = Path(entry.replace("~", str(Path.home()), 1))
        candidates = path.rglob("*") if path.is_dir() else [path] if path.is_file() else []
        files += [f for f in candidates if f.is_file() and f.suffix.lower() in FILE_TYPES]
    return files


def unique_path(path: Path) -> Path:
    k = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({k}){path.suffix}")
        k += 1
    return candidate


def same_file(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def add(tag: str, paths: list[Path]) -> tuple[str, int, list[str]]:
    name = tag_name(tag)
    if not name:
        raise ValueError(f"{tag!r} has no letters to make a tag from")
    if name in PUBLIC:
        raise ValueError(f"{name} is one of the public sources, so pick another name")
    folder = MY_WRITING / name
    copied, skipped = 0, []
    for path in paths:
        if path.is_dir():
            pairs = [(f, folder / path.name / f.relative_to(path)) for f in sorted(path.rglob("*")) if f.is_file()]
        else:
            pairs = [(path, folder / path.name)]
        for src, dest in pairs:
            if src.suffix.lower() not in FILE_TYPES:
                skipped.append(src.name)
                continue
            if dest.exists() and same_file(src, dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, unique_path(dest))
            copied += 1
    return name, copied, skipped


def remove(tag: str) -> bool:
    folder = MY_WRITING / tag_name(tag)
    if folder.is_dir():
        shutil.rmtree(folder)
        return True
    if tag_name(tag) == LOOSE_FILES_TAG and MY_WRITING.is_dir():
        loose = [p for p in MY_WRITING.iterdir() if p.is_file() and p.suffix.lower() in FILE_TYPES]
        for p in loose:
            p.unlink()
        return bool(loose)
    return False


def checks(opts: Options) -> list[tuple[bool, str, str]]:
    def has(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    results = [
        (has("numpy"), "numpy", "pip install numpy"),
        (has("torch"), "PyTorch", "pip install torch (see pytorch.org for the GPU build)"),
    ]
    pdfs = MY_WRITING.is_dir() and any(p.suffix.lower() == ".pdf" for p in MY_WRITING.rglob("*"))
    if pdfs:
        results.append((has("pypdf"), "pypdf, for the PDFs in my_writing/", "pip install pypdf"))
    try:
        tools = subprocess.run(linux_command(["sh", "-c", "command -v g++ && command -v make"]), cwd=ROOT,
                               capture_output=True, text=True, timeout=60,
                               creationflags=subprocess.CREATE_NO_WINDOW if ON_WINDOWS else 0)
        tools_ok = tools.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tools_ok = False
    fix = "install WSL, then in it: sudo apt-get install -y build-essential" if ON_WINDOWS \
        else "sudo apt-get install -y build-essential"
    results.append((tools_ok, "g++ and make" + (" in WSL" if ON_WINDOWS else ""), fix))
    has_writing = any(s.files for s in sources())
    results.append((opts.public or has_writing, "some text to train on",
                    "put your files in my_writing/, or train on the public texts too"))
    return results


BOLD = "\033[1m" if sys.stdout.isatty() else ""
DIM = "\033[2m" if sys.stdout.isatty() else ""
GREEN = "\033[38;5;114m" if sys.stdout.isatty() else ""
RED = "\033[38;5;203m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""


def print_sources() -> None:
    found = sources()
    if not found:
        print(f"no writing yet. Put files in my_writing/ (one folder per style) or run:\n"
              f"  python pipeline.py add essays path/to/your/essays")
        return
    width = max(len(s.tag) + 2 for s in found) + 2
    print(f"{'tag':{width}s} {'files':>6s} {'size':>10s}  from")
    for s in found:
        size = size_text(s.size) if s.size else ""
        print(f"<{s.tag}>".ljust(width) + f" {s.files:6d} {size:>10s}  {s.where}")


def print_built() -> None:
    for label, path in (("training text", DATA / "corpus_train.txt"), ("tokenizer", DATA / "tokenizer.bpe"),
                        ("token file", DATA / "train.bin"), ("model", CHECKPOINT)):
        if path.exists():
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
            print(f"  {GREEN}done{RESET}     {label:14s} {path.relative_to(ROOT).as_posix()}  ({when})")
        else:
            print(f"  {DIM}missing  {label:14s} {path.relative_to(ROOT).as_posix()}{RESET}")


def cli_status(args: argparse.Namespace) -> int:
    print_sources()
    print()
    print_built()
    return 0


def cli_add(args: argparse.Namespace) -> int:
    paths = [Path(p).expanduser() for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"{RED}not found:{RESET} {', '.join(missing)}")
        return 1
    try:
        name, copied, skipped = add(args.tag, paths)
    except ValueError as err:
        print(f"{RED}{err}{RESET}")
        return 1
    print(f"copied {copied} file{'s' * (copied != 1)} into my_writing/{name}/, tag <{name}>")
    if skipped:
        shown = ", ".join(skipped[:5]) + (f" and {len(skipped) - 5} more" if len(skipped) > 5 else "")
        print(f"{DIM}skipped {len(skipped)} that are not {' '.join(FILE_TYPES)}: {shown}{RESET}")
    return 0


def cli_remove(args: argparse.Namespace) -> int:
    name = tag_name(args.tag)
    if not args.yes:
        answer = input(f"delete the copies in my_writing/ for <{name}>? Your original files stay. (y/n) ")
        if not answer.strip().lower().startswith("y"):
            return 1
    if remove(name):
        print(f"removed <{name}>")
        return 0
    print(f"{RED}no my_writing/ folder for <{name}>{RESET}")
    return 1


def cli_check(args: argparse.Namespace) -> int:
    ok = True
    for passed, label, fix in checks(Options(public=not args.no_public)):
        mark = f"{GREEN}ok{RESET}  " if passed else f"{RED}no{RESET}  "
        print(mark + label + ("" if passed else f"{DIM}   fix: {fix}{RESET}"))
        ok &= passed
    return 0 if ok else 1


def cli_build(args: argparse.Namespace) -> int:
    opts = Options(public=not args.no_public, steps=args.steps, exact=args.exact)
    failed = [(label, fix) for passed, label, fix in checks(opts) if not passed]
    if failed:
        for label, fix in failed:
            print(f"{RED}missing {label}{RESET}   fix: {fix}")
        return 1
    runner = Runner(lambda line: print("  " + line, flush=True))
    count = len(STEPS)

    def on_step(step: Step, state: str, secs: float) -> None:
        n = STEP_KEYS.index(step.key) + 1
        if state == "running":
            print(f"\n{BOLD}[{n}/{count}] {step.title}{RESET}", flush=True)
        elif state == "done":
            print(f"{GREEN}done{RESET} in {secs:.1f} s", flush=True)
        else:
            print(f"{RED}{state}{RESET} after {secs:.1f} s", flush=True)

    try:
        ok = build(runner, opts, args.start, on_step)
    except KeyboardInterrupt:
        runner.stop()
        print("\nstopped")
        return 130
    if ok:
        print(f"\n{BOLD}your model is ready{RESET}: python playground.py, or python gui.py")
    return 0 if ok else 1


def cli_gui(args: argparse.Namespace) -> int:
    import gui

    gui.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if ON_WINDOWS:
        os.system("")
    p = argparse.ArgumentParser(prog="pipeline.py", description="Train the model on your own writing.")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status", help="list your writing and what is built").set_defaults(run=cli_status)

    a = sub.add_parser("add", help="copy files or folders into my_writing/TAG/")
    a.add_argument("tag", help="the style these files are, like essays or stories")
    a.add_argument("paths", nargs="+", help="files or folders (.pdf .docx .txt .md)")
    a.set_defaults(run=cli_add)

    r = sub.add_parser("remove", help="delete my_writing/TAG/ (your original files stay)")
    r.add_argument("tag")
    r.add_argument("-y", "--yes", action="store_true", help="do not ask first")
    r.set_defaults(run=cli_remove)

    c = sub.add_parser("check", help="check that everything the build needs is installed")
    c.add_argument("--no-public", action="store_true", help="the build will use only your writing")
    c.set_defaults(run=cli_check)

    b = sub.add_parser("build", help="gather, mix, train the tokenizer, train the model")
    b.add_argument("--no-public", action="store_true", help="leave out the Bible and Austen")
    b.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"most training steps, it stops early once it stops improving (default {DEFAULT_STEPS})")
    b.add_argument("--exact", action="store_true", help="train in exact fp32 mode instead of fast fp16 mode")
    b.add_argument("--from", dest="start", choices=STEP_KEYS, default="gather", help="start at this step")
    b.set_defaults(run=cli_build)

    sub.add_parser("gui", help="open the window version").set_defaults(run=cli_gui)

    args = p.parse_args(argv)
    if not args.command:
        cli_status(args)
        print(f"\n{DIM}add writing:  python pipeline.py add TAG FILES...\n"
              f"build:        python pipeline.py build\n"
              f"window:       python gui.py{RESET}")
        return 0
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
