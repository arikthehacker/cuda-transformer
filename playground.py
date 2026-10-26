"""Try each part of cuda-transformer on your own text: python playground.py"""
from __future__ import annotations

import math
import os
import random
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WORK = DATA / "playground"
TOKENIZER = DATA / "tokenizer.bpe"
CHECKPOINTS = [DATA / "ref" / "best.pt", DATA / "ref" / "play" / "best.pt"]
ON_WINDOWS = os.name == "nt"
CUDA_PATH = "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"

sys.path.insert(0, str(ROOT))
from reference.tokenizer import load_bpe


COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if ON_WINDOWS:
    os.system("")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def style(text: str, *codes: str) -> str:
    if not COLOR or not codes:
        return text
    return "".join(codes) + text + "\033[0m"


BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;221m"
ORANGE = "\033[38;5;209m"
CYAN = "\033[38;5;117m"
TAG = "\033[1;38;5;16;48;5;214m"
TOKEN_BACKGROUNDS = ["\033[48;5;24m", "\033[48;5;58m", "\033[48;5;89m", "\033[48;5;22m", "\033[48;5;94m"]
WHITE = "\033[38;5;255m"


def title(text: str) -> None:
    print("\n" + style(text, BOLD, CYAN))
    print(style("-" * len(text), DIM))


def note(text: str) -> None:
    print(style(text, DIM))


def problem(text: str) -> None:
    print(style(text, RED))


def ask(prompt: str, default: str = "") -> str:
    shown = f"{prompt} [{default}]: " if default else f"{prompt}: "
    try:
        answer = input(shown).strip()
    except EOFError:
        print()
        sys.exit(0)
    return answer or default


def ask_int(prompt: str, default: int, low: int, high: int) -> int:
    while True:
        answer = ask(prompt, str(default))
        if answer.lstrip("-").isdigit() and low <= int(answer) <= high:
            return int(answer)
        problem(f"  pick a whole number from {low} to {high}")


def ask_float(prompt: str, default: float, low: float, high: float) -> float:
    while True:
        answer = ask(prompt, str(default))
        try:
            value = float(answer)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        problem(f"  pick a number from {low} to {high}")


def read_text(prompt: str = "Type or paste text. An empty line ends it.") -> str:
    note(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def menu(name: str, options: list[tuple[str, str]]) -> str:
    title(name)
    for key, label in options:
        print(f"  {key}. {label}")
    keys = [k for k, _ in options]
    while True:
        choice = ask("menu choice")
        if choice in keys:
            return choice
        problem(f"  pick one of {', '.join(keys)}")


def linux_command(cmd: list[str]) -> list[str]:
    if ON_WINDOWS:
        return ["wsl", "--cd", str(ROOT), "--", "env", f"PATH={CUDA_PATH}"] + cmd
    return cmd


def run_linux(cmd: list[str], keep=None) -> tuple[int, list[str]]:
    env = dict(os.environ)
    if not ON_WINDOWS:
        env["PATH"] = CUDA_PATH + ":" + env.get("PATH", "")
    try:
        proc = subprocess.Popen(linux_command(cmd), cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        problem("  could not start " + ("WSL" if ON_WINDOWS else cmd[0]))
        return 127, []
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if keep is None or keep(line):
            print("  " + line)
    return proc.wait(), lines


def run_python(args: list[str]) -> int:
    return subprocess.call([sys.executable] + args, cwd=ROOT)


def exists(path: Path, hint: str) -> bool:
    if path.exists():
        return True
    problem(f"  {path.relative_to(ROOT).as_posix()} is missing. {hint}")
    return False


class Tokenizer:

    def __init__(self) -> None:
        self.vocab, self.tags = load_bpe(str(TOKENIZER))
        self.tag_ids = set(self.tags.values())
        self.first_tag = min(self.tag_ids) if self.tag_ids else len(self.vocab)
        self.merges = self._merges()

    @staticmethod
    def _merges() -> list[tuple[int, int]]:
        lines = TOKENIZER.read_text(encoding="utf-8").split("\n")
        count = int(lines[1].split()[1])
        return [tuple(int(x) for x in lines[2 + k].split()) for k in range(count)]

    def encode(self, text: str) -> list[int]:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "in.txt").write_text(text, encoding="utf-8", newline="")
        code, lines = run_linux(["./build/bpe", "-e", "-m", "data/tokenizer.bpe",
                                 "-i", "data/playground/in.txt", "-o", "data/playground/out.bin"],
                                keep=lambda line: False)
        if code != 0:
            raise RuntimeError("build/bpe failed: " + " ".join(lines[-2:]))
        raw = (WORK / "out.bin").read_bytes()
        magic, version, count = struct.unpack_from("<III", raw, 0)
        if magic != 0x31455042 or version != 1:
            raise RuntimeError("build/bpe wrote a file that is not BPE1")
        return list(struct.unpack_from(f"<{count}H", raw, 12))

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def visible(self, token_id: int) -> str:
        raw = self.vocab[token_id]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "<" + " ".join(f"{b:02x}" for b in raw) + ">"
        return text.replace("\t", "→").replace("\r", "␍")

    def show(self, ids: list[int]) -> None:
        out = []
        for k, token_id in enumerate(ids):
            text = self.visible(token_id)
            if token_id in self.tag_ids:
                out.append(style(text, TAG) if COLOR else f"[{text}]")
                continue
            pieces = text.split("\n")
            for j, piece in enumerate(pieces):
                if j > 0:
                    out.append(style("↵", DIM) + "\n")
                if piece:
                    box = TOKEN_BACKGROUNDS[k % len(TOKEN_BACKGROUNDS)]
                    out.append(style(piece, box, WHITE) if COLOR else f"[{piece}]")
        print("".join(out))


def tokenizer_ready() -> Tokenizer | None:
    if not exists(TOKENIZER, "Run: make run (in WSL)"):
        return None
    if not exists(ROOT / "build" / "bpe", "Run: make (in WSL)"):
        return None
    return Tokenizer()


def tokenize_text(tok: Tokenizer) -> None:
    text = read_text()
    if not text:
        return
    try:
        ids = tok.encode(text)
    except RuntimeError as err:
        problem("  " + str(err))
        return
    print()
    tok.show(ids)
    chars = len(text)
    print()
    print(f"  {chars:,} characters -> {len(ids):,} tokens"
          + (f", {chars / len(ids):.2f} characters per token" if ids else ""))
    same = tok.decode(ids) == text
    print("  decode(encode(text)) == text: " + (style("yes", GREEN) if same else style("NO", RED)))
    if ids and ask("show the token ids? (y/n)", "n").lower().startswith("y"):
        shown = ids[:200]
        print("  " + " ".join(str(i) for i in shown) + (" ..." if len(ids) > 200 else ""))


def tokenize_file(tok: Tokenizer) -> None:
    path = Path(ask("path to a text file"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        problem(f"  {path} is not a file")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        ids = tok.encode(text)
    except RuntimeError as err:
        problem("  " + str(err))
        return
    print(f"\n  {len(text):,} characters -> {len(ids):,} tokens"
          + (f", {len(text) / len(ids):.2f} characters per token" if ids else ""))
    used = len(set(ids))
    print(f"  {used:,} different tokens used out of {len(tok.vocab):,}")
    note("\n  the first 300 tokens:")
    tok.show(ids[:300])


def search_vocab(tok: Tokenizer) -> None:
    query = ask("text to look for inside tokens (or a token id)")
    if not query:
        return
    if query.isdigit():
        token_id = int(query)
        if token_id >= len(tok.vocab):
            problem(f"  ids go up to {len(tok.vocab) - 1}")
            return
        print(f"  id {token_id} = {tok.visible(token_id)!r}")
        if 256 <= token_id < 256 + len(tok.merges):
            a, b = tok.merges[token_id - 256]
            print(f"  made by merge {token_id - 256 + 1}: {tok.visible(a)!r} ({a}) + {tok.visible(b)!r} ({b})")
        elif token_id < 256:
            print("  one of the 256 byte values every tokenizer starts with")
        else:
            print("  a source tag: it starts every chunk of that source")
        return
    hits = [i for i in range(len(tok.vocab)) if query in tok.visible(i)]
    hits.sort(key=lambda i: (-len(tok.vocab[i]), i))
    print(f"  {len(hits)} tokens contain {query!r}" + (", the 30 longest:" if len(hits) > 30 else ":"))
    for i in hits[:30]:
        print(f"    {i:5d}  {tok.visible(i)!r}")


def longest_tokens(tok: Tokenizer) -> None:
    ids = sorted(range(256, 256 + len(tok.merges)), key=lambda i: -len(tok.vocab[i]))[:30]
    note("  the 30 longest tokens the tokenizer learned:")
    for i in ids:
        print(f"    {i:5d}  {len(tok.vocab[i]):2d} bytes  {tok.visible(i)!r}")
    note("\n  the first 20 merges it learned (the most common pairs):")
    for k in range(min(20, len(tok.merges))):
        a, b = tok.merges[k]
        print(f"    {256 + k:5d} = {tok.visible(a)!r} + {tok.visible(b)!r}")


def tokenizer_menu() -> None:
    tok = tokenizer_ready()
    if tok is None:
        return
    print(f"  {len(tok.merges):,} merges, {len(tok.tags)} tags ({' '.join(tok.tags)}), "
          f"{len(tok.vocab):,} ids in all")
    while True:
        choice = menu("tokenizer", [
            ("1", "tokenize text you type or paste"),
            ("2", "tokenize a text file"),
            ("3", "look up a token id, or search tokens for some text"),
            ("4", "longest tokens and first merges"),
            ("0", "back")])
        if choice == "0":
            return
        {"1": tokenize_text, "2": tokenize_file, "3": search_vocab, "4": longest_tokens}[choice](tok)


class Model:

    def __init__(self, path: Path, tok: Tokenizer) -> None:
        import torch
        from reference.config import ModelConfig
        from reference.model import GPT
        from reference.train import load_checkpoint

        self.torch = torch
        ckpt = load_checkpoint(str(path))
        self.cfg = ModelConfig(**ckpt["model_config"])
        if self.cfg.vocab_size != len(tok.vocab):
            raise ValueError(f"{path.name} was trained with {self.cfg.vocab_size} ids, the tokenizer has "
                             f"{len(tok.vocab)}. Train a new model after changing the tokenizer.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GPT(self.cfg)
        self.net.load_state_dict(ckpt["model"])
        self.net.to(self.device).eval()
        self.path = path
        self.step = ckpt["step"]
        self.best_val = ckpt["best_val"]


def find_checkpoint() -> Path | None:
    found = [p for p in CHECKPOINTS if p.exists()]
    found += sorted(p for p in (DATA / "ref").glob("**/best.pt") if p not in found)
    return found[0] if found else None


def pick_tag(tok: Tokenizer) -> int:
    names = list(tok.tags)
    for k, name in enumerate(names, 1):
        print(f"  {k}. {name}")
    return tok.tags[names[ask_int("tag", 1, 1, len(names)) - 1]]


def generate(model: Model, tok: Tokenizer) -> None:
    torch = model.torch
    tag = pick_tag(tok)
    prompt = read_text("Opening text for the model to continue (optional). An empty line ends it.")
    tokens = ask_int("new tokens", 200, 1, 2000)
    temperature = ask_float("temperature (0 = always the likeliest token)", 0.8, 0.0, 3.0)
    top_k = ask_int("top k (0 = all tokens)", 40, 0, len(tok.vocab))
    seed = ask_int("seed", random.randint(1, 9999), 0, 2**31 - 1)

    try:
        start = [tag, 10] + (tok.encode(prompt) if prompt else [])
    except RuntimeError as err:
        problem("  " + str(err))
        return
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    idx = torch.tensor([start], dtype=torch.long, device=model.device)
    out = model.net.generate(idx, tokens, temperature, top_k, generator=gen)[0, len(start):].tolist()
    stop = next((k for k, t in enumerate(out) if t in tok.tag_ids), None)
    print()
    print(style(tok.decode(start), BOLD) + tok.decode(out if stop is None else out[:stop]))
    if stop is not None:
        note(f"\n  [the model started a new chunk: {tok.visible(out[stop])}]")
    note(f"\n  seed {seed} (use it again for the same text)")


def score_text(model: Model, tok: Tokenizer) -> None:
    torch = model.torch
    tag = pick_tag(tok)
    text = read_text()
    if not text:
        return
    try:
        ids = [tag, 10] + tok.encode(text)
    except RuntimeError as err:
        problem("  " + str(err))
        return
    limit = model.cfg.n_ctx + 1
    if len(ids) > limit:
        note(f"  only the first {limit - 2} tokens fit in the model's context, scoring those")
        ids = ids[:limit]
    x = torch.tensor([ids[:-1]], device=model.device)
    with torch.no_grad():
        logits, _ = model.net(x)
    V = model.cfg.vocab_size
    probs = torch.softmax(logits[0, :, :V].float(), dim=-1)
    targets = torch.tensor(ids[1:], device=model.device)
    p = probs[torch.arange(len(targets)), targets].tolist()
    guess = probs.argmax(dim=-1).tolist()

    rows = list(zip(ids[2:], p[1:], guess[1:]))
    out = []
    for token_id, prob, _ in rows:
        color = GREEN if prob >= 0.5 else YELLOW if prob >= 0.1 else ORANGE if prob >= 0.01 else RED
        out.append(style(tok.visible(token_id), color))
    print("\n" + "".join(out))
    print("\n  " + style("likely >= 50%", GREEN) + "   " + style("10-50%", YELLOW) + "   "
          + style("1-10%", ORANGE) + "   " + style("< 1%", RED))

    loss = -sum(math.log(max(prob, 1e-12)) for _, prob, _ in rows) / len(rows)
    print(f"\n  average loss {loss:.3f} (a trained model's validation loss is around 3.8), "
          f"perplexity {math.exp(loss):.1f}")
    note("  perplexity: on average the model was as unsure as picking from that many tokens")
    print("\n  the 5 most surprising tokens:")
    for token_id, prob, g in sorted(rows, key=lambda r: r[1])[:5]:
        print(f"    {tok.visible(token_id)!r:24s} {100 * prob:7.3f}%   it expected {tok.visible(g)!r}")


REPLY_MAX = 120


def austen_line(message: str, speaker: str) -> str:
    text = message.strip()
    if text.endswith("."):
        text = text[:-1] + ","
    elif not text.endswith(("!", "?", ",")):
        text += ","
    return f"“{text}” said {speaker}."


def chat_austen(model: Model, tok: Tokenizer) -> None:
    torch = model.torch
    if "<austen>" not in tok.tags:
        problem("  this tokenizer has no <austen> tag")
        return
    tag = tok.tags["<austen>"]
    you = ask("who are you in the story", "Mr. Darcy")
    temperature = ask_float("temperature (higher = wilder replies)", 0.8, 0.05, 2.0)
    seed = random.randint(1, 9999)
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    note("  say something and press Enter. An empty line ends the chat, /reset starts over.")
    keep = model.cfg.n_ctx - 2

    history: list[int] = []
    while True:
        try:
            message = input(style("\n> ", BOLD)).strip()
        except EOFError:
            message = ""
        if not message:
            note(f"  chat ended (seed {seed})")
            return
        if message == "/reset":
            history = []
            note("  starting over")
            continue

        line = austen_line(message, you)
        note("  " + line)
        try:
            history += tok.encode(line + "\n\n“")
        except RuntimeError as err:
            problem("  " + str(err))
            return

        reply: list[int] = []
        printed = ""
        print("  “", end="", flush=True)
        for _ in range(REPLY_MAX):
            context = [tag, 10] + (history + reply)[-keep:]
            idx = torch.tensor([context], dtype=torch.long, device=model.device)
            nxt = model.net.generate(idx, 1, temperature, 40, generator=gen)[0, -1].item()
            if nxt in tok.tag_ids:
                break
            reply.append(nxt)
            text = tok.decode(reply)
            end = text.find("\n\n")
            shown = (text if end < 0 else text[:end]).rstrip("�")
            print(shown[len(printed):], end="", flush=True)
            printed = shown
            if end >= 0:
                break
        print()
        history += reply
        if "\n\n" not in tok.decode(reply):
            history += tok.encode("\n\n")


def train_quick() -> None:
    steps = ask_int("steps (fast mode, about 30 seconds per 1,000)", 1500, 50, 20000)
    note("  training in fast mode into data/ref/play/ (Ctrl+C stops it, the best checkpoint stays)")
    try:
        run_python(["-m", "reference.train", "--fast", "--steps", str(steps), "--out", "data/ref/play"])
    except KeyboardInterrupt:
        print()


def model_menu() -> None:
    tok = tokenizer_ready()
    if tok is None:
        return
    try:
        import torch
    except ImportError:
        problem("  PyTorch is not installed for this Python. Use the Python that runs reference/train.py.")
        return
    model = None
    while True:
        path = find_checkpoint()
        if path and (model is None or model.path != path):
            try:
                model = Model(path, tok)
            except ValueError as err:
                problem("  " + str(err))
                model = None
        if model:
            print(f"  {model.path.relative_to(ROOT).as_posix()}: step {model.step}, val {model.best_val:.3f}, "
                  f"on {model.device}")
        else:
            note("  no trained model yet: choose 4 to train a quick one")
        choice = menu("model", [
            ("1", "generate text from a tag (and your own opening, if you want)"),
            ("2", "score your text: which tokens surprise the model"),
            ("3", "chat with an Austen character"),
            ("4", "train a quick model (fast mode)"),
            ("0", "back")])
        if choice == "0":
            return
        if choice == "4":
            train_quick()
            model = None
        elif model is None:
            problem("  train a model first (choice 4)")
        elif choice == "1":
            generate(model, tok)
        elif choice == "2":
            score_text(model, tok)
        else:
            chat_austen(model, tok)


def cuda_status() -> None:
    code, lines = run_linux(["nvcc", "--version"], keep=lambda line: line.startswith("Cuda"))
    if code != 0:
        problem("  nvcc not found: install the CUDA Toolkit in WSL (see the README quickstart)")
        return
    run_linux(["make", "cuda-hello"], keep=lambda line: line.startswith(("GPU", "ok", "WRONG")))


def dump_ready(full: bool) -> bool:
    path = DATA / "ref" / ("debug_full.bin" if full else "debug.bin")
    if path.exists():
        return True
    if not ask(f"{path.relative_to(ROOT).as_posix()} is missing. Write it now? (y/n)", "y").lower().startswith("y"):
        return False
    return run_python(["-m", "reference.dump"] + (["--full"] if full else [])) == 0


def cuda_tests() -> None:
    if not dump_ready(False):
        return
    full = (DATA / "ref" / "debug_full.bin").exists()
    note("  building, then comparing every kernel and the whole model against the dump"
         + (" (tiny and full size)" if full else " (tiny; write the full dump to test the full size too)"))
    keep = re.compile(r"first mismatch|match|tolerance|error|^====")
    code, lines = run_linux(["make", "cuda", "test-cuda"], keep=lambda line: bool(keep.search(line)))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "test_cuda.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    number = r"\d\.\d{3}e[+-]\d{2}"
    rows = sum(1 for line in lines if re.search(number + r"\s+" + number, line))
    print(("  " + style("passed", GREEN) if code == 0 else "  " + style("FAILED", RED))
          + f", {rows} rows compared. Every row is in data/playground/test_cuda.txt")


def cuda_train() -> None:
    if not dump_ready(True):
        return
    steps = ask_int("steps (about 8 seconds per 100 at the full size)", 300, 10, 5000)
    every = max(steps // 4, 1)
    WORK.mkdir(parents=True, exist_ok=True)
    run_linux(["make", "cuda"], keep=lambda line: "error" in line)
    run_linux(["./build/train_gpt", "-n", str(steps), "-e", str(every), "-o", "data/playground"])


def cuda_menu() -> None:
    while True:
        choice = menu("CUDA", [
            ("1", "is CUDA set up? (nvcc and the GPU)"),
            ("2", "run the CUDA tests against the answer key"),
            ("3", "train with the CUDA code for a few hundred steps"),
            ("0", "back")])
        if choice == "0":
            return
        {"1": cuda_status, "2": cuda_tests, "3": cuda_train}[choice]()


def data_sizes() -> None:
    clean = sorted((DATA / "clean").glob("*.txt"))
    if not clean:
        problem("  no data/clean/*.txt yet. Run: python prepare_data.py")
        return
    print("  source        characters")
    for path in clean:
        print(f"  {path.stem:12s} {len(path.read_text(encoding='utf-8')):12,d}")
    for name in ("corpus_train.txt", "corpus_val.txt"):
        path = DATA / name
        if path.exists():
            print(f"  {name:20s} {path.stat().st_size:12,d} bytes")
    for name in ("train.bin", "val.bin"):
        path = DATA / name
        if path.exists():
            count = struct.unpack_from("<I", path.read_bytes()[:12], 8)[0]
            print(f"  {name:20s} {count:12,d} tokens")


def random_chunk() -> None:
    path = DATA / "corpus_train.txt"
    if not exists(path, "Run: python build_corpus.py"):
        return
    text = path.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r"(?m)^<[a-z_]+>$", text)]
    if not starts:
        problem("  no tagged chunks found")
        return
    k = random.randrange(len(starts))
    end = starts[k + 1] if k + 1 < len(starts) else len(text)
    chunk = text[starts[k]:end].rstrip("\n")
    tag, _, body = chunk.partition("\n")
    print("\n" + style(tag, TAG if COLOR else "") + "\n" + body)


def data_menu() -> None:
    while True:
        choice = menu("training data", [
            ("1", "sources and sizes"),
            ("2", "a random chunk from the training text"),
            ("0", "back")])
        if choice == "0":
            return
        {"1": data_sizes, "2": random_chunk}[choice]()


def main() -> None:
    os.chdir(ROOT)
    print(style("cuda-transformer playground", BOLD))
    note("try each part on your own text. 0 goes back, Ctrl+C quits.")
    while True:
        choice = menu("main menu", [
            ("1", "tokenizer       see how your text splits into tokens"),
            ("2", "model           generate text, and see what surprises the model"),
            ("3", "CUDA            the CUDA tests and a short CUDA training run"),
            ("4", "training data   sources, sizes and a random chunk"),
            ("0", "quit")])
        if choice == "0":
            return
        {"1": tokenizer_menu, "2": model_menu, "3": cuda_menu, "4": data_menu}[choice]()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
