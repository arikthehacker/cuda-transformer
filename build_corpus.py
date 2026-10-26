"""Mix the clean sources into one training file and one validation file.

Each chunk of text starts with a tag line naming its source, e.g. "<bible>".
Start generation with a tag and the model writes in that style.

Smaller sources are repeated so they aren't drowned out by the Bible.
The last 10% of each source is held out for validation and never repeated,
so validation loss measures text the model has not seen.

Run after prepare_data.py:  python build_corpus.py
Outputs: data/corpus_train.txt, data/corpus_val.txt
"""
import random
from pathlib import Path

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "clean"

# How many times each source appears in the training file.
REPEATS = {"bible": 1, "austen": 1}
LOCAL_REPEATS = 4  # for each of your own sources, when the public texts are in too
CHUNK_CHARS = 1000  # paragraphs are grouped into chunks of about this size
VAL_FRACTION = 0.1


def chunks(text):
    """Group paragraphs into chunks of roughly CHUNK_CHARS characters."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def main():
    rng = random.Random(42)
    train, val = [], []
    local = sorted(p.stem for p in CLEAN.glob("*.txt") if p.stem not in REPEATS)
    public = [name for name in REPEATS if (CLEAN / f"{name}.txt").exists()]
    local_reps = LOCAL_REPEATS if public else 1
    for name in public + local:
        reps = REPEATS.get(name, local_reps)
        path = CLEAN / f"{name}.txt"
        text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        if not text:
            print(f"{name:12s} empty, skipped")
            continue
        parts = chunks(text)
        cut = int(len(parts) * (1 - VAL_FRACTION))
        tagged = [f"<{name}>\n{c}\n\n" for c in parts]
        train += tagged[:cut] * reps
        val += tagged[cut:]
        print(f"{name:12s} {len(parts):5d} chunks  x{reps} in train")
    rng.shuffle(train)
    rng.shuffle(val)
    for split, items in (("train", train), ("val", val)):
        text = "".join(items)
        (ROOT / "data" / f"corpus_{split}.txt").write_text(text, encoding="utf-8", newline="\n")
        print(f"data/corpus_{split}.txt: {len(text):,} chars")
    vocab = sorted(set("".join(train + val)))
    print(f"distinct characters: {len(vocab)}")


if __name__ == "__main__":
    main()
