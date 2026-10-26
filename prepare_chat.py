"""Stage 6 data: FineWeb-Edu for pretraining, smol-smoltalk for chat fine-tuning,
both written as tagged text the BPE tool reads.

    python.exe prepare_chat.py --max-chars 20000000      a 20 MB test run first
    python.exe prepare_chat.py                           about 4.5 GB of web text + all of smol-smoltalk
    python.exe prepare_chat.py --skip-web                only the conversations
    python.exe prepare_chat.py --join                    after encoding: pretrain_NNN.bin -> pretrain_train.bin

Needs: pip install pyarrow huggingface_hub   (the Windows Python, the one with torch)

Writes, in data/chat/:
    pretrain_000.txt, pretrain_001.txt ...   about 256 MB each, every document as "<web>\\n" + text + "\\n\\n"
    pretrain_val.txt                         one document in every 100, held out
    sft_train.txt, sft_val.txt               "<user>\\n" ... "<assistant>\\n" ... "<user>\\n", a blank line between
    raw/                                     the downloaded parquet files (cached, safe to delete afterwards)

Sources (credit both in the README):
    HuggingFaceFW/fineweb-edu, sample/10BT    ODC-By 1.0
    HuggingFaceTB/smol-smoltalk                Apache 2.0
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import struct
import sys
from collections import Counter
from typing import Iterable, Iterator

FINEWEB = "HuggingFaceFW/fineweb-edu"
FINEWEB_DIR = "sample/10BT/"
SMOLTALK = "HuggingFaceTB/smol-smoltalk"
OUT = "data/chat"
RAW = os.path.join(OUT, "raw")
HOLD_OUT_EVERY = 100            # document k goes to val when k % 100 == 99

# The tokenizer makes a special token out of any line that is exactly "<" + [a-z_]+ ">"
# (tokenizer/bpe.cpp is_tag), and adds a new one for every such line it sees while
# training. Web pages have lines like "<div>" or "<br>", so those get one space in front.
TAG_LINE = re.compile(r"^<[a-z_]+>$")


# ---------------------------------------------------------------- text

def tidy(text: str) -> str:
    """Line endings to \\n, no \\x1e (generate --chat prints it alone on a line
    between answers), no trailing spaces on lines, no blank lines at either end."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x1e", "")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip("\n")


def safe_lines(text: str) -> str:
    """Every tag-shaped line gets one leading space, so it stays plain text."""
    return "\n".join(" " + line if TAG_LINE.match(line) else line for line in text.split("\n"))


def clean(text: str) -> str:
    return safe_lines(tidy(text))


# ---------------------------------------------------------------- the Hugging Face side

def list_files(repo: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(repo, repo_type="dataset")


def list_shards() -> list[str]:
    """FineWeb-Edu sample-10BT parquet files, read from the listing (never hardcoded)."""
    names = [f for f in list_files(FINEWEB) if f.startswith(FINEWEB_DIR) and f.endswith(".parquet")]
    if not names:
        raise SystemExit(f"no parquet files under {FINEWEB_DIR} in {FINEWEB}")
    return sorted(names)


def download(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", local_dir=RAW)


def parquet_column(path: str, column: str, batch_rows: int = 2048) -> Iterator:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for batch in f.iter_batches(columns=[column], batch_size=batch_rows):
        yield from batch.column(0).to_pylist()


def web_documents() -> Iterator[str]:
    """FineWeb-Edu documents in shard order. Shards download one at a time, as needed."""
    for name in list_shards():
        print(f"  shard {name}", flush=True)
        yield from parquet_column(download(FINEWEB, name), "text")


def smoltalk_conversations(split: str) -> Iterator[list[dict]]:
    """smol-smoltalk conversations of one split ("train" or "test"), in file order."""
    names = sorted(f for f in list_files(SMOLTALK)
                   if f.endswith(".parquet") and os.path.basename(f).startswith(split + "-"))
    if not names:
        raise SystemExit(f"no {split} parquet files in {SMOLTALK}")
    for name in names:
        print(f"  {name}", flush=True)
        yield from parquet_column(download(SMOLTALK, name), "messages")


# ---------------------------------------------------------------- pretraining text

class ShardWriter:
    """Text files of about shard_chars characters each: prefix_000.txt, prefix_001.txt ..."""

    def __init__(self, prefix: str, shard_chars: int):
        self.prefix, self.shard_chars = prefix, shard_chars
        self.index, self.chars, self.f = -1, 0, None
        self.paths: list[str] = []

    def write(self, text: str) -> None:
        if self.f is None or self.chars >= self.shard_chars:
            self.close()
            self.index += 1
            path = f"{self.prefix}_{self.index:03d}.txt"
            self.paths.append(path)
            self.f = open(path, "w", encoding="utf-8", newline="\n")    # \n even on Windows
            self.chars = 0
        self.f.write(text)
        self.chars += len(text)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def write_pretrain(docs: Iterable[str], out_dir: str, max_chars: int, shard_chars: int = 256_000_000) -> dict:
    """Every document as "<web>\\n" + text + "\\n\\n", one in every 100 to pretrain_val.txt,
    until max_chars characters are written (train and val together)."""
    os.makedirs(out_dir, exist_ok=True)
    train = ShardWriter(os.path.join(out_dir, "pretrain"), shard_chars)
    val_path = os.path.join(out_dir, "pretrain_val.txt")
    stats = Counter()
    with open(val_path, "w", encoding="utf-8", newline="\n") as val:
        for text in docs:
            plain = tidy(text or "")
            body = safe_lines(plain)
            if not body.strip():
                stats["empty"] += 1
                continue
            piece = "<web>\n" + body + "\n\n"
            if stats["kept"] % HOLD_OUT_EVERY == HOLD_OUT_EVERY - 1:
                val.write(piece)
                stats["val_docs"] += 1
                stats["val_chars"] += len(piece)
            else:
                train.write(piece)
                stats["train_docs"] += 1
                stats["train_chars"] += len(piece)
            stats["kept"] += 1
            stats["escaped"] += sum(1 for line in plain.split("\n") if TAG_LINE.match(line))
            if stats["train_chars"] + stats["val_chars"] >= max_chars:
                break
    train.close()
    stats["shards"] = len(train.paths)
    return dict(stats)


# ---------------------------------------------------------------- chat text

def conversation_text(messages: list[dict]) -> str | None:
    """'<user>\\n' + question + '\\n<assistant>\\n' + answer + '\\n' ... + '<user>\\n'.

    System messages are left out (the chat program has no system prompt to
    match them). The rest must alternate user, assistant, user, ... starting
    with a user turn and ending with an assistant turn, with no empty turns,
    else None. The trailing "<user>" teaches the model where an answer ends."""
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) < 2 or len(turns) % 2 != 0:
        return None
    parts = []
    for k, m in enumerate(turns):
        role = "user" if k % 2 == 0 else "assistant"
        if m.get("role") != role:
            return None
        body = safe_lines(tidy(m.get("content") or "").strip())     # strip first: escaping adds a leading space
        if not body:
            return None
        parts.append(f"<{role}>\n{body}\n")
    return "".join(parts) + "<user>\n"


def write_sft(convs: Iterable[list[dict]], path: str) -> dict:
    """Conversations, one blank line between them."""
    stats = Counter()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for messages in convs:
            stats["seen"] += 1
            stats["with_system"] += any(m.get("role") == "system" for m in messages)
            text = conversation_text(messages)
            if text is None:
                stats["skipped"] += 1
                continue
            f.write(text + "\n")
            stats["kept"] += 1
            stats["chars"] += len(text) + 1
    return dict(stats)


# ---------------------------------------------------------------- joining token files

BPE1 = struct.Struct("<III")                    # magic "BPE1", version 1, count
BPE1_MAGIC = 0x31455042


def join_token_files(paths: list[str], out: str) -> int:
    """BPE1 token files -> one BPE1 file with every id in order. Streams, so a
    few GB of tokens need no more memory than one copy buffer."""
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            magic, version, count = BPE1.unpack(f.read(12))
        if magic != BPE1_MAGIC or version != 1 or os.path.getsize(p) != 12 + 2 * count:
            raise SystemExit(f"{p} is not a complete BPE1 token file")
        total += count
    if total >= 2 ** 32:
        raise SystemExit(f"{total:,} tokens do not fit the uint32 count of a BPE1 file")
    tmp = out + ".tmp"
    with open(tmp, "wb") as o:
        o.write(BPE1.pack(BPE1_MAGIC, 1, total))
        for p in paths:
            with open(p, "rb") as f:
                f.seek(12)
                while True:
                    block = f.read(1 << 24)
                    if not block:
                        break
                    o.write(block)
    os.replace(tmp, out)
    return total


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download and write the stage 6 text. Run from the repo root.")
    p.add_argument("--max-chars", type=float, default=4.5e9, help="web text to write (default 4.5e9; 2e7 to test)")
    p.add_argument("--shard-chars", type=float, default=2.56e8, help="characters per pretrain_NNN.txt")
    p.add_argument("--skip-web", action="store_true", help="leave the FineWeb-Edu text alone")
    p.add_argument("--skip-sft", action="store_true", help="leave the smol-smoltalk text alone")
    p.add_argument("--join", action="store_true",
                   help="only join data/chat/pretrain_NNN.bin into data/chat/pretrain_train.bin")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    os.makedirs(OUT, exist_ok=True)

    if args.join:
        shards = sorted(glob.glob(os.path.join(OUT, "pretrain_[0-9][0-9][0-9].bin")))
        if not shards:
            raise SystemExit(f"no {OUT}/pretrain_NNN.bin files (encode the text first)")
        n = join_token_files(shards, os.path.join(OUT, "pretrain_train.bin"))
        print(f"{len(shards)} shards -> {OUT}/pretrain_train.bin: {n:,} tokens")
        return

    if not args.skip_web:
        print(f"web text from {FINEWEB} ({FINEWEB_DIR}), up to {args.max_chars:,.0f} characters")
        s = write_pretrain(web_documents(), OUT, int(args.max_chars), int(args.shard_chars))
        print(f"  {s.get('train_docs', 0):,} train documents ({s.get('train_chars', 0):,} chars, "
              f"{s.get('shards', 0)} files), {s.get('val_docs', 0):,} val documents "
              f"({s.get('val_chars', 0):,} chars), {s.get('empty', 0):,} empty dropped, "
              f"{s.get('escaped', 0):,} tag-shaped lines escaped")

    if not args.skip_sft:
        for split, name in (("train", "sft_train.txt"), ("test", "sft_val.txt")):
            print(f"conversations from {SMOLTALK} ({split})")
            s = write_sft(smoltalk_conversations(split), os.path.join(OUT, name))
            print(f"  {s.get('kept', 0):,} kept, {s.get('skipped', 0):,} skipped "
                  f"(of {s.get('seen', 0):,}; {s.get('with_system', 0):,} had a system message, left out), "
                  f"{s.get('chars', 0):,} chars -> {OUT}/{name}")


if __name__ == "__main__":
    main()
