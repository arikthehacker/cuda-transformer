"""Reads data/tokenizer.bpe (written by the stage 1 C++ tool) in Python.

Only decode and the tag ids are needed here. Encoding stays in C++.
"""
from __future__ import annotations

import re

TAG = re.compile(r"<[a-z_]+>")
NEWLINE = 10                    # the byte after every tag line


def load_bpe(path: str) -> tuple[list[bytes], dict[str, int]]:
    """Read "bpe v1" or "bpe v2" (the version only says which pretokenizer
    rules encode() uses, and nothing here encodes). Rebuild the vocab bytes from the merges the same way
    load() in bpe.cpp does, and return {"<bible>": id, ...} for the tags.
    Tag ids come after the merges: 256 + len(merges) + index."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = [line.rstrip("\r") for line in f.read().split("\n")]

    def fail(why: str) -> ValueError:
        return ValueError(f"{path}: {why}")

    if lines[0].split() not in (["bpe", "v1"], ["bpe", "v2"]):
        raise fail(f"first line is {lines[0]!r}, expected 'bpe v1' or 'bpe v2'")
    head = lines[1].split()
    if len(head) != 2 or head[0] != "merges":
        raise fail(f"second line is {lines[1]!r}, expected 'merges N'")
    num_merges = int(head[1])

    vocab = [bytes([b]) for b in range(256)]
    for i in range(num_merges):
        a, b = (int(s) for s in lines[2 + i].split())
        new_id = 256 + i
        if not (0 <= a < new_id and 0 <= b < new_id):
            raise fail(f"merge {i} uses ids ({a}, {b}), only ids below {new_id} exist yet")
        vocab.append(vocab[a] + vocab[b])

    at = 2 + num_merges
    head = lines[at].split()
    if len(head) != 2 or head[0] != "specials":
        raise fail(f"line {at + 1} is {lines[at]!r}, expected 'specials N'")
    tags: dict[str, int] = {}
    for j in range(int(head[1])):
        name = lines[at + 1 + j]
        if not TAG.fullmatch(name):
            raise fail(f"{name!r} is not a tag like <bible>")
        tags[name] = len(vocab)
        vocab.append(name.encode("utf-8"))

    if len(vocab) > 65536:
        raise fail(f"{len(vocab)} ids do not fit in uint16")
    return vocab, tags


def decode(vocab: list[bytes], ids: list[int]) -> str:
    """Join the bytes, then decode as UTF-8 with errors="replace", because a
    sample can stop in the middle of a multi-byte character."""
    return b"".join(vocab[int(i)] for i in ids).decode("utf-8", errors="replace")


def prompt(tags: dict[str, int], tag: str) -> list[int]:
    """[tag_id, 10]: the tag, then the newline byte that follows it.
    `tag` can be "bible" or "<bible>"."""
    name = tag if tag.startswith("<") else f"<{tag}>"
    if name not in tags:
        raise KeyError(f"unknown tag {name}, the tokenizer has: {' '.join(tags)}")
    return [tags[name], NEWLINE]
