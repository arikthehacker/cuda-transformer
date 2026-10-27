"""Conversations as token ids: which tokens are trained on, and where batches may start.

A conversation in data/chat/sft_*.txt (prepare_chat.py) is

    <user>\\n question \\n <assistant>\\n answer \\n <user>\\n ... <assistant>\\n answer \\n <user>\\n
    (blank line)
    <user>\\n the next conversation ...

Only the answers are trained on: the target of position i is token i + 1
when that token is part of an assistant turn, else -1 (ignored by the loss).
The tag right after an answer counts too, because predicting it is how the
model learns to stop.

    python -m reference.chat --tokens data/chat/sft_train.bin --tokenizer data/chat/tokenizer.bpe \\
                             --n-ctx 512 --out data/chat/sft_train.tgt

writes the targets and the batch starts for cuda/train_fast --sft
(python -m reference.chat --check-tokenizer data/chat/tokenizer.bpe only
checks that the tokenizer has exactly the three chat tags):

    .tgt, little endian:
    uint32  magic 0x31544754 ("TGT1"), version 1, count, n_starts, n_ctx
    int32   targets[count]          -1 = not counted
    uint32  starts[n_starts]        positions a batch row may start at

Numpy only (no torch), so the CUDA side's data can be made anywhere.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

from .tokenizer import NEWLINE, load_bpe

TGT_MAGIC = 0x31544754
TGT_VERSION = 1
TGT_HEADER = struct.Struct("<5I")           # magic, version, count, n_starts, n_ctx
BPE1_MAGIC = 0x31455042


def load_tokens(path: str) -> np.ndarray:
    """The uint16 ids of a BPE1 token file (like data.load_tokens, without torch)."""
    with open(path, "rb") as f:
        magic, version, count = struct.unpack("<III", f.read(12))
        if magic != BPE1_MAGIC or version != 1:
            raise ValueError(f"{path} is not a BPE1 token file")
        ids = np.fromfile(f, dtype="<u2", count=count)
    if ids.size != count:
        raise ValueError(f"{path} holds {ids.size:,} ids, the header says {count:,}")
    return ids


def _ranges(first: np.ndarray, last: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of length n, True on every [first[k], last[k]] (inclusive)."""
    d = np.bincount(first, minlength=n + 1)[:n + 1] - np.bincount(last + 1, minlength=n + 1)[:n + 1]
    return np.cumsum(d[:n]) > 0


def _tag_ids(tags: dict[str, int]) -> tuple[int, int, np.ndarray]:
    for name in ("<user>", "<assistant>"):
        if name not in tags:
            raise ValueError(f"the tokenizer has no {name} tag (it has: {' '.join(tags)})")
    return tags["<user>"], tags["<assistant>"], np.array(sorted(tags.values()), dtype=np.int64)


def sft_targets(tokens: np.ndarray, tags: dict[str, int]) -> np.ndarray:
    """int64 targets, same length as tokens: y[i] = tokens[i + 1] when token
    i + 1 is part of an assistant turn (its text, and the tag that ends it),
    else -1. The last position has no next token: -1."""
    t = np.asarray(tokens).astype(np.int64)
    n = t.size
    user, assistant, all_tags = _tag_ids(tags)
    tag_pos = np.flatnonzero(np.isin(t, all_tags))
    asst = tag_pos[t[tag_pos] == assistant]
    # the turn's text starts after the tag and the "\n" byte that follows it ...
    first = asst + 1 + (t[np.minimum(asst + 1, n - 1)] == NEWLINE)
    # ... and ends at the next tag, which is counted as the stop signal
    nxt = np.searchsorted(tag_pos, asst, side="right")
    has_next = nxt < tag_pos.size
    last = np.where(has_next, tag_pos[np.minimum(nxt, tag_pos.size - 1)], n - 1)
    ok = first <= last
    counted = _ranges(first[ok], last[ok], n)          # token j is part of an assistant turn
    y = np.full(n, -1, dtype=np.int64)
    y[:-1] = np.where(counted[1:], t[1:], -1)
    return y


def conversation_starts(tokens: np.ndarray, tags: dict[str, int]) -> np.ndarray:
    """Index of every <user> tag that begins a conversation: the first tag of
    the file, or a <user> whose previous tag is also <user> (the stop tag of
    the conversation before; inside a conversation a <user> always follows an
    <assistant>)."""
    t = np.asarray(tokens)
    user, _, all_tags = _tag_ids(tags)
    tag_pos = np.flatnonzero(np.isin(t, all_tags))
    if tag_pos.size == 0:
        return np.zeros(0, dtype=np.int64)
    is_user = t[tag_pos] == user
    prev_user = np.concatenate([[True], is_user[:-1]])
    return tag_pos[is_user & prev_user].astype(np.int64)


def cut_to_context(targets: np.ndarray, starts: np.ndarray, n_ctx: int, n_tokens: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """A batch row is the n_ctx tokens from a conversation start, so an answer
    whose last target lies past start + n_ctx - 1 can never be trained whole.
    Those answers (and every answer after them in that conversation) get -1,
    cutting the conversation at a turn boundary. A conversation with no
    answer left is dropped, and so is a start too close to the end of the file.
    Returns (targets, starts, counts)."""
    y = targets.copy()
    counted = y >= 0
    edges = np.diff(np.concatenate([[False], counted, [False]]).astype(np.int8))
    run_first = np.flatnonzero(edges == 1)            # one run per assistant turn
    run_last = np.flatnonzero(edges == -1) - 1
    conv = np.searchsorted(starts, run_first, side="right") - 1
    limit = np.where(conv >= 0, starts[np.maximum(conv, 0)] + n_ctx - 1, -1)
    keep = (conv >= 0) & (run_last <= limit)
    y[_ranges(run_first[~keep], run_last[~keep], n_tokens)] = -1

    kept_runs = np.bincount(conv[keep], minlength=starts.size) if starts.size else np.zeros(0, np.int64)
    all_runs = np.bincount(conv[conv >= 0], minlength=starts.size) if starts.size else np.zeros(0, np.int64)
    fits_file = starts + n_ctx + 1 <= n_tokens
    usable = (kept_runs > 0) & fits_file
    counts = {
        "conversations": int(starts.size),
        "whole": int(((kept_runs == all_runs) & usable).sum()),
        "cut": int(((kept_runs < all_runs) & usable).sum()),
        "dropped": int((~usable).sum()),
        "answers_dropped": int((~keep).sum()),
        "counted_targets": int((y >= 0).sum()),
    }
    return y, starts[usable], counts


def write_targets(path: str, targets: np.ndarray, starts: np.ndarray, n_ctx: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(TGT_HEADER.pack(TGT_MAGIC, TGT_VERSION, targets.size, starts.size, n_ctx))
        f.write(np.asarray(targets, dtype="<i4").tobytes())
        f.write(np.asarray(starts, dtype="<u4").tobytes())
    os.replace(tmp, path)


def read_targets(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """(targets int32, starts int64, n_ctx) from a .tgt file."""
    with open(path, "rb") as f:
        raw = f.read()
    magic, version, count, n_starts, n_ctx = TGT_HEADER.unpack_from(raw)
    if magic != TGT_MAGIC or version != TGT_VERSION:
        raise ValueError(f"{path} is not a TGT1 targets file")
    if len(raw) != TGT_HEADER.size + 4 * count + 4 * n_starts:
        raise ValueError(f"{path} is the wrong size for {count:,} targets and {n_starts:,} starts")
    y = np.frombuffer(raw, dtype="<i4", count=count, offset=TGT_HEADER.size).copy()
    s = np.frombuffer(raw, dtype="<u4", count=n_starts, offset=TGT_HEADER.size + 4 * count).astype(np.int64)
    return y, s, n_ctx


def prepare(tokens: np.ndarray, tags: dict[str, int], n_ctx: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Targets and starts, cut to n_ctx: what train.py --sft and the .tgt file use."""
    y = sft_targets(tokens, tags)
    starts = conversation_starts(tokens, tags)
    return cut_to_context(y, starts, n_ctx, tokens.size)


CHAT_TAGS = ("<web>", "<user>", "<assistant>")
# bpe -t -r reserves these (make chat-tokens), so a later stage can use a tag
# without a new tokenizer. Rows of tags the data never has stay near their init.
RESERVED = ("<system>", "<tool>", "<code>", "<book>")


def check_tokenizer(path: str) -> None:
    """Exit with a message unless the tokenizer has <web>, <user> and <assistant>,
    and every other tag is one of RESERVED: one stray tag-shaped line in the
    training text adds a special token for good, so this is checked before
    encoding anything. Also counts the ids that are only digits."""
    vocab, tags = load_bpe(path)
    missing = [t for t in CHAT_TAGS if t not in tags]
    extra = [t for t in tags if t not in CHAT_TAGS and t not in RESERVED]
    if missing or extra:
        raise SystemExit(f"{path}: specials {len(tags)} ({' '.join(tags)})"
                         + (f"; extra: {' '.join(extra)} (a tag-shaped line got past safe_lines)" if extra else "")
                         + (f"; missing: {' '.join(missing)}" if missing else ""))
    digits = sum(1 for i, v in enumerate(vocab) if i >= 256 and i not in tags.values() and v.strip(b" ").isdigit()
                 and not v.startswith(b"  "))
    print(f"{path}: {len(vocab):,} ids, specials {len(tags)} ({' '.join(tags)}), nothing else; "
          f"{digits:,} learned ids are only digits (at most 2,220 with groups of 3)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SFT targets and batch starts for cuda/train_fast --sft.")
    p.add_argument("--check-tokenizer", metavar="FILE", help="only check that FILE has exactly the 3 chat tags")
    p.add_argument("--tokens", help="a BPE1 token file of conversations, e.g. data/chat/sft_train.bin")
    p.add_argument("--tokenizer", default="data/chat/tokenizer.bpe", help="for the tag ids")
    p.add_argument("--n-ctx", type=int, default=512, help="the model's context length (default 512)")
    p.add_argument("--out", help="output .tgt (default: --tokens with .tgt instead of .bin)")
    args = p.parse_args(argv)
    if args.check_tokenizer:
        check_tokenizer(args.check_tokenizer)
        return
    if not args.tokens:
        p.error("--tokens is required")
    out = args.out or os.path.splitext(args.tokens)[0] + ".tgt"
    tokens = load_tokens(args.tokens)
    _, tags = load_bpe(args.tokenizer)
    y, starts, c = prepare(tokens, tags, args.n_ctx)
    write_targets(out, y, starts, args.n_ctx)
    print(f"{args.tokens}: {tokens.size:,} tokens, {c['conversations']:,} conversations: "
          f"{c['whole']:,} whole, {c['cut']:,} cut at a turn boundary, {c['dropped']:,} dropped "
          f"({c['answers_dropped']:,} answers past n_ctx {args.n_ctx}); "
          f"{c['counted_targets']:,} counted targets ({100 * c['counted_targets'] / max(1, tokens.size):.1f}%) "
          f"-> {out}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main()
