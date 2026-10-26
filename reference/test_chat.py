"""SFT targets on conversations written out by hand. Numpy only, so it runs
without torch:

    python -m pytest reference/test_chat.py      or      python -m reference.test_chat
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .chat import conversation_starts, cut_to_context, prepare, read_targets, sft_targets, write_targets

TAGS = {"<web>": 50, "<user>": 51, "<assistant>": 52}
U, A, NL = 51, 52, 10

# two conversations, the first with two turns. Words are ids 1 .. 9.
#        0  1  2  3  4   5  6  7  8  9  10 11 12 13  14 15 16 17  18  19  20
CONV_A = [U, NL, 1, 2, NL, A, NL, 3, 4, NL, U, NL, 5, NL, A, NL, 6, NL, U, NL, NL]
#        21 22 23 24  25 26 27 28  29 30
CONV_B = [U, NL, 7, NL, A, NL, 8, NL, U, NL]


def tokens(*convs: list[int]) -> np.ndarray:
    return np.array([t for c in convs for t in c], dtype=np.uint16)


def test_only_answers_and_their_stop_tag_count():
    t = tokens(CONV_A, CONV_B)
    y = sft_targets(t, TAGS)
    want = np.full(t.size, -1)
    # answer 1 is tokens 7, 8, 9 and the <user> at 10: the positions before them predict them
    for i in (6, 7, 8, 9, 15, 16, 17, 26, 27, 28):
        want[i] = t[i + 1]
    assert (y == want).all(), np.flatnonzero(y != want)
    assert int((y >= 0).sum()) == 10                  # counted by hand above
    assert y[5] == -1 and y[4] == -1                    # the tag and its newline are not answer text
    assert y[9] == U and y[17] == U and y[28] == U      # the stop signal counts
    assert y[-1] == -1


def test_conversation_starts():
    t = tokens(CONV_A, CONV_B)
    assert conversation_starts(t, TAGS).tolist() == [0, 21]


def test_cut_at_a_turn_boundary():
    t = tokens(CONV_A, CONV_B, CONV_A, CONV_B)         # 62 tokens, starts 0, 21, 31, 52
    y = sft_targets(t, TAGS)
    starts = conversation_starts(t, TAGS)
    assert starts.tolist() == [0, 21, 31, 52]
    # n_ctx 12: from start 0 the row reaches target 11, so answer 1 (targets 6..9) stays
    # and answer 2 (15..17) is cut, the same for the second copy of A at 31. B fits whole.
    # The last start is too close to the end of the file for a row of 12 + 1 tokens.
    y2, s2, c = cut_to_context(y, starts, 12, t.size)
    assert (y2[6:10] >= 0).all() and (y2[15:18] == -1).all()
    assert (y2[37:41] >= 0).all() and (y2[46:49] == -1).all()
    assert s2.tolist() == [0, 21, 31]
    assert c["cut"] == 2 and c["whole"] == 1 and c["dropped"] == 1 and c["answers_dropped"] == 2
    # n_ctx 6: not even answer 1 fits from start 0, so that conversation is dropped
    y3, s3, c3 = cut_to_context(y, starts, 6, t.size)
    assert 0 not in s3.tolist() and (y3[:21] == -1).all()
    # every start that is left has at least one counted target inside its row
    for s in s2:
        assert (y2[s:s + 12] >= 0).any()


def test_tgt_file_round_trip():
    t = tokens(CONV_A, CONV_B, CONV_A)
    y, starts, _ = prepare(t, TAGS, 12)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.tgt")
        write_targets(path, y, starts, 12)
        y2, s2, n_ctx = read_targets(path)
    assert (y2 == y).all() and (s2 == starts).all() and n_ctx == 12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
