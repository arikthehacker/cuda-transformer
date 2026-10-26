#pragma once
// data.h: what both trainers (train.cu, train_fast.cu) and generate.cu share
// for reading tokens and printing numbers.

#include <cstdint>
#include <string>
#include <vector>

// The BPE1 token file from stage 1 (magic, version, count, uint16 ids).
// Exits with a message if the file is bad or holds an id >= V.
std::vector<uint16_t> load_tokens16(const std::string &path, int V);
// The same ids as int (stage 3 and the small data set; 4 bytes per id)
std::vector<int> load_tokens(const std::string &path, int V);

// SFT targets from reference/chat.py (TGT1: magic, version, count, n_starts,
// n_ctx, then int32 targets[count], -1 = not counted, then uint32 starts).
struct Targets
{
    std::vector<int> y;                 // y[i] is the target of position i
    std::vector<uint32_t> starts;       // positions a batch row may start at
    int n_ctx = 0;                      // the context length the starts were cut for
};
// Exits with a message if the file is bad, does not match n_tokens, or was
// cut for a different n_ctx.
Targets load_targets(const std::string &path, size_t n_tokens, int n_ctx);

// Val batches: the same evenly spaced starts as fixed_batches() in reference/data.py
std::vector<long long> val_starts(size_t n_tokens, int T, int count);
// SFT val batches: count starts picked evenly from the list (fixed_starts() in reference/data.py)
std::vector<long long> val_starts_from(const std::vector<uint32_t> &starts, int count);

// 1234567 -> "1,234,567"
std::string with_commas(long long v);
