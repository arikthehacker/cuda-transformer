#pragma once
// data.h: what both trainers (train.cu, train_fast.cu) and generate.cu share
// for reading tokens and printing numbers.

#include <string>
#include <vector>

// The BPE1 token file from stage 1 (magic, version, count, uint16 ids).
// Exits with a message if the file is bad or holds an id >= V.
std::vector<int> load_tokens(const std::string &path, int V);

// Val batches: the same evenly spaced starts as fixed_batches() in reference/data.py
std::vector<long long> val_starts(size_t n_tokens, int T, int count);

// 1234567 -> "1,234,567"
std::string with_commas(long long v);
