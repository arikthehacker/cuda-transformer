#pragma once
// weights.h: weights.bin, the one weight file every program reads and writes
// (reference/export.py writes it from a PyTorch checkpoint, train_fast writes
// it after training, GPT::load, FastGPT and generate read it).
//
//   little endian, a 48 byte header, then the parameters:
//   uint32   magic 0x57545047 ("GPTW" as bytes), version 1
//   uint32   V, Vp, n_ctx, L, H, C
//   float32  ln_eps
//   uint32   dtype: 0 = float32, 1 = float16
//   uint64   count: how many values follow
//   data     every parameter in the dump order (layout.h), row-major

#include <string>
#include <vector>

#include "config.h"

struct Weights
{
    ModelConfig cfg;
    std::vector<float> params;              // fp16 files come back as fp32
};

Weights read_weights(const std::string &path);  // exits with a message if the file is bad
void write_weights(const std::string &path, const ModelConfig &cfg, const std::vector<float> &params,
                   bool fp16 = false);
