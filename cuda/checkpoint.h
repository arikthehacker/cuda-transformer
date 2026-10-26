#pragma once
// checkpoint.h: everything a long training run needs to pick up where it
// stopped (cuda/train_fast -c dir writes one every eval, -r resumes).
//
//   little endian, then the arrays:
//   char[4]  "GPTC", uint32 version 1
//   uint32   V, Vp, n_ctx, L, H, C, then float32 ln_eps          the model size
//   uint32   B, max_steps, eval_every, warmup_steps, sft         the run, so a
//   float32  lr, min_lr                                          resume is the same run
//   float32  best_val, int32 best_step
//   uint32   sizeof(StepState), then the StepState              loss scale, step, AdamW t ...
//   uint32   length + bytes of the data prefix (-D), then of the starts file (-s)
//   uint64   n, then float32 params[n], m[n], v[n]               weights and AdamW moments
//
// The write goes to path.tmp, is flushed to disk (fsync), and only then
// renamed over path, so a crash or a power cut never leaves half a file
// under the real name: the last complete checkpoint survives.

#include <string>
#include <vector>

#include "config.h"
#include "kernels/fast.h"

struct FastGPT;

struct RunInfo                  // what a resumed run must share with the one that wrote the file
{
    int B = 0, max_steps = 0, eval_every = 0, warmup_steps = 0;
    int sft = 0;
    float lr = 0.0f, min_lr = 0.0f;
    std::string data;           // -D prefix, "" = data/train.bin and data/val.bin
    std::string starts;         // -s file, "" = the run's own random batches
    float best_val = 0.0f;
    int best_step = 0;
};

struct Checkpoint
{
    ModelConfig cfg;
    RunInfo run;
    StepState state = {};
    std::vector<float> params, m, v;
};

void write_checkpoint(const std::string &path, const Checkpoint &c);    // exits with a message on failure
bool try_read_checkpoint(const std::string &path, Checkpoint &c, std::string &why);
Checkpoint read_checkpoint(const std::string &path);                    // exits with a message if the file is bad

// The model's weights, AdamW moments and StepState (waits for its stream)
Checkpoint checkpoint_of(FastGPT &model, const RunInfo &run);
// ... and back into a model of the same size (also rebuilds the fp16 weight copy)
void restore(FastGPT &model, const Checkpoint &c);

bool same_size(const ModelConfig &a, const ModelConfig &b);
