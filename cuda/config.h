#pragma once
// config.h: model sizes (filled from the dump header, never typed in) and
// training settings (the same values as reference/config.py TrainConfig).

struct Dump;

struct ModelConfig
{
    int V = 0;                  // real vocab: 256 bytes + merges + tags
    int Vp = 0;                 // V padded up to a multiple of 64
    int n_ctx = 0;              // longest sequence, in tokens
    int L = 0;                  // layers
    int H = 0;                  // heads
    int C = 0;                  // n_embd
    float ln_eps = 1e-5f;       // from the dump record config.ln_eps

    int D() const               // head size
    {
        return C / H;
    }
};

ModelConfig config_from_dump(const Dump &d);

// Same numbers as reference/config.py, so both trainers follow the same schedule.
struct TrainConfig
{
    int batch_size = 16;        // 16 x 256 = 4,096 tokens per step
    int max_steps = 5000;
    float lr = 6e-4f;
    float min_lr = 6e-5f;
    int warmup_steps = 200;
    float weight_decay = 0.1f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float grad_clip = 1.0f;
    int eval_every = 250;       // steps
    int eval_batches = 20;
    unsigned seed = 1337;
    int timing_skip = 10;       // first steps left out of tokens per second
};

// Linear warmup to lr over warmup_steps, then cosine down to min_lr at
// max_steps. step is 0-based, same as lr_at() in reference/train.py.
float lr_at(int step, const TrainConfig &tc);
