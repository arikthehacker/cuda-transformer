#pragma once
// generator.h: text generation with a KV cache.
//
//   prefill   a full forward pass (FastGPT, B = 1) over the prompt fills the
//             cache and gives the first token's logits
//   decode    every later token runs ONE row through the model: GEMVs, the new
//             q against the cached K and V, sampling on the GPU
//   rollover  positions come from the learned wpe (0 .. n_ctx - 1), so when
//             the cache is full the last n_ctx / 2 tokens go through prefill
//             again at positions 0 .. n_ctx / 2 - 1
//   graph     one decode step is captured once and replayed per token
//
// Mode::NoCache is the baseline: every token runs the full forward pass over
// the last n_ctx tokens (what reference/model.py generate() does).

#include <functional>
#include <vector>

#include "common.h"
#include "config.h"
#include "gpt_fast.h"
#include "graph.h"
#include "kernels/decode.h"

struct Generator
{
    enum class Mode
    {
        CachedGraph,        // KV cache, one CUDA Graph launch per token
        Cached,             // KV cache, the decode kernels launched one by one
        NoCache,            // full forward pass per token
    };

    ModelConfig cfg;
    FastGPT model;          // the weights, and the full forward pass for prefill
    DeviceArray<floatX> k_cache, v_cache;   // (L, H, n_ctx, D) each
    DeviceBuffer res, logits;               // (C) fp32 residual of the new row, (Vp) fp32
    DeviceArray<floatX> q, atty, fc;        // (C), (C), (4C)
    DeviceArray<DecodeState> state;
    DeviceInts history;                     // every token so far
    DeviceBuffer rand;                      // one uniform number per sampled token
    DeviceBuffer prefill_loss;
    CudaGraph graph;
    SampleConfig graph_sc = {-1.0f, -1};    // what the graph was captured with
    float last_ms = 0.0f;                   // GPU time of the last generate()

    // Test hook: after each draw, the logits it used (fp32, Vp values), the
    // history index of the token at cache position 0, and the position of
    // the token those logits came from. Setting it syncs after every token.
    std::function<void(const std::vector<float> &logits, int window_start, int pos)> on_logits;

    Generator(const ModelConfig &cfg, const std::vector<float> &params);

    // prompt + max_new sampled tokens. The same seed gives the same tokens.
    std::vector<int> generate(const std::vector<int> &prompt, int max_new, SampleConfig sc, unsigned long long seed,
                              Mode mode);

private:
    int capacity = 0;
    void reserve(int tokens);
    void prefill(int m, int n, SampleConfig sc, bool fill);   // window = history[n - m .. n)
    void decode_step(SampleConfig sc);
    void report(int window_start, int pos, bool from_prefill, int m);
};
