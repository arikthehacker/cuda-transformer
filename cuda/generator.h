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
//   chat      one conversation stays in the cache across turns: the user's
//             tokens go in one decode step each (no sampling), then an
//             answer is sampled until the model writes a tag
//
// Mode::NoCache is the baseline: every token runs the full forward pass over
// the last n_ctx tokens (what reference/model.py generate() does).

#include <functional>
#include <random>
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

    // ---- chat: one conversation in the cache across turns (generate --chat)
    // An empty conversation. max_new: the longest answer chat_answer will be asked for.
    void chat_start(int max_new, unsigned long long seed);
    // Appends tokens to the conversation, one cached decode step each (they
    // are a user's turn: few). The first token of a conversation just
    // becomes the pending token. Rolls the cache over like generate().
    void chat_feed(const std::vector<int> &tokens);
    // Samples until the model writes a token >= stop_from (any tag) or max_new
    // tokens came. on_token sees each token as it arrives, the tag included.
    // Afterwards `next` (the <user> tag) is the pending token: a different
    // tag is swapped for it, and an answer that ran out gets "\n" + next fed
    // after it. Returns the answer, ending with the tag when one came.
    std::vector<int> chat_answer(int max_new, SampleConfig sc, int stop_from, int next, Mode mode,
                                 const std::function<void(int)> &on_token);
    int chat_length() const                 // tokens the conversation has had so far
    {
        return chat_total;
    }

private:
    int capacity = 0;
    void reserve(int tokens);
    void prefill(int m, int n, SampleConfig sc, bool fill, bool sample = true);   // window = history[n - m .. n)
    void decode_layers();                   // the pending token through every layer, into the cache
    void decode_step(SampleConfig sc);      // decode_layers, then the logits and one draw
    void report(int window_start, int pos, bool from_prefill, int m);

    int chat_n = 0, chat_pos = 0;           // the history count and the pending token's position
    int chat_max_new = 0, chat_total = 0;
    std::mt19937_64 chat_rng;
    void feed_one(int token);
    void chat_room(int needed);             // move the last n_ctx tokens to the front when the history fills
};
