#pragma once
// decode.h: the kernels of one generation step with a KV cache (decode.cu).
//
// One new token is a (1 x C) row, so every layer is a matrix-VECTOR product:
// the time goes into reading the weights, not into math. Each GEMV reads
// every weight exactly once with 128-bit loads. Changing values (the token,
// its position, the random number for sampling) come from GPU memory, so one
// decode step can be captured as a CUDA Graph and replayed per token.
//
// Rounding follows the training forward pass (gpt_fast.cu) step for step:
// LayerNorm outputs, qkv, attention outputs, fc and gelu are rounded to
// floatX where the full forward stores them as floatX, so cached decoding and
// the full forward see the same numbers up to summation order.

#include <cuda_runtime.h>

#include "precision.h"

struct DecodeState
{
    int token;          // the newest token, not in the cache yet
    int pos;            // its position, = how many tokens the cache holds
    int n;              // tokens in the history so far (prompt + sampled)
    int sampled;        // tokens sampled so far (indexes the random numbers)
    int ban_from;       // while sampled is 0, ids >= ban_from are never drawn (0 = no ban)
};

struct SampleConfig
{
    float temperature;  // 0 = always the most likely token
    int top_k;          // keep the k most likely (ties at the k-th value too), 0 = all
};

// res (C) fp32 = wte[token] + wpe[pos]
void decode_embed(float *res, const float *wte, const float *wpe, const DecodeState *st, int C, cudaStream_t s);

// The GEMVs. W is (OC, IC) row-major floatX, b (OC) fp32.
// qkv: in = LayerNorm(res); q (C) floatX = rows 0..C-1, rows C..2C-1 go to
//      K[h][pos][d] and rows 2C..3C-1 to V[h][pos][d] of this layer's cache
void decode_qkv(floatX *q, floatX *k_cache, floatX *v_cache, const float *res, const float *ln_w, const float *ln_b,
                float eps, const floatX *W, const float *b, const DecodeState *st, int C, int H, int n_ctx,
                cudaStream_t s);
// res += floatX(W @ x + b), x (IC) floatX (the attention output, or gelu)
void decode_proj_add(float *res, const floatX *x, const floatX *W, const float *b, int OC, int IC, cudaStream_t s);
// out (OC) floatX = gelu(floatX(W @ LayerNorm(res) + b))
void decode_fc_gelu(floatX *out, const float *res, const float *ln_w, const float *ln_b, float eps, const floatX *W,
                    const float *b, int OC, int C, cudaStream_t s);
// logits (Vp) fp32 = wte @ LayerNorm(res)   (the tied output layer, no bias)
void decode_logits(float *logits, const float *res, const float *ln_w, const float *ln_b, float eps,
                   const floatX *wte, int Vp, int C, cudaStream_t s);

// atty (C) floatX: per head, softmax(q . K[0..pos] / sqrt(D)) @ V[0..pos]
void decode_attention(floatX *atty, const floatX *q, const floatX *k_cache, const floatX *v_cache,
                      const DecodeState *st, int H, int D, int n_ctx, cudaStream_t s);

// After a full forward over t tokens at positions 0..t-1 (the prefill), copy
// their K and V out of that layer's qkv (t, 3C) into the cache.
void fill_cache(floatX *k_cache, floatX *v_cache, const floatX *qkv, int t, int C, int H, int n_ctx, cudaStream_t s);

// One draw on the GPU from logits[0 .. V-1] (padded ids are never picked),
// with the random number rand[st->sampled] in [0, 1):
//   greedy (temperature 0): the first index of the largest logit
//   else: l / temperature, keep the top_k largest (ties at the k-th value
//   too, like torch.topk + masked_fill), softmax, then the smallest j whose
//   running sum of probabilities (in id order) passes rand * total.
// While st->sampled is 0 and st->ban_from > 0, the draw is over ids
// 0 .. ban_from - 1 only, so a chat answer cannot start with a tag.
// Then the state moves on: the new token is history[st->n], n and sampled
// go up by one, pos becomes set_pos (after a prefill) or pos + 1.
// TL is float (the decode logits) or floatX (the prefill's full forward logits).
template <class TL>
void sample_token(DecodeState *st, int *history, const TL *logits, const float *rand, int V, SampleConfig sc,
                  int set_pos, cudaStream_t s);

// Chat: a token the host already knows (the user's words) instead of a draw.
// history[st->n] = token, n goes up by one, token becomes the pending token,
// pos becomes set_pos (after a prefill) or pos + 1. sampled does not move.
void push_token(DecodeState *st, int *history, int token, int set_pos, cudaStream_t s);
// Swap the pending token (the last one in the history) for another, e.g. a
// stop tag the model wrote for the <user> that has to follow an answer.
void replace_token(DecodeState *st, int *history, int token, cudaStream_t s);

// the host version of the same draw, for the tests
int sample_host(const float *logits, int V, SampleConfig sc, float u);
