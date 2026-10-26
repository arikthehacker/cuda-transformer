#pragma once
// kernels.h: one launcher per kernel, with its math.

#include <cuda_runtime.h>
//
// Rules for every launcher:
//   - row-major, N = B*T rows, shapes in a comment on every pointer
//   - backward kernels ACCUMULATE (+=) into gradients, never overwrite. The
//     training step zeroes all gradients once before backward.
//   - shapes are checked with assert on the host before launching
//   - the attention launchers and sgemm take an optional stream (stage 4
//     reuses them inside a captured CUDA Graph); the default is stream 0
//
// Atomics (the only places results can change in the last bits between runs):
//   encoder_backward: dwte, because the same token appears many times.
// Every other reduction runs in a fixed order and gives the same bits every run.

// ---------------------------------------------------------------- encoder.cu
// out[b,t,:] = wte[tok[b,t],:] + wpe[t,:]
void encoder_forward(float *out,            // (B, T, C)
                     const int *tok,        // (B, T)
                     const float *wte,      // (Vp, C)
                     const float *wpe,      // (n_ctx, C), rows 0..T-1 used
                     int B, int T, int C);
// dwte[tok[b,t],:] += dout[b,t,:]          (atomicAdd)
// dwpe[t,:] += sum over b of dout[b,t,:]   (one thread loops over b, no atomics)
void encoder_backward(float *dwte,          // (Vp, C)
                      float *dwpe,          // (n_ctx, C)
                      const float *dout,    // (B, T, C)
                      const int *tok,       // (B, T)
                      int B, int T, int C);

// ---------------------------------------------------------------- layernorm.cu
// mean = sum(x) / C, var = sum((x - mean)^2) / C (divide by C, not C - 1),
// rstd = 1 / sqrt(var + eps), out = (x - mean) * rstd * w + b
void layernorm_forward(float *out,          // (N, C)
                       float *mean,         // (N)
                       float *rstd,         // (N)
                       const float *x,      // (N, C)
                       const float *w,      // (C)
                       const float *b,      // (C)
                       int N, int C, float eps);
// xhat = (x - mean) * rstd, g = dout * w
// dw += sum over rows of dout * xhat,  db += sum over rows of dout
// dx += rstd * (g - mean(g) - xhat * mean(g * xhat))      (means over C)
void layernorm_backward(float *dx,          // (N, C)
                        float *dw,          // (C)
                        float *db,          // (C)
                        const float *dout,  // (N, C)
                        const float *x,     // (N, C)
                        const float *w,     // (C)
                        const float *mean,  // (N)
                        const float *rstd,  // (N)
                        int N, int C);

// ---------------------------------------------------------------- sgemm.cu
// For each batch item:  C = alpha * op(A) @ op(B) + beta * C, all row-major.
// op(A) is M x K, op(B) is K x N, C is M x N.
// transA true means A is stored K x M, so op(A) = A^T. Same for transB.
// lda/ldb/ldc = row strides as stored. stride* = elements between batch items.
// beta 1 gives "+=", which is what the backward pass needs. With beta 0, C is
// never read, so it may hold garbage.
void sgemm(bool transA, bool transB, int M, int N, int K, float alpha,
           const float *A, int lda, long long strideA,
           const float *B, int ldb, long long strideB,
           float beta, float *C, int ldc, long long strideC, int batch, cudaStream_t stream = 0);
// Same contract, one thread per output, no tricks: the test oracle for sgemm().
void sgemm_naive(bool transA, bool transB, int M, int N, int K, float alpha,
                 const float *A, int lda, long long strideA,
                 const float *B, int ldb, long long strideB,
                 float beta, float *C, int ldc, long long strideC, int batch);

// Linear layers on top of sgemm (W is stored (out, in), like nn.Linear):
//   forward   out (N, OC) = inp (N, IC) @ W^T + b
//   backward  dinp += dout @ W,  dW += dout^T @ inp,  db += sum over rows of dout
// b / db may be nullptr (the tied logits layer has no bias).
void linear_forward(float *out,             // (N, OC)
                    const float *inp,       // (N, IC)
                    const float *w,         // (OC, IC)
                    const float *b,         // (OC) or nullptr
                    int N, int IC, int OC);
void linear_backward(float *dinp,           // (N, IC)
                     float *dw,             // (OC, IC)
                     float *db,             // (OC) or nullptr
                     const float *dout,     // (N, OC)
                     const float *inp,      // (N, IC)
                     const float *w,        // (OC, IC)
                     int N, int IC, int OC);

// ---------------------------------------------------------------- bias.cu
// out[n,o] += bias[o]                      (after the forward sgemm)
void add_bias(float *out,                   // (N, OC)
              const float *bias,            // (OC)
              int N, int OC);
// dbias[o] += sum over n of dout[n,o]
void bias_backward(float *dbias,            // (OC)
                   const float *dout,       // (N, OC)
                   int N, int OC);

// ---------------------------------------------------------------- attention.cu
// qkv (B,T,3C) -> q, k, v (B,H,T,D). Head h of q is columns h*D .. h*D+D-1
// of the first C columns, k is the next C, v the last C.
void attention_permute(float *q, float *k, float *v,    // (B, H, T, D) each
                       const float *qkv,                // (B, T, 3C)
                       int B, int T, int H, int D, cudaStream_t stream = 0);
// In place on scores: set j > i to -inf (the dump stores -inf there), then
// probs = softmax over each row. The row max only looks at j <= i, and the
// diagonal is always there, so the max is never -inf.
void causal_softmax(float *probs,           // (B, H, T, T)
                    float *scores,          // (B, H, T, T)
                    int B, int H, int T, cudaStream_t stream = 0);
// y (B,H,T,D) -> atty (B,T,C), atty[b,t,h*D+d] = y[b,h,t,d]
void attention_unpermute(float *atty,       // (B, T, C)
                         const float *y,    // (B, H, T, D)
                         int B, int T, int H, int D, cudaStream_t stream = 0);
// dscores[i,j] += probs[i,j] * (dprobs[i,j] - sum_k probs[i,k] * dprobs[i,k])
// (masked spots come out 0 on their own, because probs is 0 there)
void causal_softmax_backward(float *dscores,        // (B, H, T, T)
                             const float *dprobs,   // (B, H, T, T)
                             const float *probs,    // (B, H, T, T)
                             int B, int H, int T, cudaStream_t stream = 0);
// The reverse of the two permutes, for the gradients (both +=):
void attention_unpermute_backward(float *dy,            // (B, H, T, D)
                                  const float *datty,   // (B, T, C)
                                  int B, int T, int H, int D, cudaStream_t stream = 0);
void attention_permute_backward(float *dqkv,            // (B, T, 3C)
                                const float *dq, const float *dk, const float *dv,  // (B, H, T, D)
                                int B, int T, int H, int D, cudaStream_t stream = 0);

// The whole attention, built from the pieces above and sgemm:
//   q, k, v = permute(qkv); scores = q @ k^T / sqrt(D); probs = causal_softmax(scores)
//   y = probs @ v; atty = unpermute(y)
void attention_forward(float *atty,                     // (B, T, C)
                       float *q, float *k, float *v,    // (B, H, T, D) each
                       float *scores, float *probs,     // (B, H, T, T) each
                       float *y,                        // (B, H, T, D)
                       const float *qkv,                // (B, T, 3C)
                       int B, int T, int H, int D, cudaStream_t stream = 0);
//   dy = unpermute_backward(datty); dprobs = dy @ v^T; dv = probs^T @ dy
//   dscores = causal_softmax_backward(dprobs, probs)
//   dq = dscores @ k / sqrt(D); dk = dscores^T @ q / sqrt(D)
//   dqkv = permute_backward(dq, dk, dv)
// Every output is accumulated, so all the d* buffers start at 0.
void attention_backward(float *dqkv,                    // (B, T, 3C)
                        float *dq, float *dk, float *dv,    // (B, H, T, D) each
                        float *dscores, float *dprobs,  // (B, H, T, T) each
                        float *dy,                      // (B, H, T, D)
                        const float *datty,             // (B, T, C)
                        const float *q, const float *k, const float *v,    // (B, H, T, D)
                        const float *probs,             // (B, H, T, T)
                        int B, int T, int H, int D, cudaStream_t stream = 0);

// ---------------------------------------------------------------- gelu.cu
// s = sqrt(2/pi), u = s * (x + 0.044715 x^3)
// out = 0.5 * x * (1 + tanh(u))
void gelu_forward(float *out, const float *x, long long n);
// dx += dout * (0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * s * (1 + 3 * 0.044715 x^2))
void gelu_backward(float *dx, const float *x, const float *dout, long long n);

// ---------------------------------------------------------------- residual.cu
void residual_forward(float *out, const float *a, const float *b, long long n);   // out = a + b
// dst += src. The residual backward: out = a + b sends dout to a and to b unchanged.
void accumulate(float *dst, const float *src, long long n);

// ---------------------------------------------------------------- crossentropy.cu
// One block per row, V real columns out of Vp.
// m = max_j<V logits[n,j],  lse = m + log(sum_j<V exp(logits[n,j] - m))
// losses[n] = lse - logits[n, target[n]]
// The scalar loss is the mean of losses over N, summed on the host in double.
void crossentropy_forward(float *losses,            // (N)
                          const float *logits,      // (N, Vp)
                          const int *targets,       // (N)
                          int N, int V, int Vp);
// dlogits[n,j] += (softmax[n,j] - (j == target[n])) / N    for j < V
// dlogits[n,j] stays 0                                    for V <= j < Vp
void crossentropy_backward(float *dlogits,          // (N, Vp)
                           const float *logits,     // (N, Vp)
                           const int *targets,      // (N)
                           int N, int V, int Vp);

// ---------------------------------------------------------------- adamw.cu
// PyTorch's exact order, t starts at 1:
//   p = p * (1 - lr * wd)                  wd is 0 for 1D parameters
//   m = m + (1 - b1) * (g - m)             (= b1 * m + (1 - b1) * g)
//   v = b2 * v + (1 - b2) * g * g
//   p = p - (lr / (1 - b1^t)) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)
void adamw_step(float *p, const float *g, float *m, float *v, long long n,
                float lr, float b1, float b2, float eps, float wd, int t);
// sqrt(sum of g^2) over all n values: per block partial sums in a fixed
// order, then the final sum on the host in double.
float grad_norm(const float *g, long long n);
// x *= s (gradient clipping scales every gradient by min(1, clip / norm))
void scale(float *x, float s, long long n);
