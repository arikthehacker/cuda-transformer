// attention.cu: permute, causal softmax, unpermute, forward and backward.
// The two matmuls per direction go through sgemm with batch = B*H.

#include <cassert>
#include <cmath>

#include "common.h"
#include "kernels.h"
#include "reduce.cuh"

using namespace std;

static const int BLOCK = 256;
static const int ROWS_PER_BLOCK = BLOCK / 32;   // softmax: one warp per row

// One thread per (b, h, t, d), d fastest. Reads qkv[b, t, h*D + d] and the
// same spot C and 2C columns further on.
__global__ void permute_kernel(float *q, float *k, float *v, const float *qkv,
                               int T, int H, int D, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        int d = static_cast<int>(idx % D);
        int t = static_cast<int>((idx / D) % T);
        int h = static_cast<int>((idx / (static_cast<long long>(D) * T)) % H);
        long long b = idx / (static_cast<long long>(D) * T * H);
        int C = H * D;
        long long src = (b * T + t) * 3 * C + h * D + d;
        q[idx] = qkv[src];
        k[idx] = qkv[src + C];
        v[idx] = qkv[src + 2 * C];
    }
}

// The reverse, for the gradients: dqkv[b, t, h*D + d] += dq[b, h, t, d], and
// the same for dk (+C) and dv (+2C). Every dqkv value has exactly one source.
__global__ void permute_backward_kernel(float *dqkv, const float *dq, const float *dk, const float *dv,
                                        int T, int H, int D, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        int d = static_cast<int>(idx % D);
        int t = static_cast<int>((idx / D) % T);
        int h = static_cast<int>((idx / (static_cast<long long>(D) * T)) % H);
        long long b = idx / (static_cast<long long>(D) * T * H);
        int C = H * D;
        long long dst = (b * T + t) * 3 * C + h * D + d;
        dqkv[dst] += dq[idx];
        dqkv[dst + C] += dk[idx];
        dqkv[dst + 2 * C] += dv[idx];
    }
}

// One thread per (b, t, c) of atty, c fastest (coalesced writes):
// atty[b, t, h*D + d] = y[b, h, t, d]
__global__ void unpermute_kernel(float *atty, const float *y, int T, int H, int D, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        int C = H * D;
        int c = static_cast<int>(idx % C);
        int t = static_cast<int>((idx / C) % T);
        long long b = idx / (static_cast<long long>(C) * T);
        int h = c / D;
        int d = c % D;
        atty[idx] = y[((b * H + h) * T + t) * D + d];
    }
}

// dy[b, h, t, d] += datty[b, t, h*D + d]
__global__ void unpermute_backward_kernel(float *dy, const float *datty, int T, int H, int D, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        int C = H * D;
        int c = static_cast<int>(idx % C);
        int t = static_cast<int>((idx / C) % T);
        long long b = idx / (static_cast<long long>(C) * T);
        int h = c / D;
        int d = c % D;
        dy[((b * H + h) * T + t) * D + d] += datty[idx];
    }
}

// One warp per row i of one (b, h). Only j <= i are real; the row max and the
// sum only look at those, so nothing ever computes -inf - (-inf).
__global__ void causal_softmax_kernel(float *probs, float *scores, int T, long long rows)
{
    long long row = static_cast<long long>(blockIdx.x) * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < rows)
    {
        int i = static_cast<int>(row % T);
        float *s = scores + row * T;
        float *p = probs + row * T;

        float m = -INFINITY;
        for (int j = lane; j <= i; j += 32)
        {
            m = fmaxf(m, s[j]);
        }
        m = warp_max(m);                    // finite: lane 0 always sees j = 0

        float sum = 0.0f;
        for (int j = lane; j <= i; j += 32)
        {
            sum += expf(s[j] - m);
        }
        sum = warp_sum(sum);

        for (int j = lane; j < T; j += 32)
        {
            if (j <= i)
            {
                p[j] = expf(s[j] - m) / sum;
            }
            else
            {
                s[j] = -INFINITY;
                p[j] = 0.0f;
            }
        }
    }
}

// dscores[i,j] += probs[i,j] * (dprobs[i,j] - sum_k probs[i,k] * dprobs[i,k])
__global__ void causal_softmax_backward_kernel(float *dscores, const float *dprobs, const float *probs,
                                               int T, long long rows)
{
    long long row = static_cast<long long>(blockIdx.x) * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < rows)
    {
        const float *p = probs + row * T;
        const float *dp = dprobs + row * T;
        float *ds = dscores + row * T;

        float dot = 0.0f;
        for (int j = lane; j < T; j += 32)
        {
            dot += p[j] * dp[j];
        }
        dot = warp_sum(dot);

        for (int j = lane; j < T; j += 32)
        {
            ds[j] += p[j] * (dp[j] - dot);
        }
    }
}

// ---------------------------------------------------------------- launchers

void attention_permute(float *q, float *k, float *v, const float *qkv, int B, int T, int H, int D, cudaStream_t stream)
{
    assert(B > 0 && T > 0 && H > 0 && D > 0);
    long long total = static_cast<long long>(B) * H * T * D;
    permute_kernel<<<ceil_div(total, BLOCK), BLOCK, 0, stream>>>(q, k, v, qkv, T, H, D, total);
    CUDA_CHECK_LAUNCH();
}

void attention_permute_backward(float *dqkv, const float *dq, const float *dk, const float *dv,
                                int B, int T, int H, int D, cudaStream_t stream)
{
    assert(B > 0 && T > 0 && H > 0 && D > 0);
    long long total = static_cast<long long>(B) * H * T * D;
    permute_backward_kernel<<<ceil_div(total, BLOCK), BLOCK, 0, stream>>>(dqkv, dq, dk, dv, T, H, D, total);
    CUDA_CHECK_LAUNCH();
}

void attention_unpermute(float *atty, const float *y, int B, int T, int H, int D, cudaStream_t stream)
{
    assert(B > 0 && T > 0 && H > 0 && D > 0);
    long long total = static_cast<long long>(B) * T * H * D;
    unpermute_kernel<<<ceil_div(total, BLOCK), BLOCK, 0, stream>>>(atty, y, T, H, D, total);
    CUDA_CHECK_LAUNCH();
}

void attention_unpermute_backward(float *dy, const float *datty, int B, int T, int H, int D, cudaStream_t stream)
{
    assert(B > 0 && T > 0 && H > 0 && D > 0);
    long long total = static_cast<long long>(B) * T * H * D;
    unpermute_backward_kernel<<<ceil_div(total, BLOCK), BLOCK, 0, stream>>>(dy, datty, T, H, D, total);
    CUDA_CHECK_LAUNCH();
}

void causal_softmax(float *probs, float *scores, int B, int H, int T, cudaStream_t stream)
{
    assert(B > 0 && H > 0 && T > 0);
    long long rows = static_cast<long long>(B) * H * T;
    causal_softmax_kernel<<<ceil_div(rows, ROWS_PER_BLOCK), BLOCK, 0, stream>>>(probs, scores, T, rows);
    CUDA_CHECK_LAUNCH();
}

void causal_softmax_backward(float *dscores, const float *dprobs, const float *probs, int B, int H, int T,
                             cudaStream_t stream)
{
    assert(B > 0 && H > 0 && T > 0);
    long long rows = static_cast<long long>(B) * H * T;
    causal_softmax_backward_kernel<<<ceil_div(rows, ROWS_PER_BLOCK), BLOCK, 0, stream>>>(dscores, dprobs, probs, T,
                                                                                         rows);
    CUDA_CHECK_LAUNCH();
}

void attention_forward(float *atty, float *q, float *k, float *v, float *scores, float *probs, float *y,
                       const float *qkv, int B, int T, int H, int D, cudaStream_t stream)
{
    long long td = static_cast<long long>(T) * D;
    long long tt = static_cast<long long>(T) * T;
    float scale = 1.0f / sqrtf(static_cast<float>(D));

    attention_permute(q, k, v, qkv, B, T, H, D, stream);
    // scores = q @ k^T / sqrt(D): batch B*H, M T, N T, K D, transB
    sgemm(false, true, T, T, D, scale, q, D, td, k, D, td, 0.0f, scores, T, tt, B * H, stream);
    causal_softmax(probs, scores, B, H, T, stream);
    // y = probs @ v: batch B*H, M T, N D, K T
    sgemm(false, false, T, D, T, 1.0f, probs, T, tt, v, D, td, 0.0f, y, D, td, B * H, stream);
    attention_unpermute(atty, y, B, T, H, D, stream);
}

void attention_backward(float *dqkv, float *dq, float *dk, float *dv, float *dscores, float *dprobs, float *dy,
                        const float *datty, const float *q, const float *k, const float *v, const float *probs,
                        int B, int T, int H, int D, cudaStream_t stream)
{
    long long td = static_cast<long long>(T) * D;
    long long tt = static_cast<long long>(T) * T;
    float scale = 1.0f / sqrtf(static_cast<float>(D));

    attention_unpermute_backward(dy, datty, B, T, H, D, stream);
    // dprobs += dy @ v^T: M T, N T, K D, transB
    sgemm(false, true, T, T, D, 1.0f, dy, D, td, v, D, td, 1.0f, dprobs, T, tt, B * H, stream);
    // dv += probs^T @ dy: transA, M T, N D, K T
    sgemm(true, false, T, D, T, 1.0f, probs, T, tt, dy, D, td, 1.0f, dv, D, td, B * H, stream);
    causal_softmax_backward(dscores, dprobs, probs, B, H, T, stream);
    // dq += dscores @ k / sqrt(D): M T, N D, K T
    sgemm(false, false, T, D, T, scale, dscores, T, tt, k, D, td, 1.0f, dq, D, td, B * H, stream);
    // dk += dscores^T @ q / sqrt(D): transA, M T, N D, K T
    sgemm(true, false, T, D, T, scale, dscores, T, tt, q, D, td, 1.0f, dk, D, td, B * H, stream);
    attention_permute_backward(dqkv, dq, dk, dv, B, T, H, D, stream);
}
