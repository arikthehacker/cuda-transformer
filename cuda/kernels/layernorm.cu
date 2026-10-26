// layernorm.cu: forward and backward, one warp per row

#include <cassert>

#include "common.h"
#include "kernels.h"
#include "reduce.cuh"

using namespace std;

static const int BLOCK = 256;               // 8 warps = 8 rows per block
static const int ROWS_PER_BLOCK = BLOCK / 32;
static const int GROUPS = 8;                // dw/db kernel: 8 groups of 32 columns' worth of rows

// mean = sum(x)/C, var = sum((x-mean)^2)/C, rstd = 1/sqrt(var + eps)
// out = (x - mean) * rstd * w + b. Each lane handles columns lane, lane+32, ...
__global__ void layernorm_forward_kernel(float *out, float *mean, float *rstd, const float *x,
                                         const float *w, const float *b, int N, int C, float eps)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < N)                            // the whole warp shares the row, so shuffles stay full
    {
        const float *xr = x + static_cast<long long>(row) * C;
        float *o = out + static_cast<long long>(row) * C;

        float sum = 0.0f;
        for (int c = lane; c < C; c += 32)
        {
            sum += xr[c];
        }
        float m = warp_sum(sum) / C;

        float sq = 0.0f;
        for (int c = lane; c < C; c += 32)
        {
            float d = xr[c] - m;
            sq += d * d;
        }
        float s = 1.0f / sqrtf(warp_sum(sq) / C + eps);

        for (int c = lane; c < C; c += 32)
        {
            o[c] = (xr[c] - m) * s * w[c] + b[c];
        }
        if (lane == 0)
        {
            mean[row] = m;
            rstd[row] = s;
        }
    }
}

// dx += rstd * (g - mean(g) - xhat * mean(g * xhat)),  g = dout * w
__global__ void layernorm_backward_dx_kernel(float *dx, const float *dout, const float *x, const float *w,
                                             const float *mean, const float *rstd, int N, int C)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < N)
    {
        long long base = static_cast<long long>(row) * C;
        float m = mean[row];
        float s = rstd[row];

        float sum_g = 0.0f;
        float sum_gx = 0.0f;
        for (int c = lane; c < C; c += 32)
        {
            float xhat = (x[base + c] - m) * s;
            float g = dout[base + c] * w[c];
            sum_g += g;
            sum_gx += g * xhat;
        }
        float mean_g = warp_sum(sum_g) / C;
        float mean_gx = warp_sum(sum_gx) / C;

        for (int c = lane; c < C; c += 32)
        {
            float xhat = (x[base + c] - m) * s;
            float g = dout[base + c] * w[c];
            dx[base + c] += s * (g - mean_g - xhat * mean_gx);
        }
    }
}

// dw += sum over rows of dout * xhat, db += sum over rows of dout.
// A block owns 32 columns. Its 8 warps take every 8th row (warp k: rows k,
// k+8, ...), reading 32 neighboring columns at a time, then warp 0 adds the
// 8 partial sums in a fixed order. No atomics: the same bits every run.
__global__ void layernorm_backward_dwdb_kernel(float *dw, float *db, const float *dout, const float *x,
                                               const float *mean, const float *rstd, int N, int C)
{
    __shared__ float part_w[GROUPS][32];
    __shared__ float part_b[GROUPS][32];
    int lane = threadIdx.x % 32;
    int group = threadIdx.x / 32;
    int c = blockIdx.x * 32 + lane;

    float sw = 0.0f;
    float sb = 0.0f;
    if (c < C)
    {
        for (int n = group; n < N; n += GROUPS)
        {
            long long i = static_cast<long long>(n) * C + c;
            float xhat = (x[i] - mean[n]) * rstd[n];
            sw += dout[i] * xhat;
            sb += dout[i];
        }
    }
    part_w[group][lane] = sw;
    part_b[group][lane] = sb;
    __syncthreads();

    if (group == 0 && c < C)
    {
        float tw = 0.0f;
        float tb = 0.0f;
        for (int g = 0; g < GROUPS; g++)
        {
            tw += part_w[g][lane];
            tb += part_b[g][lane];
        }
        dw[c] += tw;
        db[c] += tb;
    }
}

void layernorm_forward(float *out, float *mean, float *rstd, const float *x, const float *w, const float *b,
                       int N, int C, float eps)
{
    assert(N > 0 && C > 0);
    layernorm_forward_kernel<<<ceil_div(N, ROWS_PER_BLOCK), BLOCK>>>(out, mean, rstd, x, w, b, N, C, eps);
    CUDA_CHECK_LAUNCH();
}

void layernorm_backward(float *dx, float *dw, float *db, const float *dout, const float *x, const float *w,
                        const float *mean, const float *rstd, int N, int C)
{
    assert(N > 0 && C > 0);
    layernorm_backward_dx_kernel<<<ceil_div(N, ROWS_PER_BLOCK), BLOCK>>>(dx, dout, x, w, mean, rstd, N, C);
    CUDA_CHECK_LAUNCH();
    layernorm_backward_dwdb_kernel<<<ceil_div(C, 32), GROUPS * 32>>>(dw, db, dout, x, mean, rstd, N, C);
    CUDA_CHECK_LAUNCH();
}
