// fused.cu: the small kernels of the fast model, each saving at least one
// trip through memory compared to stage 3.
//   residual + LayerNorm      one pass writes the fp32 residual AND its LayerNorm
//   LayerNorm backward        adds into the fp32 residual gradient and writes the
//                             floatX copy the next GEMMs read, in the same pass
//   cross entropy in place    reads a logits row, writes the loss, overwrites
//                             the row with dlogits (logits is the biggest activation)
//   GELU recompute            the backward pass rebuilds gelu from fc, so no
//                             (B, T, 4C) gelu buffer is kept per layer

#include <algorithm>
#include <cassert>
#include <cmath>

#include "common.h"
#include "fast.h"
#include "gemm_common.cuh"
#include "reduce.cuh"

using namespace std;

static const int BLOCK = 256;
static const int ROWS = BLOCK / 32;         // row kernels: one warp per row
static const int GROUPS = 8;                // column sums: 8 warps, warp g takes every 8th row of its chunk

// ---------------------------------------------------------------- LayerNorm

__device__ __forceinline__ void layernorm_row(floatX *o, float *mean, float *rstd, const float *xr, const float *w,
                                              const float *b, int row, int C, float eps, int lane)
{
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
        o[c] = from_float<floatX>((xr[c] - m) * s * w[c] + b[c]);
    }
    if (lane == 0)
    {
        mean[row] = m;
        rstd[row] = s;
    }
}

__global__ void layernorm_fwd_kernel(floatX *out, float *mean, float *rstd, const float *x, const float *w,
                                     const float *b, int N, int C, float eps)
{
    int row = blockIdx.x * ROWS + threadIdx.x / 32;
    if (row < N)
    {
        layernorm_row(out + static_cast<long long>(row) * C, mean, rstd, x + static_cast<long long>(row) * C, w, b,
                      row, C, eps, threadIdx.x % 32);
    }
}

// res = x + y, then LayerNorm(res). The row of res is written first and read
// back by the same lanes (it is still in L1), so nothing races.
__global__ void residual_layernorm_kernel(float *res, floatX *out, float *mean, float *rstd, const float *x,
                                          const floatX *y, const float *w, const float *b, int N, int C, float eps)
{
    int row = blockIdx.x * ROWS + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < N)
    {
        long long base = static_cast<long long>(row) * C;
        for (int c = lane; c < C; c += 32)
        {
            res[base + c] = x[base + c] + to_float(y[base + c]);
        }
        layernorm_row(out + base, mean, rstd, res + base, w, b, row, C, eps, lane);
    }
}

// dres += rstd * (g - mean(g) - xhat * mean(g * xhat)), g = dout * w; dres_h = floatX(dres)
__global__ void layernorm_bwd_dx_kernel(float *dres, floatX *dres_h, const floatX *dout, const float *x,
                                        const float *w, const float *mean, const float *rstd, int N, int C)
{
    int row = blockIdx.x * ROWS + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < N)
    {
        long long base = static_cast<long long>(row) * C;
        float m = mean[row], s = rstd[row];
        float sum_g = 0.0f, sum_gx = 0.0f;
        for (int c = lane; c < C; c += 32)
        {
            float xhat = (x[base + c] - m) * s;
            float g = to_float(dout[base + c]) * w[c];
            sum_g += g;
            sum_gx += g * xhat;
        }
        float mean_g = warp_sum(sum_g) / C;
        float mean_gx = warp_sum(sum_gx) / C;
        for (int c = lane; c < C; c += 32)
        {
            float xhat = (x[base + c] - m) * s;
            float g = to_float(dout[base + c]) * w[c];
            float v = dres[base + c] + s * (g - mean_g - xhat * mean_gx);
            dres[base + c] = v;
            dres_h[base + c] = from_float<floatX>(v);
        }
    }
}

// ---------------------------------------------------------------- column sums
// Parameter gradients that are sums over all N rows (bias, LayerNorm w and b)
// in two passes, so the work spreads over many blocks and the result does
// not depend on timing: block (x, y) adds rows of chunk y for 32 columns into
// partial[y][c], then one thread per column adds the COLSUM_SPLITS partials
// in order. LN: a = dout * xhat, b = dout. Otherwise a = dout.
template <bool LN>
__global__ void colsum_partial_kernel(float *part_a, float *part_b, const floatX *dout, const float *x,
                                      const float *mean, const float *rstd, int N, int C)
{
    __shared__ float sa[GROUPS][32];
    __shared__ float sb[GROUPS][32];
    int lane = threadIdx.x % 32, group = threadIdx.x / 32;
    int c = blockIdx.x * 32 + lane;
    int chunk = (N + gridDim.y - 1) / gridDim.y;
    int r0 = blockIdx.y * chunk;
    int r1 = min(N, r0 + chunk);
    float a = 0.0f, b = 0.0f;
    if (c < C)
    {
        for (int n = r0 + group; n < r1; n += GROUPS)
        {
            long long i = static_cast<long long>(n) * C + c;
            float d = to_float(dout[i]);
            if (LN)
            {
                a += d * (x[i] - mean[n]) * rstd[n];
                b += d;
            }
            else
            {
                a += d;
            }
        }
    }
    sa[group][lane] = a;
    sb[group][lane] = b;
    __syncthreads();
    if (group == 0 && c < C)
    {
        float ta = 0.0f, tb = 0.0f;
        for (int g = 0; g < GROUPS; g++)
        {
            ta += sa[g][lane];
            tb += sb[g][lane];
        }
        part_a[blockIdx.y * C + c] = ta;
        if (LN)
        {
            part_b[blockIdx.y * C + c] = tb;
        }
    }
}

__global__ void colsum_final_kernel(float *out_a, float *out_b, const float *part_a, const float *part_b, int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C)
    {
        float a = 0.0f, b = 0.0f;
        for (int k = 0; k < COLSUM_SPLITS; k++)
        {
            a += part_a[k * C + c];
            if (out_b != nullptr)
            {
                b += part_b[k * C + c];
            }
        }
        out_a[c] += a;
        if (out_b != nullptr)
        {
            out_b[c] += b;
        }
    }
}

void layernorm_fwd(floatX *out, float *mean, float *rstd, const float *x, const float *w, const float *b, int N, int C,
                   float eps, cudaStream_t s)
{
    assert(N > 0 && C > 0);
    layernorm_fwd_kernel<<<ceil_div(N, ROWS), BLOCK, 0, s>>>(out, mean, rstd, x, w, b, N, C, eps);
    CUDA_CHECK_LAUNCH();
}

void residual_layernorm_fwd(float *res, floatX *out, float *mean, float *rstd, const float *x, const floatX *y,
                            const float *w, const float *b, int N, int C, float eps, cudaStream_t s)
{
    assert(N > 0 && C > 0);
    residual_layernorm_kernel<<<ceil_div(N, ROWS), BLOCK, 0, s>>>(res, out, mean, rstd, x, y, w, b, N, C, eps);
    CUDA_CHECK_LAUNCH();
}

void layernorm_bwd(float *dres, floatX *dres_h, float *dw, float *db, const floatX *dout, const float *x,
                   const float *w, const float *mean, const float *rstd, int N, int C, float *scratch, cudaStream_t s)
{
    assert(N > 0 && C > 0);
    layernorm_bwd_dx_kernel<<<ceil_div(N, ROWS), BLOCK, 0, s>>>(dres, dres_h, dout, x, w, mean, rstd, N, C);
    CUDA_CHECK_LAUNCH();
    float *part_w = scratch, *part_b = scratch + COLSUM_SPLITS * C;
    dim3 grid(ceil_div(C, 32), COLSUM_SPLITS);
    colsum_partial_kernel<true><<<grid, GROUPS * 32, 0, s>>>(part_w, part_b, dout, x, mean, rstd, N, C);
    CUDA_CHECK_LAUNCH();
    colsum_final_kernel<<<ceil_div(C, BLOCK), BLOCK, 0, s>>>(dw, db, part_w, part_b, C);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- encoder

__global__ void encoder_fwd_kernel(float *emb, const int *tok, const float *wte, const float *wpe, int T, int C,
                                   long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        long long bt = idx / C;
        int c = static_cast<int>(idx % C);
        emb[idx] = wte[static_cast<long long>(tok[bt]) * C + c] + wpe[(bt % T) * C + c];
    }
}

__global__ void encoder_bwd_wte_kernel(float *dwte, const float *dres, const int *tok, int C, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        atomicAdd(&dwte[static_cast<long long>(tok[idx / C]) * C + idx % C], dres[idx]);
    }
}

__global__ void encoder_bwd_wpe_kernel(float *dwpe, const float *dres, int B, int T, int C)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(T) * C)
    {
        float sum = 0.0f;
        for (int b = 0; b < B; b++)
        {
            sum += dres[static_cast<long long>(b) * T * C + idx];
        }
        dwpe[idx] += sum;
    }
}

void encoder_fwd(float *emb, const int *tok, const float *wte, const float *wpe, int B, int T, int C, cudaStream_t s)
{
    long long total = static_cast<long long>(B) * T * C;
    encoder_fwd_kernel<<<ceil_div(total, BLOCK), BLOCK, 0, s>>>(emb, tok, wte, wpe, T, C, total);
    CUDA_CHECK_LAUNCH();
}

void encoder_bwd(float *dwte, float *dwpe, const float *dres, const int *tok, int B, int T, int C, cudaStream_t s)
{
    long long total = static_cast<long long>(B) * T * C;
    encoder_bwd_wte_kernel<<<ceil_div(total, BLOCK), BLOCK, 0, s>>>(dwte, dres, tok, C, total);
    CUDA_CHECK_LAUNCH();
    encoder_bwd_wpe_kernel<<<ceil_div(static_cast<long long>(T) * C, BLOCK), BLOCK, 0, s>>>(dwpe, dres, B, T, C);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- bias, gelu, casts

__global__ void gelu_fwd_kernel(floatX *out, const floatX *in, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = from_float<floatX>(gelu_f(to_float(in[idx])));    // the same gelu_f as the GEMM epilogue
    }
}

__global__ void cast_kernel(floatX *out, const float *in, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = from_float<floatX>(in[idx]);
    }
}

__global__ void uncast_kernel(float *out, const floatX *in, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = to_float(in[idx]);
    }
}

void bias_bwd(float *dbias, const floatX *dout, int N, int OC, float *scratch, cudaStream_t s)
{
    assert(N > 0 && OC > 0);
    dim3 grid(ceil_div(OC, 32), COLSUM_SPLITS);
    colsum_partial_kernel<false><<<grid, GROUPS * 32, 0, s>>>(scratch, nullptr, dout, nullptr, nullptr, nullptr, N, OC);
    CUDA_CHECK_LAUNCH();
    colsum_final_kernel<<<ceil_div(OC, BLOCK), BLOCK, 0, s>>>(dbias, nullptr, scratch, nullptr, OC);
    CUDA_CHECK_LAUNCH();
}

void gelu_fwd(floatX *out, const floatX *in, long long n, cudaStream_t s)
{
    gelu_fwd_kernel<<<ceil_div(n, BLOCK), BLOCK, 0, s>>>(out, in, n);
    CUDA_CHECK_LAUNCH();
}

void cast_to_x(floatX *out, const float *in, long long n, cudaStream_t s)
{
    cast_kernel<<<ceil_div(n, BLOCK), BLOCK, 0, s>>>(out, in, n);
    CUDA_CHECK_LAUNCH();
}

void cast_to_f(float *out, const floatX *in, long long n, cudaStream_t s)
{
    uncast_kernel<<<ceil_div(n, BLOCK), BLOCK, 0, s>>>(out, in, n);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- cross entropy

// One block per row. Every thread owns columns j = tid, tid + 256, ... in all
// three passes, so the in-place overwrite never touches a value another
// thread still has to read. l[target] is read before the first barrier.
__global__ void crossentropy_kernel(float *losses, floatX *logits, const int *targets, int N, int V, int Vp,
                                    const float *scale, const int *count, bool grad)
{
    int n = blockIdx.x;
    floatX *l = logits + static_cast<long long>(n) * Vp;
    int target = targets[n];
    if (target < 0)                         // not counted: no loss, no gradient (the whole block takes this branch)
    {
        if (threadIdx.x == 0)
        {
            losses[n] = 0.0f;
        }
        if (grad)
        {
            for (int j = threadIdx.x; j < Vp; j += blockDim.x)
            {
                l[j] = from_float<floatX>(0.0f);
            }
        }
        return;
    }
    float lt = to_float(l[target]);

    float m = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        m = fmaxf(m, to_float(l[j]));
    }
    m = block_max(m);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        sum += expf(to_float(l[j]) - m);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0)
    {
        losses[n] = m + logf(sum) - lt;
    }
    if (grad)
    {
        float k = *scale / (count ? *count : N);    // loss scale, and the mean over the counted rows
        for (int j = threadIdx.x; j < Vp; j += blockDim.x)
        {
            float d = 0.0f;
            if (j < V)
            {
                float p = expf(to_float(l[j]) - m) / sum;
                d = (p - (j == target ? 1.0f : 0.0f)) * k;
            }
            l[j] = from_float<floatX>(d);
        }
    }
}

__global__ void __launch_bounds__(1024) mean_loss_kernel(float *out, const float *losses, int N, const int *count)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        sum += losses[i];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0)
    {
        out[0] = sum / (count ? *count : N);
    }
}

// One block of 1024 threads, an exact integer tree sum.
__global__ void __launch_bounds__(1024) count_targets_kernel(StepState *st, const int *targets, int N)
{
    __shared__ int part[1024];
    int c = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        c += targets[i] >= 0 ? 1 : 0;
    }
    part[threadIdx.x] = c;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k /= 2)
    {
        if (static_cast<int>(threadIdx.x) < k)
        {
            part[threadIdx.x] += part[threadIdx.x + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        st->n_valid = max(part[0], 1);
    }
}

void crossentropy_fused(float *losses, floatX *logits, const int *targets, int N, int V, int Vp, const float *scale,
                        const int *count, bool grad, cudaStream_t s)
{
    assert(N > 0 && V > 0 && V <= Vp);
    crossentropy_kernel<<<N, BLOCK, 0, s>>>(losses, logits, targets, N, V, Vp, scale, count, grad);
    CUDA_CHECK_LAUNCH();
}

void mean_loss(float *out, const float *losses, int N, const int *count, cudaStream_t s)
{
    mean_loss_kernel<<<1, 1024, 0, s>>>(out, losses, N, count);
    CUDA_CHECK_LAUNCH();
}

void count_targets(StepState *st, const int *targets, int N, cudaStream_t s)
{
    assert(N > 0);
    count_targets_kernel<<<1, 1024, 0, s>>>(st, targets, N);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- batches

__global__ void load_batch_kernel(int *x, int *y, const uint16_t *tokens, const int *targets, const int *starts,
                                  int nsteps, const StepState *st, int B, int T)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T)
    {
        int b = idx / T, t = idx % T;
        long long start = starts[static_cast<long long>(st->step % nsteps) * B + b];
        x[idx] = tokens[start + t];
        y[idx] = targets ? targets[start + t] : static_cast<int>(tokens[start + t + 1]);
    }
}

void load_batch(int *x, int *y, const uint16_t *tokens, const int *targets, const int *starts, int nsteps,
                const StepState *st, int B, int T, cudaStream_t s)
{
    assert(B > 0 && T > 0 && nsteps > 0);
    load_batch_kernel<<<ceil_div(B * T, BLOCK), BLOCK, 0, s>>>(x, y, tokens, targets, starts, nsteps, st, B, T);
    CUDA_CHECK_LAUNCH();
}
