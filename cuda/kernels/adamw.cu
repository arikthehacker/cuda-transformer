// adamw.cu: one AdamW step, the global gradient norm, and scaling (for clipping)

#include <cassert>
#include <cmath>
#include <vector>

#include "common.h"
#include "kernels.h"
#include "reduce.cuh"

using namespace std;

static const int BLOCK = 256;
static const int NORM_BLOCKS = 512;         // fixed, so the partial sums add up the same way every time

// Written the way torch.optim.AdamW does it, so the float rounding matches:
//   p *= decay                 decay = 1 - lr * wd
//   m += (1 - b1) * (g - m)    (lerp)
//   v = v * b2 + (1 - b2) * g * g
//   p -= step_size * m / (sqrt(v) / bc2_sqrt + eps)
//   step_size = lr / (1 - b1^t), bc2_sqrt = sqrt(1 - b2^t)
__global__ void adamw_kernel(float *p, const float *g, float *m, float *v, long long n,
                             float b1, float b2, float eps, float decay, float step_size, float bc2_sqrt)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float grad = g[idx];
        float param = p[idx] * decay;
        float mi = m[idx] + (1.0f - b1) * (grad - m[idx]);
        float vi = v[idx] * b2 + (1.0f - b2) * grad * grad;
        float denom = sqrtf(vi) / bc2_sqrt + eps;
        p[idx] = param - step_size * (mi / denom);
        m[idx] = mi;
        v[idx] = vi;
    }
}

// Each block adds the squares of its grid-stride share, then block_sum.
__global__ void sum_squares_kernel(float *partial, const float *g, long long n)
{
    float sum = 0.0f;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        sum += g[i] * g[i];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = sum;
    }
}

__global__ void scale_kernel(float *x, float s, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        x[idx] *= s;
    }
}

void adamw_step(float *p, const float *g, float *m, float *v, long long n,
                float lr, float b1, float b2, float eps, float wd, int t)
{
    assert(n > 0 && t >= 1);
    // the scalars in double, like PyTorch's Python-side math, then rounded once
    double bc1 = 1.0 - pow(static_cast<double>(b1), t);
    double bc2 = 1.0 - pow(static_cast<double>(b2), t);
    float decay = static_cast<float>(1.0 - static_cast<double>(lr) * wd);
    float step_size = static_cast<float>(lr / bc1);
    float bc2_sqrt = static_cast<float>(sqrt(bc2));
    adamw_kernel<<<ceil_div(n, BLOCK), BLOCK>>>(p, g, m, v, n, b1, b2, eps, decay, step_size, bc2_sqrt);
    CUDA_CHECK_LAUNCH();
}

float grad_norm(const float *g, long long n)
{
    assert(n > 0);
    static DeviceBuffer partial(NORM_BLOCKS);   // made once, reused every call
    sum_squares_kernel<<<NORM_BLOCKS, BLOCK>>>(partial.ptr, g, n);
    CUDA_CHECK_LAUNCH();
    vector<float> host = partial.download();
    double total = 0.0;
    for (float s : host)
    {
        total += s;
    }
    return static_cast<float>(sqrt(total));
}

void scale(float *x, float s, long long n)
{
    assert(n > 0);
    scale_kernel<<<ceil_div(n, BLOCK), BLOCK>>>(x, s, n);
    CUDA_CHECK_LAUNCH();
}
