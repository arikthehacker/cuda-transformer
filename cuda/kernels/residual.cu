// residual.cu: out = a + b, and its backward (the gradient passes through to both)

#include <cassert>

#include "common.h"
#include "kernels.h"

using namespace std;

static const int BLOCK = 256;

__global__ void residual_forward_kernel(float *out, const float *a, const float *b, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void accumulate_kernel(float *dst, const float *src, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] += src[idx];
    }
}

void residual_forward(float *out, const float *a, const float *b, long long n)
{
    assert(n > 0);
    residual_forward_kernel<<<ceil_div(n, BLOCK), BLOCK>>>(out, a, b, n);
    CUDA_CHECK_LAUNCH();
}

void accumulate(float *dst, const float *src, long long n)
{
    assert(n > 0);
    accumulate_kernel<<<ceil_div(n, BLOCK), BLOCK>>>(dst, src, n);
    CUDA_CHECK_LAUNCH();
}
