// gelu.cu: GELU, tanh version, forward and backward

#include <cassert>
#include <cmath>

#include "common.h"
#include "kernels.h"

using namespace std;

static const int BLOCK = 256;

// s = sqrt(2/pi), u = s * (x + 0.044715 x^3), out = 0.5 * x * (1 + tanh(u))
__global__ void gelu_forward_kernel(float *out, const float *x, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const float s = 0.7978845608028654f;
        float xi = x[idx];
        float u = s * (xi + 0.044715f * xi * xi * xi);
        out[idx] = 0.5f * xi * (1.0f + tanhf(u));
    }
}

// dx += dout * (0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * s * (1 + 3 * 0.044715 x^2))
__global__ void gelu_backward_kernel(float *dx, const float *x, const float *dout, long long n)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const float s = 0.7978845608028654f;
        float xi = x[idx];
        float th = tanhf(s * (xi + 0.044715f * xi * xi * xi));
        float left = 0.5f * (1.0f + th);
        float right = 0.5f * xi * (1.0f - th * th) * s * (1.0f + 3.0f * 0.044715f * xi * xi);
        dx[idx] += dout[idx] * (left + right);
    }
}

void gelu_forward(float *out, const float *x, long long n)
{
    assert(n > 0);
    gelu_forward_kernel<<<ceil_div(n, BLOCK), BLOCK>>>(out, x, n);
    CUDA_CHECK_LAUNCH();
}

void gelu_backward(float *dx, const float *x, const float *dout, long long n)
{
    assert(n > 0);
    gelu_backward_kernel<<<ceil_div(n, BLOCK), BLOCK>>>(dx, x, dout, n);
    CUDA_CHECK_LAUNCH();
}
