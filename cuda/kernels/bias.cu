// bias.cu: add bias, and the bias gradient (column sums)

#include <cassert>

#include "common.h"
#include "kernels.h"

using namespace std;

static const int BLOCK = 256;
static const int GROUPS = 8;                // column sums: 8 warps, each takes every 8th row

// out[n,o] += bias[o], one thread per value
__global__ void add_bias_kernel(float *out, const float *bias, int OC, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        out[idx] += bias[idx % OC];
    }
}

// dbias[o] += sum over n of dout[n,o]. A block owns 32 columns; warp k adds
// rows k, k+8, k+16, ... (32 neighboring columns per read), then warp 0 adds
// the 8 partial sums in a fixed order. No atomics.
__global__ void bias_backward_kernel(float *dbias, const float *dout, int N, int OC)
{
    __shared__ float part[GROUPS][32];
    int lane = threadIdx.x % 32;
    int group = threadIdx.x / 32;
    int o = blockIdx.x * 32 + lane;

    float sum = 0.0f;
    if (o < OC)
    {
        for (int n = group; n < N; n += GROUPS)
        {
            sum += dout[static_cast<long long>(n) * OC + o];
        }
    }
    part[group][lane] = sum;
    __syncthreads();

    if (group == 0 && o < OC)
    {
        float total = 0.0f;
        for (int g = 0; g < GROUPS; g++)
        {
            total += part[g][lane];
        }
        dbias[o] += total;
    }
}

void add_bias(float *out, const float *bias, int N, int OC)
{
    assert(N > 0 && OC > 0);
    long long total = static_cast<long long>(N) * OC;
    add_bias_kernel<<<ceil_div(total, BLOCK), BLOCK>>>(out, bias, OC, total);
    CUDA_CHECK_LAUNCH();
}

void bias_backward(float *dbias, const float *dout, int N, int OC)
{
    assert(N > 0 && OC > 0);
    bias_backward_kernel<<<ceil_div(OC, 32), GROUPS * 32>>>(dbias, dout, N, OC);
    CUDA_CHECK_LAUNCH();
}
