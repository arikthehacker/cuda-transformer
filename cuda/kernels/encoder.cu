// encoder.cu: token + position embedding, forward and backward

#include <cassert>

#include "common.h"
#include "kernels.h"

using namespace std;

static const int BLOCK = 256;

// one thread per output value, walking C fastest (coalesced)
__global__ void encoder_forward_kernel(float *out, const int *tok, const float *wte, const float *wpe,
                                       int T, int C, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        long long bt = idx / C;
        int c = static_cast<int>(idx % C);
        int t = static_cast<int>(bt % T);
        out[idx] = wte[static_cast<long long>(tok[bt]) * C + c] + wpe[static_cast<long long>(t) * C + c];
    }
}

// one thread per (b, t, c). The same token can sit at many (b, t), so the
// adds into its wte row race: atomicAdd. The add order changes between runs.
__global__ void encoder_backward_wte_kernel(float *dwte, const float *dout, const int *tok,
                                            int C, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        long long bt = idx / C;
        int c = static_cast<int>(idx % C);
        atomicAdd(&dwte[static_cast<long long>(tok[bt]) * C + c], dout[idx]);
    }
}

// one thread per (t, c), looping over b itself: no atomics, fixed order
__global__ void encoder_backward_wpe_kernel(float *dwpe, const float *dout, int B, int T, int C)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(T) * C)
    {
        long long t = idx / C;
        int c = static_cast<int>(idx % C);
        float sum = 0.0f;
        for (int b = 0; b < B; b++)
        {
            sum += dout[(static_cast<long long>(b) * T + t) * C + c];
        }
        dwpe[idx] += sum;
    }
}

void encoder_forward(float *out, const int *tok, const float *wte, const float *wpe, int B, int T, int C)
{
    assert(B > 0 && T > 0 && C > 0);
    long long total = static_cast<long long>(B) * T * C;
    encoder_forward_kernel<<<ceil_div(total, BLOCK), BLOCK>>>(out, tok, wte, wpe, T, C, total);
    CUDA_CHECK_LAUNCH();
}

void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *tok, int B, int T, int C)
{
    assert(B > 0 && T > 0 && C > 0);
    long long total = static_cast<long long>(B) * T * C;
    encoder_backward_wte_kernel<<<ceil_div(total, BLOCK), BLOCK>>>(dwte, dout, tok, C, total);
    CUDA_CHECK_LAUNCH();
    encoder_backward_wpe_kernel<<<ceil_div(static_cast<long long>(T) * C, BLOCK), BLOCK>>>(dwpe, dout, B, T, C);
    CUDA_CHECK_LAUNCH();
}
