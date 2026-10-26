// crossentropy.cu: softmax + loss over the V real columns, and its gradient.
// One block per row. Padded columns V .. Vp-1 are never read or written.

#include <cassert>
#include <cmath>

#include "common.h"
#include "kernels.h"
#include "reduce.cuh"

using namespace std;

static const int BLOCK = 256;

// m = max_j<V l[j], lse = m + log(sum_j<V exp(l[j] - m)), losses[n] = lse - l[target]
__global__ void crossentropy_forward_kernel(float *losses, const float *logits, const int *targets, int V, int Vp)
{
    int n = blockIdx.x;
    const float *l = logits + static_cast<long long>(n) * Vp;

    float m = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        m = fmaxf(m, l[j]);
    }
    m = block_max(m);

    float sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        sum += expf(l[j] - m);
    }
    sum = block_sum(sum);

    if (threadIdx.x == 0)
    {
        losses[n] = m + logf(sum) - l[targets[n]];
    }
}

// dlogits[n,j] += (softmax[n,j] - (j == target)) / N, j < V
__global__ void crossentropy_backward_kernel(float *dlogits, const float *logits, const int *targets,
                                             int N, int V, int Vp)
{
    int n = blockIdx.x;
    const float *l = logits + static_cast<long long>(n) * Vp;
    float *dl = dlogits + static_cast<long long>(n) * Vp;
    int target = targets[n];

    float m = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        m = fmaxf(m, l[j]);
    }
    m = block_max(m);

    float sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        sum += expf(l[j] - m);
    }
    sum = block_sum(sum);

    for (int j = threadIdx.x; j < V; j += blockDim.x)
    {
        float p = expf(l[j] - m) / sum;
        float hit = (j == target) ? 1.0f : 0.0f;
        dl[j] += (p - hit) / N;
    }
}

void crossentropy_forward(float *losses, const float *logits, const int *targets, int N, int V, int Vp)
{
    assert(N > 0 && V > 0 && V <= Vp);
    crossentropy_forward_kernel<<<N, BLOCK>>>(losses, logits, targets, V, Vp);
    CUDA_CHECK_LAUNCH();
}

void crossentropy_backward(float *dlogits, const float *logits, const int *targets, int N, int V, int Vp)
{
    assert(N > 0 && V > 0 && V <= Vp);
    crossentropy_backward_kernel<<<N, BLOCK>>>(dlogits, logits, targets, N, V, Vp);
    CUDA_CHECK_LAUNCH();
}
