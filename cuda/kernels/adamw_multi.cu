// adamw_multi.cu: one AdamW launch for every parameter of the model (stage 3
// launched once per tensor, 16 L + 4 launches).
//
// Each thread updates 4 neighboring values. Which tensor they belong to (so
// whether weight decay applies) comes from a binary search over the tensor
// table, copied into shared memory once per block. The math is the stage 3
// kernel's, in torch's order, with the clip coefficient applied to the
// gradient first (torch scales the gradients in place, then steps).

#include <cassert>

#include "common.h"
#include "fast.h"

using namespace std;

static const int BLOCK = 256;
static const int MAX_GRID = 1024;           // grid stride past this

__device__ __forceinline__ float adamw_one(float &p, float &mi, float &vi, float g, float b1, float b2, float eps,
                                           float decay, float step_size, float bc2_sqrt)
{
    float param = p * decay;
    mi = mi + (1.0f - b1) * (g - mi);
    vi = vi * b2 + (1.0f - b2) * g * g;
    float denom = sqrtf(vi) / bc2_sqrt + eps;
    p = param - step_size * (mi / denom);
    return p;
}

__global__ void adamw_multi_kernel(float *params, floatX *params_x, const float *grads, float *m, float *v,
                                   long long n4, const TensorInfo *tensors, int ntensors, const StepState *st,
                                   float b1, float b2, float eps)
{
    __shared__ long long first[MAX_TENSORS];   // first value of each tensor
    __shared__ int decays[MAX_TENSORS];
    for (int k = threadIdx.x; k < ntensors; k += blockDim.x)
    {
        first[k] = tensors[k].offset;
        decays[k] = tensors[k].decay;
    }
    __syncthreads();
    if (st->found_inf)
    {
        return;                             // the whole step is skipped (every thread reads the same flag)
    }
    float coef = st->clip_coef;
    float step_size = st->step_size;
    float bc2_sqrt = st->bc2_sqrt;
    float wd_decay = st->decay;

    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride)
    {
        long long at = i * 4;
        int lo = 0, hi = ntensors - 1;      // the last tensor whose first value is <= at
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (first[mid] <= at)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        float decay = decays[lo] ? wd_decay : 1.0f;

        float4 p = reinterpret_cast<float4 *>(params)[i];
        float4 g = reinterpret_cast<const float4 *>(grads)[i];
        float4 mm = reinterpret_cast<float4 *>(m)[i];
        float4 vv = reinterpret_cast<float4 *>(v)[i];
        adamw_one(p.x, mm.x, vv.x, g.x * coef, b1, b2, eps, decay, step_size, bc2_sqrt);
        adamw_one(p.y, mm.y, vv.y, g.y * coef, b1, b2, eps, decay, step_size, bc2_sqrt);
        adamw_one(p.z, mm.z, vv.z, g.z * coef, b1, b2, eps, decay, step_size, bc2_sqrt);
        adamw_one(p.w, mm.w, vv.w, g.w * coef, b1, b2, eps, decay, step_size, bc2_sqrt);
        reinterpret_cast<float4 *>(params)[i] = p;
        reinterpret_cast<float4 *>(m)[i] = mm;
        reinterpret_cast<float4 *>(v)[i] = vv;
        if (params_x != nullptr)
        {
            floatX *px = params_x + at;
            px[0] = from_float<floatX>(p.x);
            px[1] = from_float<floatX>(p.y);
            px[2] = from_float<floatX>(p.z);
            px[3] = from_float<floatX>(p.w);
        }
    }
}

void adamw_multi(float *params, floatX *params_x, const float *grads, float *m, float *v, long long n,
                 const TensorInfo *tensors, int ntensors, const StepState *st, float beta1, float beta2,
                 float eps, cudaStream_t s)
{
    assert(n > 0 && n % 4 == 0);
    assert(ntensors > 0 && ntensors <= MAX_TENSORS);
    long long n4 = n / 4;
    int grid = ceil_div(n4, BLOCK);
    if (grid > MAX_GRID)
    {
        grid = MAX_GRID;
    }
    adamw_multi_kernel<<<grid, BLOCK, 0, s>>>(params, params_x, grads, m, v, n4, tensors, ntensors, st, beta1, beta2,
                                              eps);
    CUDA_CHECK_LAUNCH();
}
