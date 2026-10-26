#pragma once
// reduce.cuh: warp and block reductions shared by the kernels.
//
// The warp versions use a butterfly (xor shuffle): at every step both lanes
// of a pair add the same two numbers, so all 32 lanes end with bit identical
// results. That matters for LayerNorm, where every lane normalizes its own
// columns with the mean it computed itself.

#include <cmath>

__device__ __forceinline__ float warp_sum(float v)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Every thread of the block must call these, and blockDim.x must be a
// multiple of 32 (at most 1024). Every thread gets the same result.
__device__ __forceinline__ float block_sum(float v)
{
    __shared__ float partial[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int warps = blockDim.x / 32;

    v = warp_sum(v);
    __syncthreads();                        // a previous call may still be reading partial
    if (lane == 0)
    {
        partial[warp] = v;
    }
    __syncthreads();
    float total = lane < warps ? partial[lane] : 0.0f;
    return warp_sum(total);                 // every warp adds the partials the same way
}

__device__ __forceinline__ float block_max(float v)
{
    __shared__ float partial[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int warps = blockDim.x / 32;

    v = warp_max(v);
    __syncthreads();
    if (lane == 0)
    {
        partial[warp] = v;
    }
    __syncthreads();
    float best = lane < warps ? partial[lane] : -INFINITY;
    return warp_max(best);
}
