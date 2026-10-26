#pragma once
// precision.h: floatX is the storage type of activations, activation
// gradients and the weight copies the GEMMs read. Picked at compile time:
//   nvcc -DFP16 ...   floatX = __half  (stage 4 mixed precision)
//   nvcc ...          floatX = float   (the correctness anchor: matches the dump
//                                       at stage 3 tolerances)
// Master weights, parameter gradients, AdamW state, the residual stream,
// LayerNorm statistics and every reduction stay fp32 in both builds.

#include <cuda_fp16.h>

#ifdef FP16
using floatX = __half;
#define FLOATX_NAME "fp16"
#else
using floatX = float;
#define FLOATX_NAME "fp32"
#endif

__host__ __device__ __forceinline__ float to_float(float x)
{
    return x;
}

__host__ __device__ __forceinline__ float to_float(__half x)
{
    return __half2float(x);
}

template <class T>
__host__ __device__ __forceinline__ T from_float(float x);

template <>
__host__ __device__ __forceinline__ float from_float<float>(float x)
{
    return x;
}

template <>
__host__ __device__ __forceinline__ __half from_float<__half>(float x)
{
    return __float2half(x);
}
