#pragma once
// gemm_common.cuh: what every GEMM implementation shares, so the math after
// the multiply (bias, GELU, accumulate) is one piece of code for all three.
//
// C (M x N) = op(A) (M x K) @ op(B) (K x N), row-major, then the epilogue:
//   EP_STORE       C = acc                                 C floatX
//   EP_BIAS        C = acc + bias[col]                     C floatX
//   EP_BIAS_GELU   C = acc + bias[col] (the pre-activation, kept for backward)
//                  C2 = gelu(C)                            both floatX
//   EP_GELU_GRAD   C = acc * gelu'(aux[row, col])          aux = the saved fc
//   EP_ACCUM       C += acc                                C fp32 (parameter gradients)
// GELU is computed from the value already rounded to floatX, the same as
// PyTorch autocast (fc is stored in fp16, then gelu reads it), so the
// backward pass, which recomputes gelu from the stored fc, sees the same
// numbers the forward pass used.

#include <cmath>

#include "precision.h"

enum
{
    EP_STORE = 0,
    EP_BIAS = 1,
    EP_BIAS_GELU = 2,
    EP_GELU_GRAD = 3,
    EP_ACCUM = 4,
};

// 0.5 x (1 + tanh(s (x + 0.044715 x^3))), s = sqrt(2/pi)
__device__ __forceinline__ float gelu_f(float x)
{
    const float s = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(s * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float gelu_grad_f(float x)
{
    const float s = 0.7978845608028654f;
    float th = tanhf(s * (x + 0.044715f * x * x * x));
    return 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * s * (1.0f + 3.0f * 0.044715f * x * x);
}

struct EpilogueArgs
{
    const float *bias;      // EP_BIAS, EP_BIAS_GELU
    floatX *out2;           // EP_BIAS_GELU: gelu output, same shape and ldc as C
    const floatX *aux;      // EP_GELU_GRAD: fc, same shape and ldc as C
};

template <int EP, class TC>
__device__ __forceinline__ void epilogue_one(TC *C, long long idx, int col, float v, const EpilogueArgs &e)
{
    if (EP == EP_ACCUM)
    {
        C[idx] = from_float<TC>(to_float(C[idx]) + v);
    }
    else if (EP == EP_STORE)
    {
        C[idx] = from_float<TC>(v);
    }
    else if (EP == EP_BIAS)
    {
        C[idx] = from_float<TC>(v + e.bias[col]);
    }
    else if (EP == EP_BIAS_GELU)
    {
        TC pre = from_float<TC>(v + e.bias[col]);
        C[idx] = pre;
        e.out2[idx] = from_float<floatX>(gelu_f(to_float(pre)));
    }
    else
    {
        C[idx] = from_float<TC>(v * gelu_grad_f(to_float(e.aux[idx])));
    }
}

// Two neighboring outputs (idx even, col even) with one 32-bit (fp16) or
// 64-bit (fp32) access instead of two, for the GEMMs whose threads own
// column pairs (mma.sync: d[0], d[1] are C[g][2t], C[g][2t + 1]).
__device__ __forceinline__ void store2(float *p, float a, float b)
{
    *reinterpret_cast<float2 *>(p) = make_float2(a, b);
}

__device__ __forceinline__ void store2(__half *p, float a, float b)
{
    *reinterpret_cast<__half2 *>(p) = __floats2half2_rn(a, b);
}

__device__ __forceinline__ void load2(const float *p, float &a, float &b)
{
    float2 v = *reinterpret_cast<const float2 *>(p);
    a = v.x;
    b = v.y;
}

__device__ __forceinline__ void load2(const __half *p, float &a, float &b)
{
    __half2 v = *reinterpret_cast<const __half2 *>(p);
    a = __low2float(v);
    b = __high2float(v);
}

// the same math as epilogue_one, for C[idx] and C[idx + 1]
template <int EP, class TC>
__device__ __forceinline__ void epilogue_pair(TC *C, long long idx, int col, float v0, float v1, const EpilogueArgs &e)
{
    if (EP == EP_ACCUM)
    {
        float c0, c1;
        load2(C + idx, c0, c1);
        store2(C + idx, c0 + v0, c1 + v1);
    }
    else if (EP == EP_STORE)
    {
        store2(C + idx, v0, v1);
    }
    else if (EP == EP_BIAS)
    {
        store2(C + idx, v0 + e.bias[col], v1 + e.bias[col + 1]);
    }
    else if (EP == EP_BIAS_GELU)
    {
        float p0 = to_float(from_float<TC>(v0 + e.bias[col]));    // rounded first, like epilogue_one
        float p1 = to_float(from_float<TC>(v1 + e.bias[col + 1]));
        store2(C + idx, p0, p1);
        store2(e.out2 + idx, gelu_f(p0), gelu_f(p1));
    }
    else
    {
        float x0, x1;
        load2(e.aux + idx, x0, x1);
        store2(C + idx, v0 * gelu_grad_f(x0), v1 * gelu_grad_f(x1));
    }
}
