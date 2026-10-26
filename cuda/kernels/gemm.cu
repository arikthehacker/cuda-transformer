// gemm.cu: picks the GEMM implementation for gemm_act / gemm_grad.
// fp16 build: mma.sync by default (set_gemm_impl to compare); fp32 build: SIMT only.

#include <cstdio>
#include <cstdlib>

#include "common.h"
#include "fast.h"
#include "gemm_common.cuh"

using namespace std;

#ifdef FP16
static GemmImpl g_impl = GemmImpl::Mma;
#else
static GemmImpl g_impl = GemmImpl::Simt;
#endif

void set_gemm_impl(GemmImpl impl)
{
#ifndef FP16
    if (impl != GemmImpl::Simt)
    {
        fprintf(stderr, "error: the fp32 build only has the SIMT GEMM\n");
        exit(1);
    }
#endif
    g_impl = impl;
}

GemmImpl gemm_impl()
{
    return g_impl;
}

const char *gemm_impl_name(GemmImpl impl)
{
    return impl == GemmImpl::Mma ? "mma.sync" : impl == GemmImpl::Wmma ? "wmma" : "simt";
}

template <int EP, class TC>
static void run(bool ta, bool tb, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb, TC *C,
                int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s)
{
    if (g_impl == GemmImpl::Mma)
    {
        gemm_mma<EP, TC>(ta, tb, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
    }
    else if (g_impl == GemmImpl::Wmma)
    {
        gemm_wmma<EP, TC>(ta, tb, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
    }
    else
    {
        gemm_simt<EP, TC>(ta, tb, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
    }
}

void gemm_act(int ep, bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B,
              int ldb, floatX *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s)
{
    switch (ep)
    {
    case EP_STORE:
        run<EP_STORE, floatX>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
        break;
    case EP_BIAS:
        run<EP_BIAS, floatX>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
        break;
    case EP_BIAS_GELU:
        run<EP_BIAS_GELU, floatX>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
        break;
    case EP_GELU_GRAD:
        run<EP_GELU_GRAD, floatX>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, bias, out2, aux, s);
        break;
    default:
        fprintf(stderr, "error: gemm_act with epilogue %d\n", ep);
        exit(1);
    }
}

void gemm_grad(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
               float *C, int ldc, cudaStream_t s)
{
    run<EP_ACCUM, float>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, nullptr, nullptr, nullptr, s);
}
