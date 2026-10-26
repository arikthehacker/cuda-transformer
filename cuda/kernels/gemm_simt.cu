// gemm_simt.cu: the GEMM on fp32 CUDA cores, with the same epilogues as the
// tensor core versions. The fp32 build uses it for every matmul (that build
// is the correctness anchor), and the tests use it as a second opinion.
// Same tiling as the stage 3 sgemm: 64 x 64 per block, 4 x 4 per thread, K by 16.

#include <cassert>

#include "common.h"
#include "fast.h"
#include "gemm_common.cuh"

namespace
{
const int TM = 64;
const int TN = 64;
const int TK = 16;
const int THREADS = 256;

template <bool TA, bool TB, int EP, class TC>
__global__ void gemm_simt_kernel(int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
                                 TC *C, int ldc, EpilogueArgs e)
{
    __shared__ float As[TK][TM + 1];
    __shared__ float Bs[TK][TN + 1];
    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;
    int i0 = blockIdx.y * TM;
    int j0 = blockIdx.x * TN;

    float acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++)
    {
        #pragma unroll
        for (int q = 0; q < 4; q++)
        {
            acc[r][q] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += TK)
    {
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int l = tid + r * THREADS;
            int ii = TA ? l % TM : l / TK;
            int kk = TA ? l / TM : l % TK;
            int gi = i0 + ii, gk = k0 + kk;
            float val = 0.0f;
            if (gi < M && gk < K)
            {
                val = to_float(TA ? A[static_cast<long long>(gk) * lda + gi] : A[static_cast<long long>(gi) * lda + gk]);
            }
            As[kk][ii] = val;
        }
        #pragma unroll
        for (int r = 0; r < 4; r++)
        {
            int l = tid + r * THREADS;
            int jj = TB ? l / TK : l % TN;
            int kk = TB ? l % TK : l / TN;
            int gj = j0 + jj, gk = k0 + kk;
            float val = 0.0f;
            if (gj < N && gk < K)
            {
                val = to_float(TB ? B[static_cast<long long>(gj) * ldb + gk] : B[static_cast<long long>(gk) * ldb + gj]);
            }
            Bs[kk][jj] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TK; kk++)
        {
            float av[4], bv[4];
            #pragma unroll
            for (int r = 0; r < 4; r++)
            {
                av[r] = As[kk][ty + 16 * r];
            }
            #pragma unroll
            for (int q = 0; q < 4; q++)
            {
                bv[q] = Bs[kk][tx + 16 * q];
            }
            #pragma unroll
            for (int r = 0; r < 4; r++)
            {
                #pragma unroll
                for (int q = 0; q < 4; q++)
                {
                    acc[r][q] += av[r] * bv[q];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; r++)
    {
        int i = i0 + ty + 16 * r;
        #pragma unroll
        for (int q = 0; q < 4; q++)
        {
            int j = j0 + tx + 16 * q;
            if (i < M && j < N)
            {
                epilogue_one<EP, TC>(C, static_cast<long long>(i) * ldc + j, j, acc[r][q], e);
            }
        }
    }
}

template <bool TA, bool TB, int EP, class TC>
void launch(int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb, TC *C, int ldc,
            EpilogueArgs e, cudaStream_t s)
{
    dim3 grid(ceil_div(N, TN), ceil_div(M, TM));
    gemm_simt_kernel<TA, TB, EP, TC><<<grid, THREADS, 0, s>>>(M, N, K, A, lda, B, ldb, C, ldc, e);
    CUDA_CHECK_LAUNCH();
}
} // namespace

template <int EP, class TC>
void gemm_simt(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
               TC *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s)
{
    assert(M > 0 && N > 0 && K > 0 && ceil_div(M, TM) <= 65535);
    EpilogueArgs e{bias, out2, aux};
    if (!transA && transB)
    {
        launch<false, true, EP, TC>(M, N, K, A, lda, B, ldb, C, ldc, e, s);
    }
    else if (!transA && !transB)
    {
        launch<false, false, EP, TC>(M, N, K, A, lda, B, ldb, C, ldc, e, s);
    }
    else if (transA && !transB)
    {
        launch<true, false, EP, TC>(M, N, K, A, lda, B, ldb, C, ldc, e, s);
    }
    else
    {
        launch<true, true, EP, TC>(M, N, K, A, lda, B, ldb, C, ldc, e, s);
    }
}

template void gemm_simt<EP_STORE, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, floatX *,
                                          int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_simt<EP_BIAS, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, floatX *,
                                         int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_simt<EP_BIAS_GELU, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int,
                                              floatX *, int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_simt<EP_GELU_GRAD, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int,
                                              floatX *, int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_simt<EP_ACCUM, float>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, float *,
                                         int, const float *, floatX *, const floatX *, cudaStream_t);
