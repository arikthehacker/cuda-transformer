// gemm_wmma.cu: tensor core GEMM with the WMMA API (the first step before PTX).
//
// 64 x 64 per block, 4 warps of 32 x 32 (2 x 2 fragments of 16 x 16), K by 32.
// Transposes are handled by the fragment layout: a tile stored the other way
// around in shared memory is simply loaded as col_major instead of row_major.
// The accumulators go through shared memory for the epilogue, because WMMA
// does not say which thread holds which element.

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "common.h"
#include "fast.h"
#include "gemm_common.cuh"
#include "mma.cuh"

#ifdef FP16
#include <mma.h>
#include <type_traits>

using namespace nvcuda;

namespace
{
const int BM = 64;
const int BN = 64;
const int BK = 32;
const int THREADS = 128;
const int PADK = BK + 8;        // row lengths with 8 halves of padding (keeps ldm a multiple of 8
const int PADM = BM + 8;        // and spreads the rows over the banks)
const int PADC = BN + 4;
const int TILE = BM * PADK > BK * PADM ? BM * PADK : BK * PADM;   // either layout fits

// Copy one operand tile to shared memory, 8 halves at a time.
//   KMAJ:  stored (rows x K), tile 64 rows x 32 k, shared [row][k] with PADK
//   !KMAJ: stored (K x rows), tile 32 k x 64 rows, shared [k][row] with PADM
template <bool KMAJ>
__device__ __forceinline__ void copy_tile(__half *dst, const __half *src, int ld, int rows, int K, int r0, int k0)
{
    #pragma unroll
    for (int i = 0; i < 2; i++)                 // 64 x 32 = 2048 halves = 256 chunks, 2 per thread
    {
        int c = threadIdx.x + i * THREADS;
        int r = KMAJ ? c >> 2 : (c & 7) * 8;
        int k = KMAJ ? (c & 3) * 8 : c >> 3;
        int gr = r0 + r, gk = k0 + k;
        long long at = KMAJ ? static_cast<long long>(gr) * ld + gk : static_cast<long long>(gk) * ld + gr;
        const __half *p = src + at;
        __half *q = KMAJ ? dst + r * PADK + k : dst + k * PADM + r;
        *reinterpret_cast<int4 *>(q) = load_or_zero(p, gr < rows && gk < K);
    }
}

template <bool TA, bool TB, int EP, class TC>
__global__ void gemm_wmma_kernel(int M, int N, int K, const __half *A, int lda, const __half *B, int ldb,
                                 TC *C, int ldc, EpilogueArgs e)
{
    __shared__ __align__(32) __half sA[TILE];
    __shared__ __align__(32) __half sB[TILE];
    __shared__ __align__(32) float sC[BM * PADC];

    using LayoutA = typename std::conditional<TA, wmma::col_major, wmma::row_major>::type;
    using LayoutB = typename std::conditional<TB, wmma::col_major, wmma::row_major>::type;
    int warp = threadIdx.x / 32;
    int wm = warp >> 1, wn = warp & 1;
    int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        copy_tile<!TA>(sA, A, lda, M, K, m0, k0);
        copy_tile<TB>(sB, B, ldb, N, K, n0, k0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, LayoutA> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, LayoutB> fb[2];
            #pragma unroll
            for (int i = 0; i < 2; i++)
            {
                int m = wm * 32 + i * 16;
                // row_major: element (m, k) at [m][k]; col_major: element (m, k) at [k][m]
                const __half *p = TA ? sA + kk * PADM + m : sA + m * PADK + kk;
                wmma::load_matrix_sync(fa[i], p, TA ? PADM : PADK);
            }
            #pragma unroll
            for (int j = 0; j < 2; j++)
            {
                int n = wn * 32 + j * 16;
                // col_major: element (k, n) at [n][k]; row_major: element (k, n) at [k][n]
                const __half *p = TB ? sB + n * PADK + kk : sB + kk * PADM + n;
                wmma::load_matrix_sync(fb[j], p, TB ? PADK : PADM);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++)
            {
                #pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        #pragma unroll
        for (int j = 0; j < 2; j++)
        {
            wmma::store_matrix_sync(sC + (wm * 32 + i * 16) * PADC + wn * 32 + j * 16, acc[i][j], PADC,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * BN; idx += THREADS)
    {
        int r = idx / BN, c = idx % BN;
        int row = m0 + r, col = n0 + c;
        if (row < M && col < N)
        {
            epilogue_one<EP, TC>(C, static_cast<long long>(row) * ldc + col, col, sC[r * PADC + c], e);
        }
    }
}

template <bool TA, bool TB, int EP, class TC>
void launch(int M, int N, int K, const __half *A, int lda, const __half *B, int ldb, TC *C, int ldc,
            EpilogueArgs e, cudaStream_t s)
{
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm_wmma_kernel<TA, TB, EP, TC><<<grid, THREADS, 0, s>>>(M, N, K, A, lda, B, ldb, C, ldc, e);
    CUDA_CHECK_LAUNCH();
}
} // namespace

template <int EP, class TC>
void gemm_wmma(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
               TC *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s)
{
    assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0 && lda % 8 == 0 && ldb % 8 == 0);
    assert(ceil_div(M, BM) <= 65535);
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

#else

template <int EP, class TC>
void gemm_wmma(bool, bool, int, int, int, const floatX *, int, const floatX *, int, TC *, int, const float *,
               floatX *, const floatX *, cudaStream_t)
{
    fprintf(stderr, "error: the WMMA GEMM needs the fp16 build (-DFP16)\n");
    exit(1);
}

#endif

template void gemm_wmma<EP_STORE, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, floatX *,
                                          int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_wmma<EP_BIAS, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, floatX *,
                                         int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_wmma<EP_BIAS_GELU, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int,
                                              floatX *, int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_wmma<EP_GELU_GRAD, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int,
                                              floatX *, int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_wmma<EP_ACCUM, float>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, float *,
                                         int, const float *, floatX *, const floatX *, cudaStream_t);
