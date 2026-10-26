// gemm_mma.cu: tensor core GEMM with inline PTX mma.sync m16n8k8 + ldmatrix.
//
// Outside in:
//   block tile  128 x 128 of C, K in steps of 32, 128 threads (4 warps, 2 x 2)
//   warp tile   64 x 64 = 4 (m16) x 8 (n8) mma tiles, 128 fp32 accumulators per thread
//   mma tile    16 x 8 x 8 per instruction
// Every operand tile sits in shared memory the way it sits in global memory
// (so global -> shared copies are 128-bit, 8 halves at a time), and ldmatrix
// or ldmatrix.trans turns it into fragments:
//   A not transposed: rows of 32 k  -> ldmatrix        (the A fragment layout)
//   A transposed:     rows of 128 m -> ldmatrix.trans
//   B transposed:     rows of 32 k  -> ldmatrix        (the B "col" layout)
//   B not transposed: rows of 128 n -> ldmatrix.trans
// No cp.async on Turing, so the next K tile goes global -> registers while
// the tensor cores work on the current one, then registers -> the other
// shared buffer: one __syncthreads per K step.
//
// Split K: the weight gradients have few output tiles and a long K (384 x 384
// with K = 4,096 is 9 tiles for 46 SMs). Those (EP_ACCUM) split K over
// blockIdx.z and add their parts into C with fp32 atomics, so the last bits
// of a weight gradient can differ from run to run.
//
// Swizzle: a row holds 16-byte chunks. Reading the same chunk column from 8
// rows (one ldmatrix tile) would hit the same banks, so chunk c of row r is
// stored at c ^ f(r):
//   32-half rows (4 chunks, 64 B):   c ^ ((r >> 1) & 3)
//   128-half rows (16 chunks, 256 B): c ^ (r & 7)
// and any 8 consecutive rows then cover all 32 banks once.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "common.h"
#include "fast.h"
#include "gemm_common.cuh"
#include "mma.cuh"

using namespace std;

#ifdef FP16

namespace
{
const int BM = 128;
const int BN = 128;
const int BK = 32;
const int THREADS = 128;
const int TARGET_BLOCKS = 92;               // 2 per SM on the 2080's 46 SMs, the most that fit at once

// element offset of chunk `chunk` of row `row` in a tile with 32-half rows
__device__ __forceinline__ int kmaj(int row, int chunk)
{
    return row * BK + ((chunk ^ ((row >> 1) & 3)) << 3);
}

// element offset of chunk `chunk` of row `k` in a tile with 128-half rows
__device__ __forceinline__ int mmaj(int k, int chunk)
{
    return k * BM + ((chunk ^ (k & 7)) << 3);
}

// One operand tile = 4096 halves = 512 chunks of 8, 4 per thread.
//   KMAJ:  stored (rows x K), the tile is 128 rows (from r0) x 32 k (from k0)
//   !KMAJ: stored (K x rows), the tile is 32 k (from k0) x 128 rows (from r0)
// Chunks past the edge of the matrix are zeros, so they add nothing.
template <bool KMAJ>
__device__ __forceinline__ void load_tile(int4 (&reg)[4], const __half *src, int ld, int rows, int K, int r0, int k0)
{
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int c = threadIdx.x + i * THREADS;
        int r = KMAJ ? (c >> 2) : (c & 15) * 8;
        int k = KMAJ ? (c & 3) * 8 : (c >> 4);
        int gr = r0 + r;
        int gk = k0 + k;
        long long at = KMAJ ? static_cast<long long>(gr) * ld + gk : static_cast<long long>(gk) * ld + gr;
        const __half *p = src + at;
        reg[i] = load_or_zero(p, gr < rows && gk < K);
    }
}

template <bool KMAJ>
__device__ __forceinline__ void store_tile(__half *dst, const int4 (&reg)[4])
{
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int c = threadIdx.x + i * THREADS;
        int at = KMAJ ? kmaj(c >> 2, c & 3) : mmaj(c >> 4, c & 15);
        *reinterpret_cast<int4 *>(dst + at) = reg[i];
    }
}

template <bool TA, bool TB, int EP, class TC>
__global__ void __launch_bounds__(THREADS) gemm_mma_kernel(int M, int N, int K, int k_per, const __half *A, int lda,
                                                           const __half *B, int ldb, TC *C, int ldc, EpilogueArgs e)
{
    __shared__ __align__(16) __half sA[2][BM * BK];
    __shared__ __align__(16) __half sB[2][BN * BK];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int wm = warp >> 1;                     // warp tile row (0, 1)
    int wn = warp & 1;                      // warp tile column (0, 1)
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;
    int kb = blockIdx.z * k_per;            // this block's share of K: kb .. ke - 1
    int ke = min(K, kb + k_per);

    float acc[4][8][4];
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
    {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                acc[mt][nt][i] = 0.0f;
            }
        }
    }

    int4 ra[4];
    int4 rb[4];
    load_tile<!TA>(ra, A, lda, M, ke, m0, kb);
    load_tile<TB>(rb, B, ldb, N, ke, n0, kb);
    store_tile<!TA>(sA[0], ra);
    store_tile<TB>(sB[0], rb);
    __syncthreads();

    int ktiles = (ke - kb + BK - 1) / BK;
    for (int kt = 0; kt < ktiles; kt++)
    {
        bool more = kt + 1 < ktiles;
        if (more)                           // next tile into registers, in flight during the math
        {
            load_tile<!TA>(ra, A, lda, M, ke, m0, kb + (kt + 1) * BK);
            load_tile<TB>(rb, B, ldb, N, ke, n0, kb + (kt + 1) * BK);
        }
        const __half *a = sA[kt & 1];
        const __half *b = sB[kt & 1];

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16)
        {
            // A fragments for 4 m16 tiles x 2 k8 steps. Tile j of the x4 load:
            // rows (j % 2) * 8, k (j / 2) * 8 -> registers {a0, a1} of step 0, then of step 1.
            uint32_t af[4][4];
            #pragma unroll
            for (int mt = 0; mt < 4; mt++)
            {
                int mb = wm * 64 + mt * 16;
                if (!TA)
                {
                    int r = mb + (lane & 15);
                    int k = kk + (lane >> 4) * 8;
                    ldmatrix_x4(af[mt], a + kmaj(r, k >> 3));
                }
                else
                {
                    int k = kk + (lane >> 4) * 8 + (lane & 7);
                    int m = mb + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4_trans(af[mt], a + mmaj(k, m >> 3));
                }
            }
            // B fragments for 8 n8 tiles x 2 k8 steps, two n8 tiles per x4 load:
            // tile j: n (j / 2) * 8, k (j % 2) * 8.
            uint32_t bf[8][2];
            #pragma unroll
            for (int np = 0; np < 4; np++)
            {
                int nb = wn * 64 + np * 16;
                uint32_t r[4];
                if (TB)
                {
                    int n = nb + (lane >> 4) * 8 + (lane & 7);
                    int k = kk + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(r, b + kmaj(n, k >> 3));
                }
                else
                {
                    int k = kk + ((lane >> 3) & 1) * 8 + (lane & 7);
                    int n = nb + (lane >> 4) * 8;
                    ldmatrix_x4_trans(r, b + mmaj(k, n >> 3));
                }
                bf[2 * np][0] = r[0];
                bf[2 * np][1] = r[1];
                bf[2 * np + 1][0] = r[2];
                bf[2 * np + 1][1] = r[3];
            }
            #pragma unroll
            for (int mt = 0; mt < 4; mt++)
            {
                #pragma unroll
                for (int nt = 0; nt < 8; nt++)
                {
                    mma_16808(acc[mt][nt], &af[mt][0], bf[nt][0]);
                    mma_16808(acc[mt][nt], &af[mt][2], bf[nt][1]);
                }
            }
        }

        if (more)
        {
            store_tile<!TA>(sA[(kt + 1) & 1], ra);
            store_tile<TB>(sB[(kt + 1) & 1], rb);
        }
        __syncthreads();                    // the new tile is in, and nobody still reads the old one
    }

    // acc[mt][nt][2h + i] is C[g + 8h][2t + i] of mma tile (mt, nt)
    int g = lane >> 2;
    int t = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
    {
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            #pragma unroll
            for (int h = 0; h < 2; h++)
            {
                int row = m0 + wm * 64 + mt * 16 + g + 8 * h;
                int col = n0 + wn * 64 + nt * 8 + 2 * t;
                if (row < M && col < N)
                {
                    long long idx = static_cast<long long>(row) * ldc + col;
                    if (EP == EP_ACCUM && gridDim.z > 1)
                    {
                        atomicAdd(reinterpret_cast<float *>(C) + idx, acc[mt][nt][2 * h]);
                        atomicAdd(reinterpret_cast<float *>(C) + idx + 1, acc[mt][nt][2 * h + 1]);
                    }
                    else
                    {
                        epilogue_pair<EP, TC>(C, idx, col, acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1], e);
                    }
                }
            }
        }
    }
}

template <bool TA, bool TB, int EP, class TC>
void launch(int M, int N, int K, const __half *A, int lda, const __half *B, int ldb, TC *C, int ldc,
            EpilogueArgs e, cudaStream_t s)
{
    int tiles = ceil_div(N, BN) * ceil_div(M, BM);
    int splits = 1;
    if (EP == EP_ACCUM)                     // only C += acc can be added up in parts
    {
        int most = max(1, K / (4 * BK));    // every part keeps at least 4 K tiles
        splits = min(most, max(1, ceil_div(TARGET_BLOCKS, tiles)));
    }
    int k_per = ceil_div(ceil_div(K, splits), BK) * BK;
    splits = ceil_div(K, k_per);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM), splits);
    gemm_mma_kernel<TA, TB, EP, TC><<<grid, THREADS, 0, s>>>(M, N, K, k_per, A, lda, B, ldb, C, ldc, e);
    CUDA_CHECK_LAUNCH();
}
} // namespace

template <int EP, class TC>
void gemm_mma(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
              TC *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s)
{
    assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0 && lda % 8 == 0 && ldb % 8 == 0 && ldc % 2 == 0);
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

#else   // the fp32 build has no fp16 inputs for the tensor cores

template <int EP, class TC>
void gemm_mma(bool, bool, int, int, int, const floatX *, int, const floatX *, int, TC *, int, const float *,
              floatX *, const floatX *, cudaStream_t)
{
    fprintf(stderr, "error: the mma.sync GEMM needs the fp16 build (-DFP16)\n");
    exit(1);
}

#endif

template void gemm_mma<EP_STORE, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, floatX *,
                                         int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_mma<EP_BIAS, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, floatX *,
                                        int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_mma<EP_BIAS_GELU, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int,
                                             floatX *, int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_mma<EP_GELU_GRAD, floatX>(bool, bool, int, int, int, const floatX *, int, const floatX *, int,
                                             floatX *, int, const float *, floatX *, const floatX *, cudaStream_t);
template void gemm_mma<EP_ACCUM, float>(bool, bool, int, int, int, const floatX *, int, const floatX *, int, float *,
                                        int, const float *, floatX *, const floatX *, cudaStream_t);
