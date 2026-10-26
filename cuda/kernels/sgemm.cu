// sgemm.cu: ONE batched matrix multiply used by every matmul in the model,
// plus the linear layer helpers built on it.
//
// For each batch item z:  C = alpha * op(A) @ op(B) + beta * C, all row-major.
//   op(A)[i][k] = transA ? A[k * lda + i] : A[i * lda + k]
//   op(B)[k][j] = transB ? B[j * ldb + k] : B[k * ldb + j]
//   C[i][j]     = C[i * ldc + j]
// Both kernels add the products in the same order (k = 0, 1, 2, ...), so the
// tiled one can be checked against the naive one tightly.

#include <cassert>

#include "common.h"
#include "kernels.h"

using namespace std;

// ---------------------------------------------------------------- naive

static const int NAIVE = 16;                // 16 x 16 threads, one output each

__global__ void sgemm_naive_kernel(bool transA, bool transB, int M, int N, int K, float alpha,
                                   const float *A, int lda, long long strideA,
                                   const float *B, int ldb, long long strideB,
                                   float beta, float *C, int ldc, long long strideC)
{
    int j = blockIdx.x * NAIVE + threadIdx.x;
    int i = blockIdx.y * NAIVE + threadIdx.y;
    if (i < M && j < N)
    {
        const float *a = A + blockIdx.z * strideA;
        const float *b = B + blockIdx.z * strideB;
        float *c = C + blockIdx.z * strideC;
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
        {
            float av = transA ? a[static_cast<long long>(k) * lda + i] : a[static_cast<long long>(i) * lda + k];
            float bv = transB ? b[static_cast<long long>(j) * ldb + k] : b[static_cast<long long>(k) * ldb + j];
            acc += av * bv;
        }
        long long ci = static_cast<long long>(i) * ldc + j;
        float out = alpha * acc;
        if (beta != 0.0f)                   // beta 0 never reads C, it may be uninitialized
        {
            out += beta * c[ci];
        }
        c[ci] = out;
    }
}

// ---------------------------------------------------------------- tiled
// A block computes a 64 x 64 tile of C with 256 threads, 4 x 4 outputs each.
// Thread (tx, ty) = (tid % 16, tid / 16) owns rows ty + 16r and columns
// tx + 16c (r, c = 0..3), so a warp writes 16 neighboring columns at a time.
// K goes in steps of 16: load a 64 x 16 slice of op(A) and a 16 x 64 slice of
// op(B) into shared memory (zeros past the edges), sync, multiply, sync.

static const int TM = 64;
static const int TN = 64;
static const int TK = 16;
static const int TILED_THREADS = 256;

__global__ void sgemm_tiled_kernel(bool transA, bool transB, int M, int N, int K, float alpha,
                                   const float *A, int lda, long long strideA,
                                   const float *B, int ldb, long long strideB,
                                   float beta, float *C, int ldc, long long strideC)
{
    // As[kk][ii] = op(A)[i0 + ii][k0 + kk], Bs[kk][jj] = op(B)[k0 + kk][j0 + jj].
    // The +1 keeps the column-wise stores from all landing in one bank.
    __shared__ float As[TK][TM + 1];
    __shared__ float Bs[TK][TN + 1];

    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;
    int i0 = blockIdx.y * TM;
    int j0 = blockIdx.x * TN;
    const float *a = A + blockIdx.z * strideA;
    const float *b = B + blockIdx.z * strideB;
    float *c = C + blockIdx.z * strideC;

    float acc[4][4];
    for (int r = 0; r < 4; r++)
    {
        for (int q = 0; q < 4; q++)
        {
            acc[r][q] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += TK)
    {
        // 64 x 16 = 1024 values of op(A), 4 per thread. Threads walk the
        // dimension that is contiguous in memory, so the global reads coalesce.
        for (int r = 0; r < 4; r++)
        {
            int l = tid + r * TILED_THREADS;
            int ii = transA ? l % TM : l / TK;
            int kk = transA ? l / TM : l % TK;
            int gi = i0 + ii;
            int gk = k0 + kk;
            float val = 0.0f;
            if (gi < M && gk < K)
            {
                val = transA ? a[static_cast<long long>(gk) * lda + gi] : a[static_cast<long long>(gi) * lda + gk];
            }
            As[kk][ii] = val;
        }
        // 16 x 64 values of op(B), 4 per thread
        for (int r = 0; r < 4; r++)
        {
            int l = tid + r * TILED_THREADS;
            int jj = transB ? l / TK : l % TN;
            int kk = transB ? l % TK : l / TN;
            int gj = j0 + jj;
            int gk = k0 + kk;
            float val = 0.0f;
            if (gj < N && gk < K)
            {
                val = transB ? b[static_cast<long long>(gj) * ldb + gk] : b[static_cast<long long>(gk) * ldb + gj];
            }
            Bs[kk][jj] = val;
        }
        __syncthreads();                    // the tile is complete before anyone reads it

        for (int kk = 0; kk < TK; kk++)
        {
            float av[4];
            float bv[4];
            for (int r = 0; r < 4; r++)
            {
                av[r] = As[kk][ty + 16 * r];
            }
            for (int q = 0; q < 4; q++)
            {
                bv[q] = Bs[kk][tx + 16 * q];
            }
            for (int r = 0; r < 4; r++)
            {
                for (int q = 0; q < 4; q++)
                {
                    acc[r][q] += av[r] * bv[q];
                }
            }
        }
        __syncthreads();                    // everyone is done reading before the next load overwrites it
    }

    for (int r = 0; r < 4; r++)
    {
        int i = i0 + ty + 16 * r;
        for (int q = 0; q < 4; q++)
        {
            int j = j0 + tx + 16 * q;
            if (i < M && j < N)
            {
                long long ci = static_cast<long long>(i) * ldc + j;
                float out = alpha * acc[r][q];
                if (beta != 0.0f)
                {
                    out += beta * c[ci];
                }
                c[ci] = out;
            }
        }
    }
}

// ---------------------------------------------------------------- launchers

static void check_sgemm(bool transA, bool transB, int M, int N, int K, int lda, int ldb, int ldc, int batch)
{
    assert(M > 0 && N > 0 && K > 0);
    assert(batch >= 1 && batch <= 65535);           // gridDim.z limit
    assert(lda >= (transA ? M : K));
    assert(ldb >= (transB ? K : N));
    assert(ldc >= N);
    (void)transA; (void)transB; (void)M; (void)N; (void)K; (void)lda; (void)ldb; (void)ldc; (void)batch;
}

void sgemm(bool transA, bool transB, int M, int N, int K, float alpha,
           const float *A, int lda, long long strideA,
           const float *B, int ldb, long long strideB,
           float beta, float *C, int ldc, long long strideC, int batch)
{
    check_sgemm(transA, transB, M, N, K, lda, ldb, ldc, batch);
    assert(ceil_div(M, TM) <= 65535);               // gridDim.y limit
    dim3 grid(ceil_div(N, TN), ceil_div(M, TM), batch);
    sgemm_tiled_kernel<<<grid, TILED_THREADS>>>(transA, transB, M, N, K, alpha, A, lda, strideA,
                                                B, ldb, strideB, beta, C, ldc, strideC);
    CUDA_CHECK_LAUNCH();
}

void sgemm_naive(bool transA, bool transB, int M, int N, int K, float alpha,
                 const float *A, int lda, long long strideA,
                 const float *B, int ldb, long long strideB,
                 float beta, float *C, int ldc, long long strideC, int batch)
{
    check_sgemm(transA, transB, M, N, K, lda, ldb, ldc, batch);
    assert(ceil_div(M, NAIVE) <= 65535);
    dim3 grid(ceil_div(N, NAIVE), ceil_div(M, NAIVE), batch);
    dim3 block(NAIVE, NAIVE);
    sgemm_naive_kernel<<<grid, block>>>(transA, transB, M, N, K, alpha, A, lda, strideA,
                                        B, ldb, strideB, beta, C, ldc, strideC);
    CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- linear layers

// out (N,OC) = inp (N,IC) @ W^T: transB, M N, N OC, K IC, beta 0. Then the bias.
void linear_forward(float *out, const float *inp, const float *w, const float *b, int N, int IC, int OC)
{
    sgemm(false, true, N, OC, IC, 1.0f, inp, IC, 0, w, IC, 0, 0.0f, out, OC, 0, 1);
    if (b != nullptr)
    {
        add_bias(out, b, N, OC);
    }
}

void linear_backward(float *dinp, float *dw, float *db, const float *dout, const float *inp, const float *w,
                     int N, int IC, int OC)
{
    // dinp (N,IC) += dout (N,OC) @ W (OC,IC):  M N, N IC, K OC, beta 1
    sgemm(false, false, N, IC, OC, 1.0f, dout, OC, 0, w, IC, 0, 1.0f, dinp, IC, 0, 1);
    // dW (OC,IC) += dout^T @ inp:  transA, M OC, N IC, K N, beta 1
    sgemm(true, false, OC, IC, N, 1.0f, dout, OC, 0, inp, IC, 0, 1.0f, dw, IC, 0, 1);
    if (db != nullptr)
    {
        bias_backward(db, dout, N, OC);
    }
}
