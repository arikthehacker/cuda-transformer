// flash_attn.cu: fused causal attention for Turing (FlashAttention-2 style,
// written for mma.sync m16n8k8, which FlashAttention 2 itself does not support).
//
// Never writes the (B, H, T, T) scores or probs. Reads q, k, v straight out
// of qkv (B, T, 3C) and writes atty (B, T, C), so the permute kernels go too.
//
// Forward, one block per (b, h, 64 query rows), 4 warps x 16 rows:
//   for each K/V tile of 64 keys, only up to the diagonal:
//     S = Q K^T / sqrt(D)       mma, fp32 accumulate, -inf above the diagonal
//     m_new = max(m, rowmax S); P = exp(S - m_new); l = l e^(m - m_new) + rowsum P
//     O = O e^(m - m_new) + P V
//   O /= l, and L = m + log(l) per row for the backward pass.
//   The S accumulators of one 8-key tile are, packed to halves, exactly the A
//   fragment of the P V product for those 8 keys (mma.cuh), so P never
//   leaves registers.
//
// Backward, one block per (b, h, 64 keys), 4 warps x 16 keys, recomputing P:
//   D_i = rowsum(dO_i * O_i)                      (a small kernel first)
//   for each query tile from the diagonal down:
//     P^T  = exp(K Q^T / sqrt(D) - L)            per warp: its 16 keys x 64 queries
//     dV  += P^T dO
//     dP^T = V dO^T
//     dS^T = P^T * (dP^T - D)
//     dK  += dS^T Q / sqrt(D)
//     dQ  += dS K / sqrt(D)   -> needs dS with queries as rows, so dS^T goes
//                                through shared memory once; added into an
//                                fp32 buffer with atomicAdd (several key blocks
//                                add into the same query rows), converted at the end.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "common.h"
#include "fast.h"
#include "mma.cuh"

#ifdef FP16

namespace
{
const int TILE = 64;            // queries or keys per tile
const int THREADS = 128;
const unsigned FULL = 0xffffffffu;

// Element offset of 16-byte chunk `chunk` of row `row` in a tile with W-half
// rows, swizzled so that 8 rows of one chunk column hit all 32 banks.
template <int W>
__device__ __forceinline__ int swz(int row, int chunk)
{
    const int cpr = W / 8;      // chunks per row
    int x = cpr >= 8 ? (row & 7) : cpr == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1);
    return row * W + ((chunk ^ x) << 3);
}

// rows x D halves (row stride ld in global memory) -> shared, swizzled
template <int D>
__device__ __forceinline__ void load_rows(__half *dst, const __half *src, long long ld)
{
    const int cpr = D / 8;
    for (int c = threadIdx.x; c < TILE * cpr; c += THREADS)
    {
        int r = c / cpr, ch = c % cpr;
        *reinterpret_cast<int4 *>(dst + swz<D>(r, ch)) = *reinterpret_cast<const int4 *>(src + r * ld + ch * 8);
    }
}

// A fragments of rows r0 .. r0+15, all D columns, from a swizzled [row][D] tile
template <int D>
__device__ __forceinline__ void load_a_rows(uint32_t (&a)[D / 16][4], const __half *s, int r0, int lane)
{
    #pragma unroll
    for (int kc = 0; kc < D / 16; kc++)
    {
        int r = r0 + (lane & 15);
        int d = kc * 16 + (lane >> 4) * 8;
        ldmatrix_x4(a[kc], s + swz<D>(r, d >> 3));
    }
}

// acc[8][4] (16 rows x 64 columns) += A (16 x D, fragments) @ X^T, X a swizzled
// [64][D] tile: column n of the result is row n of X (B "col" layout, plain ldmatrix)
template <int D>
__device__ __forceinline__ void mma_abt(float (&acc)[8][4], const uint32_t (&a)[D / 16][4], const __half *x, int lane)
{
    #pragma unroll
    for (int kc = 0; kc < D / 16; kc++)
    {
        #pragma unroll
        for (int np = 0; np < 4; np++)
        {
            uint32_t r[4];
            int n = np * 16 + (lane >> 4) * 8 + (lane & 7);
            int d = kc * 16 + ((lane >> 3) & 1) * 8;
            ldmatrix_x4(r, x + swz<D>(n, d >> 3));
            mma_16808(acc[2 * np], &a[kc][0], r[0]);
            mma_16808(acc[2 * np], &a[kc][2], r[1]);
            mma_16808(acc[2 * np + 1], &a[kc][0], r[2]);
            mma_16808(acc[2 * np + 1], &a[kc][2], r[3]);
        }
    }
}

// acc[D/8][4] (16 rows x D) += P (16 x 64, as 8 k-step fragments) @ X, X a
// swizzled [64][D] tile used as B with rows = k (ldmatrix.trans)
template <int D>
__device__ __forceinline__ void mma_px(float (&acc)[D / 8][4], const uint32_t (&p)[8][2], const __half *x, int lane)
{
    #pragma unroll
    for (int kp = 0; kp < 4; kp++)
    {
        #pragma unroll
        for (int dp = 0; dp < D / 16; dp++)
        {
            uint32_t r[4];
            int k = kp * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
            int d = dp * 16 + (lane >> 4) * 8;
            ldmatrix_x4_trans(r, x + swz<D>(k, d >> 3));
            mma_16808(acc[2 * dp], p[2 * kp], r[0]);
            mma_16808(acc[2 * dp], p[2 * kp + 1], r[1]);
            mma_16808(acc[2 * dp + 1], p[2 * kp], r[2]);
            mma_16808(acc[2 * dp + 1], p[2 * kp + 1], r[3]);
        }
    }
}

// fp32 accumulators of a 16 x 64 tile -> the A fragments of its 8 k-steps
__device__ __forceinline__ void to_a_frags(uint32_t (&p)[8][2], const float (&v)[8][4])
{
    #pragma unroll
    for (int nt = 0; nt < 8; nt++)
    {
        p[nt][0] = pack_half2(v[nt][0], v[nt][1]);
        p[nt][1] = pack_half2(v[nt][2], v[nt][3]);
    }
}

template <int N>
__device__ __forceinline__ void zero(float (&acc)[N][4])
{
    #pragma unroll
    for (int i = 0; i < N; i++)
    {
        #pragma unroll
        for (int e = 0; e < 4; e++)
        {
            acc[i][e] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------- forward

template <int D>
__global__ void __launch_bounds__(THREADS) flash_fwd_kernel(__half *atty, float *L, const __half *qkv, int T, int H)
{
    __shared__ __align__(16) __half sQ[TILE * D];
    __shared__ __align__(16) __half sK[TILE * D];
    __shared__ __align__(16) __half sV[TILE * D];

    const int C = H * D;
    const long long ld = 3LL * C;
    int qt = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H, h = bh % H;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane >> 2, t = lane & 3;
    const __half *base = qkv + static_cast<long long>(b) * T * ld;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    load_rows<D>(sQ, base + static_cast<long long>(qt) * TILE * ld + h * D, ld);
    __syncthreads();
    uint32_t aq[D / 16][4];
    load_a_rows<D>(aq, sQ, warp * 16, lane);

    float o[D / 8][4];
    zero(o);
    float m[2] = {-INFINITY, -INFINITY};    // rows g and g + 8 of this warp's 16
    float l[2] = {0.0f, 0.0f};              // this lane's share of the row sums
    int q_row = qt * TILE + warp * 16 + g;

    for (int kt = 0; kt <= qt; kt++)
    {
        __syncthreads();                    // everyone is done with the previous K/V tile
        load_rows<D>(sK, base + static_cast<long long>(kt) * TILE * ld + C + h * D, ld);
        load_rows<D>(sV, base + static_cast<long long>(kt) * TILE * ld + 2 * C + h * D, ld);
        __syncthreads();

        float s[8][4];
        zero(s);
        mma_abt<D>(s, aq, sK, lane);

        float mx[2] = {m[0], m[1]};
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                int key = kt * TILE + nt * 8 + 2 * t + (e & 1);
                int q = q_row + 8 * (e >> 1);
                float v = s[nt][e] * scale;
                if (kt == qt && key > q)
                {
                    v = -INFINITY;
                }
                s[nt][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++)         // the 4 lanes of a group share a row
        {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(FULL, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(FULL, mx[i], 2));
        }
        float alpha[2] = {expf(m[0] - mx[0]), expf(m[1] - mx[1])};  // 0 on the first tile (m = -inf)
        m[0] = mx[0];
        m[1] = mx[1];

        float rowsum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                float p = expf(s[nt][e] - m[e >> 1]);   // masked: exp(-inf) = 0
                s[nt][e] = p;
                rowsum[e >> 1] += p;
            }
        }
        l[0] = l[0] * alpha[0] + rowsum[0];
        l[1] = l[1] * alpha[1] + rowsum[1];
        #pragma unroll
        for (int dt = 0; dt < D / 8; dt++)
        {
            o[dt][0] *= alpha[0];
            o[dt][1] *= alpha[0];
            o[dt][2] *= alpha[1];
            o[dt][3] *= alpha[1];
        }
        uint32_t ap[8][2];
        to_a_frags(ap, s);
        mma_px<D>(o, ap, sV, lane);
    }

    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        l[i] += __shfl_xor_sync(FULL, l[i], 1);
        l[i] += __shfl_xor_sync(FULL, l[i], 2);
    }
    #pragma unroll
    for (int hh = 0; hh < 2; hh++)
    {
        int q = q_row + 8 * hh;
        float inv = 1.0f / l[hh];
        __half *out = atty + (static_cast<long long>(b) * T + q) * C + h * D;
        #pragma unroll
        for (int dt = 0; dt < D / 8; dt++)
        {
            *reinterpret_cast<__half2 *>(out + dt * 8 + 2 * t) =
                __floats2half2_rn(o[dt][2 * hh] * inv, o[dt][2 * hh + 1] * inv);
        }
        if (t == 0)
        {
            L[static_cast<long long>(bh) * T + q] = m[hh] + logf(l[hh]);
        }
    }
}

// ---------------------------------------------------------------- backward

// D[b, h, i] = sum over d of dO[b, i, h*D + d] * O[b, i, h*D + d], one warp per row
__global__ void flash_bwd_prep_kernel(float *Dq, const __half *datty, const __half *atty, int T, int H, int D,
                                      long long rows)
{
    long long row = static_cast<long long>(blockIdx.x) * (THREADS / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row < rows)
    {
        int i = static_cast<int>(row % T);
        long long bh = row / T;
        int h = static_cast<int>(bh % H);
        long long b = bh / H;
        long long at = (b * T + i) * static_cast<long long>(H) * D + static_cast<long long>(h) * D;
        float sum = 0.0f;
        for (int d = lane; d < D; d += 32)
        {
            sum += __half2float(datty[at + d]) * __half2float(atty[at + d]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off /= 2)
        {
            sum += __shfl_xor_sync(FULL, sum, off);
        }
        if (lane == 0)
        {
            Dq[row] = sum;
        }
    }
}

template <int D>
__global__ void __launch_bounds__(THREADS) flash_bwd_kernel(__half *dqkv, float *dq_acc, const __half *qkv,
                                                            const __half *datty, const float *L, const float *Dq,
                                                            int T, int H)
{
    __shared__ __align__(16) __half sK[TILE * D];
    __shared__ __align__(16) __half sV[TILE * D];
    __shared__ __align__(16) __half sQ[TILE * D];
    __shared__ __align__(16) __half sdO[TILE * D];
    __shared__ __align__(16) __half sdS[TILE * TILE];     // dS^T: [key][query]
    __shared__ float sL[TILE];
    __shared__ float sD[TILE];

    const int C = H * D;
    const long long ld = 3LL * C;
    int kvt = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H, h = bh % H;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane >> 2, t = lane & 3;
    const __half *base = qkv + static_cast<long long>(b) * T * ld;
    const __half *dobase = datty + static_cast<long long>(b) * T * C;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    int ntiles = T / TILE;

    load_rows<D>(sK, base + static_cast<long long>(kvt) * TILE * ld + C + h * D, ld);
    load_rows<D>(sV, base + static_cast<long long>(kvt) * TILE * ld + 2 * C + h * D, ld);
    __syncthreads();
    uint32_t ak[D / 16][4];
    load_a_rows<D>(ak, sK, warp * 16, lane);

    float dk[D / 8][4];
    float dv[D / 8][4];
    zero(dk);
    zero(dv);
    int key0 = warp * 16 + g;               // this lane's key rows: key0 and key0 + 8 (in the tile)

    for (int qt = kvt; qt < ntiles; qt++)
    {
        __syncthreads();                    // the previous query tile is no longer read
        load_rows<D>(sQ, base + static_cast<long long>(qt) * TILE * ld + h * D, ld);
        load_rows<D>(sdO, dobase + static_cast<long long>(qt) * TILE * C + h * D, C);
        if (threadIdx.x < TILE)
        {
            long long at = static_cast<long long>(bh) * T + qt * TILE + threadIdx.x;
            sL[threadIdx.x] = L[at];
            sD[threadIdx.x] = Dq[at];
        }
        __syncthreads();

        // P^T (this warp's 16 keys x 64 queries)
        float p[8][4];
        zero(p);
        mma_abt<D>(p, ak, sQ, lane);
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                int ql = nt * 8 + 2 * t + (e & 1);
                int key = kvt * TILE + key0 + 8 * (e >> 1);
                int q = qt * TILE + ql;
                p[nt][e] = (qt == kvt && q < key) ? 0.0f : expf(p[nt][e] * scale - sL[ql]);
            }
        }
        uint32_t ap[8][2];
        to_a_frags(ap, p);
        mma_px<D>(dv, ap, sdO, lane);       // dV += P^T dO

        float ds[8][4];                     // dP^T = V dO^T, then dS^T = P^T (dP^T - D)
        zero(ds);
        {
            uint32_t av[D / 16][4];         // reloaded from sV each tile: holding it live spills at D = 64
            load_a_rows<D>(av, sV, warp * 16, lane);
            mma_abt<D>(ds, av, sdO, lane);
        }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                int ql = nt * 8 + 2 * t + (e & 1);
                ds[nt][e] = p[nt][e] * (ds[nt][e] - sD[ql]);
            }
        }
        uint32_t ads[8][2];
        to_a_frags(ads, ds);
        mma_px<D>(dk, ads, sQ, lane);       // dK += dS^T Q   (scaled at the end)

        // dS^T -> shared, so dQ can read it with queries as rows
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            *reinterpret_cast<uint32_t *>(sdS + swz<TILE>(key0, nt) + 2 * t) = ads[nt][0];
            *reinterpret_cast<uint32_t *>(sdS + swz<TILE>(key0 + 8, nt) + 2 * t) = ads[nt][1];
        }
        __syncthreads();

        // dQ (this warp's 16 queries x D) = dS (16 x 64 keys) @ K (64 x D)
        float dq[D / 8][4];
        zero(dq);
        #pragma unroll
        for (int kp = 0; kp < 4; kp++)
        {
            uint32_t a[4];                  // dS[q][key] from dS^T stored [key][q]: ldmatrix.trans
            int k = kp * 16 + (lane >> 4) * 8 + (lane & 7);
            int mq = warp * 16 + ((lane >> 3) & 1) * 8;
            ldmatrix_x4_trans(a, sdS + swz<TILE>(k, mq >> 3));
            #pragma unroll
            for (int dp = 0; dp < D / 16; dp++)
            {
                uint32_t r[4];
                int kk = kp * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                int d = dp * 16 + (lane >> 4) * 8;
                ldmatrix_x4_trans(r, sK + swz<D>(kk, d >> 3));
                mma_16808(dq[2 * dp], &a[0], r[0]);
                mma_16808(dq[2 * dp], &a[2], r[1]);
                mma_16808(dq[2 * dp + 1], &a[0], r[2]);
                mma_16808(dq[2 * dp + 1], &a[2], r[3]);
            }
        }
        #pragma unroll
        for (int dt = 0; dt < D / 8; dt++)
        {
            #pragma unroll
            for (int e = 0; e < 4; e++)
            {
                int q = qt * TILE + warp * 16 + g + 8 * (e >> 1);
                int d = dt * 8 + 2 * t + (e & 1);
                atomicAdd(&dq_acc[(static_cast<long long>(b) * T + q) * C + h * D + d], dq[dt][e] * scale);
            }
        }
    }

    #pragma unroll
    for (int hh = 0; hh < 2; hh++)
    {
        int key = kvt * TILE + key0 + 8 * hh;
        __half *row = dqkv + (static_cast<long long>(b) * T + key) * ld;
        #pragma unroll
        for (int dt = 0; dt < D / 8; dt++)
        {
            int d = h * D + dt * 8 + 2 * t;
            *reinterpret_cast<__half2 *>(row + C + d) = __floats2half2_rn(dk[dt][2 * hh] * scale, dk[dt][2 * hh + 1] * scale);
            *reinterpret_cast<__half2 *>(row + 2 * C + d) = __floats2half2_rn(dv[dt][2 * hh], dv[dt][2 * hh + 1]);
        }
    }
}

// dqkv[b, t, 0 .. C-1] = dq_acc[b, t, :]
__global__ void flash_bwd_dq_kernel(__half *dqkv, const float *dq_acc, int C, long long total)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        long long bt = idx / C;
        int c = static_cast<int>(idx % C);
        dqkv[bt * 3 * C + c] = __float2half(dq_acc[idx]);
    }
}

template <int D>
void fwd(floatX *atty, float *L, const floatX *qkv, int B, int T, int H, cudaStream_t s)
{
    dim3 grid(T / TILE, B * H);
    flash_fwd_kernel<D><<<grid, THREADS, 0, s>>>(atty, L, qkv, T, H);
    CUDA_CHECK_LAUNCH();
}

template <int D>
void bwd(floatX *dqkv, float *Dq, float *dq_acc, const floatX *datty, const floatX *qkv, const floatX *atty,
         const float *L, int B, int T, int H, cudaStream_t s)
{
    long long rows = static_cast<long long>(B) * H * T;
    flash_bwd_prep_kernel<<<ceil_div(rows, THREADS / 32), THREADS, 0, s>>>(Dq, datty, atty, T, H, D, rows);
    CUDA_CHECK_LAUNCH();
    long long total = static_cast<long long>(B) * T * H * D;
    CUDA_CHECK(cudaMemsetAsync(dq_acc, 0, total * sizeof(float), s));
    dim3 grid(T / TILE, B * H);
    flash_bwd_kernel<D><<<grid, THREADS, 0, s>>>(dqkv, dq_acc, qkv, datty, L, Dq, T, H);
    CUDA_CHECK_LAUNCH();
    flash_bwd_dq_kernel<<<ceil_div(total, 256), 256, 0, s>>>(dqkv, dq_acc, H * D, total);
    CUDA_CHECK_LAUNCH();
}
} // namespace

void flash_attn_fwd(floatX *atty, float *L, const floatX *qkv, int B, int T, int H, int D, cudaStream_t s)
{
    assert(T % TILE == 0 && B * H <= 65535);
    switch (D)
    {
    case 16: fwd<16>(atty, L, qkv, B, T, H, s); break;
    case 32: fwd<32>(atty, L, qkv, B, T, H, s); break;
    case 64: fwd<64>(atty, L, qkv, B, T, H, s); break;
    case 128: fwd<128>(atty, L, qkv, B, T, H, s); break;
    default:
        fprintf(stderr, "error: flash attention supports head sizes 16, 32, 64, 128, not %d\n", D);
        exit(1);
    }
}

void flash_attn_bwd(floatX *dqkv, float *Dq, float *dq_acc, const floatX *datty, const floatX *qkv,
                    const floatX *atty, const float *L, int B, int T, int H, int D, cudaStream_t s)
{
    assert(T % TILE == 0 && B * H <= 65535);
    switch (D)                              // D 128 would need 72 KB of shared memory here
    {
    case 16: bwd<16>(dqkv, Dq, dq_acc, datty, qkv, atty, L, B, T, H, s); break;
    case 32: bwd<32>(dqkv, Dq, dq_acc, datty, qkv, atty, L, B, T, H, s); break;
    case 64: bwd<64>(dqkv, Dq, dq_acc, datty, qkv, atty, L, B, T, H, s); break;
    default:
        fprintf(stderr, "error: the flash attention backward supports head sizes 16, 32, 64, not %d\n", D);
        exit(1);
    }
}

#else   // fp32 build: attention goes through the stage 3 kernels instead

void flash_attn_fwd(floatX *, float *, const floatX *, int, int, int, int, cudaStream_t)
{
    fprintf(stderr, "error: flash attention needs the fp16 build (-DFP16)\n");
    exit(1);
}

void flash_attn_bwd(floatX *, float *, float *, const floatX *, const floatX *, const floatX *, const float *, int,
                    int, int, int, cudaStream_t)
{
    fprintf(stderr, "error: flash attention needs the fp16 build (-DFP16)\n");
    exit(1);
}

#endif
