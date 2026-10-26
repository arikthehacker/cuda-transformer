// decode.cu: one generation step with a KV cache (see decode.h)

#include "decode.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "common.h"
#include "gemm_common.cuh"
#include "reduce.cuh"

using namespace std;

static const int GEMV_THREADS = 256;
static const int GEMV_WARPS = GEMV_THREADS / 32;
static const int MAX_GRID = 1024;
static const int MAX_IC = 4096;             // the input vector lives in shared memory
static const int ATT_THREADS = 256;
static const int MAX_CTX = 2048;
static const int MAX_D = 128;
static const int SAMPLE_THREADS = 1024;
constexpr int VEC = 16 / sizeof(floatX);    // values per 128-bit load: 8 halves or 4 floats

// ---------------------------------------------------------------- GEMV pieces

// dot of VEC weights (one 128-bit load) with VEC inputs from shared memory
__device__ __forceinline__ float dot_vec(const floatX *w, const float *xs)
{
    int4 raw = *reinterpret_cast<const int4 *>(w);
    floatX v[VEC];
    memcpy(v, &raw, sizeof(raw));           // not a pointer cast: reading an int4 as halves through one is undefined
    float acc = 0.0f;
    for (int i = 0; i < VEC; i++)
    {
        acc += to_float(v[i]) * xs[i];
    }
    return acc;
}

// one warp, one row: W[row] . xs, every lane gets the sum
__device__ __forceinline__ float row_dot(const floatX *W, int row, int IC, const float *xs, int lane)
{
    const floatX *w = W + static_cast<long long>(row) * IC;
    float acc = 0.0f;
    for (int c = lane; c < IC / VEC; c += 32)
    {
        acc += dot_vec(w + c * VEC, xs + c * VEC);
    }
    return warp_sum(acc);
}

// xs = floatX(LayerNorm(res)), the same arithmetic as layernorm_row in
// fused.cu: warp 0 finds mean and rstd, then the whole block normalizes.
__device__ void stage_layernorm(float *xs, const float *res, const float *w, const float *b, float eps, int C)
{
    __shared__ float stats[2];
    if (threadIdx.x < 32)
    {
        int lane = threadIdx.x;
        float sum = 0.0f;
        for (int c = lane; c < C; c += 32)
        {
            sum += res[c];
        }
        float m = warp_sum(sum) / C;
        float sq = 0.0f;
        for (int c = lane; c < C; c += 32)
        {
            float d = res[c] - m;
            sq += d * d;
        }
        float s = 1.0f / sqrtf(warp_sum(sq) / C + eps);
        if (lane == 0)
        {
            stats[0] = m;
            stats[1] = s;
        }
    }
    __syncthreads();
    float m = stats[0], s = stats[1];
    for (int c = threadIdx.x; c < C; c += blockDim.x)
    {
        xs[c] = to_float(from_float<floatX>((res[c] - m) * s * w[c] + b[c]));
    }
    __syncthreads();
}

__device__ void stage_vector(float *xs, const floatX *x, int n)
{
    for (int c = threadIdx.x; c < n; c += blockDim.x)
    {
        xs[c] = to_float(x[c]);
    }
    __syncthreads();
}

static int gemv_grid(int rows)
{
    return min(ceil_div(rows, GEMV_WARPS), MAX_GRID);
}

// ---------------------------------------------------------------- GEMV kernels

__global__ void qkv_kernel(floatX *q, floatX *kc, floatX *vc, const float *res, const float *ln_w, const float *ln_b,
                           float eps, const floatX *W, const float *b, const DecodeState *st, int C, int H, int n_ctx)
{
    __shared__ __align__(16) float xs[MAX_IC];
    stage_layernorm(xs, res, ln_w, ln_b, eps, C);
    int pos = st->pos;
    int D = C / H;
    int lane = threadIdx.x % 32;
    for (int o = blockIdx.x * GEMV_WARPS + threadIdx.x / 32; o < 3 * C; o += gridDim.x * GEMV_WARPS)
    {
        float acc = row_dot(W, o, C, xs, lane);
        if (lane == 0)
        {
            floatX v = from_float<floatX>(acc + b[o]);
            int part = o / C, c = o % C;
            if (part == 0)
            {
                q[c] = v;
            }
            else
            {
                floatX *cache = part == 1 ? kc : vc;
                cache[(static_cast<long long>(c / D) * n_ctx + pos) * D + c % D] = v;
            }
        }
    }
}

__global__ void proj_add_kernel(float *res, const floatX *x, const floatX *W, const float *b, int OC, int IC)
{
    __shared__ __align__(16) float xs[MAX_IC];
    stage_vector(xs, x, IC);
    int lane = threadIdx.x % 32;
    for (int o = blockIdx.x * GEMV_WARPS + threadIdx.x / 32; o < OC; o += gridDim.x * GEMV_WARPS)
    {
        float acc = row_dot(W, o, IC, xs, lane);
        if (lane == 0)
        {
            res[o] += to_float(from_float<floatX>(acc + b[o]));     // the full forward keeps this in floatX
        }
    }
}

__global__ void fc_gelu_kernel(floatX *out, const float *res, const float *ln_w, const float *ln_b, float eps,
                               const floatX *W, const float *b, int OC, int C)
{
    __shared__ __align__(16) float xs[MAX_IC];
    stage_layernorm(xs, res, ln_w, ln_b, eps, C);
    int lane = threadIdx.x % 32;
    for (int o = blockIdx.x * GEMV_WARPS + threadIdx.x / 32; o < OC; o += gridDim.x * GEMV_WARPS)
    {
        float acc = row_dot(W, o, C, xs, lane);
        if (lane == 0)
        {
            out[o] = from_float<floatX>(gelu_f(to_float(from_float<floatX>(acc + b[o]))));
        }
    }
}

__global__ void logits_kernel(float *logits, const float *res, const float *ln_w, const float *ln_b, float eps,
                              const floatX *wte, int Vp, int C)
{
    __shared__ __align__(16) float xs[MAX_IC];
    stage_layernorm(xs, res, ln_w, ln_b, eps, C);
    int lane = threadIdx.x % 32;
    for (int o = blockIdx.x * GEMV_WARPS + threadIdx.x / 32; o < Vp; o += gridDim.x * GEMV_WARPS)
    {
        float acc = row_dot(wte, o, C, xs, lane);
        if (lane == 0)
        {
            logits[o] = acc;
        }
    }
}

// ---------------------------------------------------------------- embedding, attention, cache

__global__ void embed_kernel(float *res, const float *wte, const float *wpe, const DecodeState *st, int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C)
    {
        res[c] = wte[static_cast<long long>(st->token) * C + c] + wpe[static_cast<long long>(st->pos) * C + c];
    }
}

// One block per head. Thread j scores key j; then the threads split into
// groups of D (thread d of group g adds keys g, g + G, ...) for p @ V.
__global__ void attention_kernel(floatX *atty, const floatX *q, const floatX *kc, const floatX *vc,
                                 const DecodeState *st, int H, int D, int n_ctx)
{
    __shared__ __align__(16) float qs[MAX_D];
    __shared__ float p[MAX_CTX];
    __shared__ float part[ATT_THREADS];
    int h = blockIdx.x;
    int t = st->pos + 1;                    // keys 0 .. pos, the new one included
    for (int d = threadIdx.x; d < D; d += blockDim.x)
    {
        qs[d] = to_float(q[h * D + d]);
    }
    __syncthreads();
    const floatX *K = kc + static_cast<long long>(h) * n_ctx * D;
    const floatX *V = vc + static_cast<long long>(h) * n_ctx * D;
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < t; j += blockDim.x)
    {
        float s = 0.0f;
        for (int c = 0; c < D / VEC; c++)
        {
            s += dot_vec(K + static_cast<long long>(j) * D + c * VEC, qs + c * VEC);
        }
        s *= scale;
        p[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < t; j += blockDim.x)
    {
        float e = expf(p[j] - mx);
        p[j] = e;
        sum += e;
    }
    sum = block_sum(sum);                   // its barriers also make every p[j] visible
    int G = blockDim.x / D;
    int g = threadIdx.x / D, d = threadIdx.x % D;
    float acc = 0.0f;
    if (g < G)
    {
        for (int j = g; j < t; j += G)
        {
            acc += p[j] * to_float(V[static_cast<long long>(j) * D + d]);
        }
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < D)
    {
        float total = 0.0f;
        for (int k = 0; k < G; k++)
        {
            total += part[k * D + threadIdx.x];
        }
        atty[h * D + threadIdx.x] = from_float<floatX>(total / sum);
    }
}

__global__ void fill_cache_kernel(floatX *kc, floatX *vc, const floatX *qkv, int t, int C, int D, int n_ctx)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(t) * C)
    {
        int j = static_cast<int>(idx / C), c = static_cast<int>(idx % C);
        long long dst = (static_cast<long long>(c / D) * n_ctx + j) * D + c % D;
        kc[dst] = qkv[static_cast<long long>(j) * 3 * C + C + c];
        vc[dst] = qkv[static_cast<long long>(j) * 3 * C + 2 * C + c];
    }
}

// ---------------------------------------------------------------- sampling

// float -> unsigned with the same order (negative numbers flipped)
__host__ __device__ __forceinline__ unsigned order_key(unsigned bits)
{
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__host__ __device__ __forceinline__ unsigned order_unkey(unsigned key)
{
    return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
}

// The weights w_j = exp(l_j - max) are added up by 32 lanes, each over one
// contiguous chunk of ids, then the 33 running totals by one lane in order.
// The chunk that holds rand * total is walked from 0 again, so every number
// is computed one way only, and sample_host() repeats it on the CPU.
template <class TL>
__global__ void __launch_bounds__(SAMPLE_THREADS) sample_kernel(DecodeState *st, int *history, const TL *logits,
                                                                const float *rand, int V, SampleConfig sc, int set_pos)
{
    __shared__ int hist[256];
    __shared__ unsigned sh_prefix;
    __shared__ int sh_left;
    __shared__ float excl[33];
    __shared__ float bestv[SAMPLE_THREADS];
    __shared__ int besti[SAMPLE_THREADS];
    __shared__ int result;
    int tid = threadIdx.x;
    float temp = sc.temperature;
    int chosen;

    if (temp <= 0.0f)
    {
        // greedy: the largest value, the smallest id among equals
        float bv = -INFINITY;
        int bi = V;
        for (int j = tid; j < V; j += blockDim.x)
        {
            float v = to_float(logits[j]);
            if (v > bv)
            {
                bv = v;
                bi = j;
            }
        }
        bestv[tid] = bv;
        besti[tid] = bi;
        __syncthreads();
        for (int half = blockDim.x / 2; half > 0; half /= 2)
        {
            if (tid < half)
            {
                float v2 = bestv[tid + half];
                int i2 = besti[tid + half];
                if (v2 > bestv[tid] || (v2 == bestv[tid] && i2 < besti[tid]))
                {
                    bestv[tid] = v2;
                    besti[tid] = i2;
                }
            }
            __syncthreads();
        }
        chosen = besti[0];
    }
    else
    {
        // the top_k-th largest l / temperature, one byte of its key per pass
        float tau = -INFINITY;
        if (sc.top_k > 0 && sc.top_k < V)
        {
            unsigned prefix = 0, mask = 0;
            int left = sc.top_k;
            for (int shift = 24; shift >= 0; shift -= 8)
            {
                for (int k = tid; k < 256; k += blockDim.x)
                {
                    hist[k] = 0;
                }
                __syncthreads();
                for (int j = tid; j < V; j += blockDim.x)
                {
                    unsigned key = order_key(__float_as_uint(to_float(logits[j]) / temp));
                    if ((key & mask) == prefix)
                    {
                        atomicAdd(&hist[(key >> shift) & 255], 1);
                    }
                }
                __syncthreads();
                if (tid == 0)
                {
                    int above = 0;
                    for (int bin = 255; bin >= 0; bin--)
                    {
                        if (above + hist[bin] >= left)
                        {
                            sh_prefix = prefix | (static_cast<unsigned>(bin) << shift);
                            sh_left = left - above;
                            break;
                        }
                        above += hist[bin];
                    }
                }
                __syncthreads();
                prefix = sh_prefix;
                left = sh_left;
                mask |= 0xffu << shift;
                __syncthreads();            // everyone has read sh_* before the next pass writes them
            }
            tau = __uint_as_float(order_unkey(prefix));
        }
        float mx = -INFINITY;
        for (int j = tid; j < V; j += blockDim.x)
        {
            float l = to_float(logits[j]) / temp;
            if (l >= tau)
            {
                mx = fmaxf(mx, l);
            }
        }
        mx = block_max(mx);
        if (tid < 32)
        {
            int per = (V + 31) / 32;
            int lo = min(V, tid * per), hi = min(V, lo + per);
            float csum = 0.0f;
            int last_kept = -1;
            for (int j = lo; j < hi; j++)
            {
                float l = to_float(logits[j]) / temp;
                if (l >= tau)
                {
                    csum += expf(l - mx);
                    last_kept = j;
                }
            }
            bestv[tid] = csum;
            besti[tid] = last_kept;
            __syncwarp();
            if (tid == 0)
            {
                float run = 0.0f;
                int last = -1;
                for (int k = 0; k < 32; k++)
                {
                    excl[k] = run;
                    run += bestv[k];
                    last = besti[k] >= 0 ? besti[k] : last;
                }
                excl[32] = run;
                result = last;              // used when rand * total rounds up to the total
            }
            __syncwarp();
            float target = rand[st->sampled] * excl[32];
            if (excl[tid] <= target && target < excl[tid + 1])
            {
                float local = target - excl[tid];
                float c = 0.0f;
                int pick = last_kept;
                for (int j = lo; j < hi; j++)
                {
                    float l = to_float(logits[j]) / temp;
                    if (l >= tau)
                    {
                        c += expf(l - mx);
                        if (c > local)
                        {
                            pick = j;
                            break;
                        }
                    }
                }
                result = pick;
            }
        }
        __syncthreads();
        chosen = result;
    }
    if (tid == 0)
    {
        history[st->n] = chosen;
        st->n++;
        st->sampled++;
        st->token = chosen;
        st->pos = set_pos >= 0 ? set_pos : st->pos + 1;
    }
}

int sample_host(const float *logits, int V, SampleConfig sc, float u)
{
    float temp = sc.temperature;
    if (temp <= 0.0f)
    {
        int best = 0;
        for (int j = 1; j < V; j++)
        {
            if (logits[j] > logits[best])
            {
                best = j;
            }
        }
        return best;
    }
    vector<float> l(V);
    for (int j = 0; j < V; j++)
    {
        l[j] = logits[j] / temp;
    }
    float tau = -INFINITY;
    if (sc.top_k > 0 && sc.top_k < V)
    {
        vector<float> sorted = l;
        nth_element(sorted.begin(), sorted.begin() + (sc.top_k - 1), sorted.end(), greater<float>());
        tau = sorted[sc.top_k - 1];
    }
    float mx = -INFINITY;
    for (float x : l)
    {
        if (x >= tau)
        {
            mx = fmaxf(mx, x);
        }
    }
    int per = (V + 31) / 32;
    float excl[33];
    float run = 0.0f;
    int last = -1;
    for (int k = 0; k < 32; k++)
    {
        excl[k] = run;
        int lo = min(V, k * per), hi = min(V, lo + per);
        float csum = 0.0f;
        for (int j = lo; j < hi; j++)
        {
            if (l[j] >= tau)
            {
                csum += expf(l[j] - mx);
                last = j;
            }
        }
        run += csum;
    }
    excl[32] = run;
    float target = u * excl[32];
    for (int k = 0; k < 32; k++)
    {
        if (excl[k] <= target && target < excl[k + 1])
        {
            int lo = min(V, k * per), hi = min(V, lo + per);
            float local = target - excl[k];
            float c = 0.0f;
            int pick = -1;
            for (int j = lo; j < hi; j++)
            {
                if (l[j] >= tau)
                {
                    c += expf(l[j] - mx);
                    pick = j;
                    if (c > local)
                    {
                        return j;
                    }
                }
            }
            return pick;
        }
    }
    return last;
}

// ---------------------------------------------------------------- launchers

void decode_embed(float *res, const float *wte, const float *wpe, const DecodeState *st, int C, cudaStream_t s)
{
    embed_kernel<<<ceil_div(C, 256), 256, 0, s>>>(res, wte, wpe, st, C);
    CUDA_CHECK_LAUNCH();
}

void decode_qkv(floatX *q, floatX *k_cache, floatX *v_cache, const float *res, const float *ln_w, const float *ln_b,
                float eps, const floatX *W, const float *b, const DecodeState *st, int C, int H, int n_ctx,
                cudaStream_t s)
{
    assert(C <= MAX_IC && C % VEC == 0 && C % H == 0);
    qkv_kernel<<<gemv_grid(3 * C), GEMV_THREADS, 0, s>>>(q, k_cache, v_cache, res, ln_w, ln_b, eps, W, b, st, C, H,
                                                          n_ctx);
    CUDA_CHECK_LAUNCH();
}

void decode_proj_add(float *res, const floatX *x, const floatX *W, const float *b, int OC, int IC, cudaStream_t s)
{
    assert(IC <= MAX_IC && IC % VEC == 0);
    proj_add_kernel<<<gemv_grid(OC), GEMV_THREADS, 0, s>>>(res, x, W, b, OC, IC);
    CUDA_CHECK_LAUNCH();
}

void decode_fc_gelu(floatX *out, const float *res, const float *ln_w, const float *ln_b, float eps, const floatX *W,
                    const float *b, int OC, int C, cudaStream_t s)
{
    assert(C <= MAX_IC && C % VEC == 0);
    fc_gelu_kernel<<<gemv_grid(OC), GEMV_THREADS, 0, s>>>(out, res, ln_w, ln_b, eps, W, b, OC, C);
    CUDA_CHECK_LAUNCH();
}

void decode_logits(float *logits, const float *res, const float *ln_w, const float *ln_b, float eps,
                   const floatX *wte, int Vp, int C, cudaStream_t s)
{
    assert(C <= MAX_IC && C % VEC == 0);
    logits_kernel<<<gemv_grid(Vp), GEMV_THREADS, 0, s>>>(logits, res, ln_w, ln_b, eps, wte, Vp, C);
    CUDA_CHECK_LAUNCH();
}

void decode_attention(floatX *atty, const floatX *q, const floatX *k_cache, const floatX *v_cache,
                      const DecodeState *st, int H, int D, int n_ctx, cudaStream_t s)
{
    assert(D <= MAX_D && D % VEC == 0 && ATT_THREADS % D == 0 && n_ctx <= MAX_CTX);
    attention_kernel<<<H, ATT_THREADS, 0, s>>>(atty, q, k_cache, v_cache, st, H, D, n_ctx);
    CUDA_CHECK_LAUNCH();
}

void fill_cache(floatX *k_cache, floatX *v_cache, const floatX *qkv, int t, int C, int H, int n_ctx, cudaStream_t s)
{
    assert(t > 0 && t <= n_ctx);
    fill_cache_kernel<<<ceil_div(static_cast<long long>(t) * C, 256), 256, 0, s>>>(k_cache, v_cache, qkv, t, C, C / H,
                                                                                  n_ctx);
    CUDA_CHECK_LAUNCH();
}

template <class TL>
void sample_token(DecodeState *st, int *history, const TL *logits, const float *rand, int V, SampleConfig sc,
                  int set_pos, cudaStream_t s)
{
    sample_kernel<TL><<<1, SAMPLE_THREADS, 0, s>>>(st, history, logits, rand, V, sc, set_pos);
    CUDA_CHECK_LAUNCH();
}

template void sample_token<float>(DecodeState *, int *, const float *, const float *, int, SampleConfig, int,
                                  cudaStream_t);

__global__ void push_kernel(DecodeState *st, int *history, int token, int set_pos)
{
    history[st->n] = token;
    st->n++;
    st->token = token;
    st->pos = set_pos >= 0 ? set_pos : st->pos + 1;
}

__global__ void replace_kernel(DecodeState *st, int *history, int token)
{
    history[st->n - 1] = token;
    st->token = token;
}

void push_token(DecodeState *st, int *history, int token, int set_pos, cudaStream_t s)
{
    push_kernel<<<1, 1, 0, s>>>(st, history, token, set_pos);
    CUDA_CHECK_LAUNCH();
}

void replace_token(DecodeState *st, int *history, int token, cudaStream_t s)
{
    replace_kernel<<<1, 1, 0, s>>>(st, history, token);
    CUDA_CHECK_LAUNCH();
}
#ifdef FP16
template void sample_token<__half>(DecodeState *, int *, const __half *, const float *, int, SampleConfig, int,
                                   cudaStream_t);
#endif
