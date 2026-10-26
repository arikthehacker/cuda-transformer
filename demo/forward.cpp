// forward.cpp: the GPT forward pass with a KV cache, on the CPU, for the
// browser. Compiled to WebAssembly with SIMD and no C library:
//
//   clang++ --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -fno-exceptions -fno-rtti \
//           -Wl,--no-entry -o demo/forward.wasm demo/forward.cpp       (make demo)
//
// Plain clang and wasm-ld are enough (no Emscripten): there is no C library,
// memory comes from memory.grow, and the page talks to five exported functions.
//
// The page (index.html) copies weights.bin into wasm memory, calls load(),
// then step(token, pos) once per token and sample() on the logits. The math
// is the CUDA decode step's (cuda/kernels/decode.cu), in fp32: weights are
// shipped as fp16 to halve the download and widened once at load time.

#include <wasm_simd128.h>

extern unsigned char __heap_base;

namespace
{
unsigned char *heap = &__heap_base;

void *grab(unsigned bytes)                  // bump allocator: the model allocates once
{
    unsigned long at = (reinterpret_cast<unsigned long>(heap) + 15) & ~15ul;
    unsigned long end = at + bytes;
    unsigned long have = __builtin_wasm_memory_size(0) * 65536ul;
    if (end > have)
    {
        unsigned long pages = (end - have + 65535) / 65536;
        if (__builtin_wasm_memory_grow(0, pages) == static_cast<unsigned long>(-1))
        {
            return nullptr;
        }
    }
    heap = reinterpret_cast<unsigned char *>(end);
    return reinterpret_cast<void *>(at);
}

// ---------------------------------------------------------------- math without libm

float exp_approx(float x)                   // relative error about 2e-7 over the range softmax and gelu use
{
    if (x < -87.0f)
    {
        return 0.0f;
    }
    if (x > 88.0f)
    {
        x = 88.0f;
    }
    const float ln2 = 0.69314718056f;
    float n = __builtin_floorf(x / ln2 + 0.5f);
    float r = x - n * ln2;                  // |r| <= ln2 / 2
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6 + r * (1.0f / 24 + r * (1.0f / 120 + r * (1.0f / 720))))));
    int e = static_cast<int>(n) + 127;
    float scale = __builtin_bit_cast(float, e << 23);
    return p * scale;
}

float tanh_approx(float x)
{
    if (x > 9.0f)
    {
        return 1.0f;
    }
    if (x < -9.0f)
    {
        return -1.0f;
    }
    float e = exp_approx(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
}

float gelu(float x)
{
    const float s = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanh_approx(s * (x + 0.044715f * x * x * x)));
}

float half_to_float(unsigned short h)
{
    unsigned sign = (h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 0x1f;
    unsigned man = h & 0x3ff;
    unsigned bits;
    if (exp == 0)
    {
        if (man == 0)
        {
            bits = sign;
        }
        else                                // subnormal: renormalize
        {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0)
            {
                man <<= 1;
                exp--;
            }
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    }
    else if (exp == 31)
    {
        bits = sign | 0x7f800000u | (man << 13);
    }
    else
    {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    return __builtin_bit_cast(float, bits);
}

// ---------------------------------------------------------------- the model

struct Model
{
    int V, Vp, n_ctx, L, H, C;
    float eps;
    float *params;                          // dump order (cuda/layout.h)
    float *wte, *wpe, *lnf_w, *lnf_b;
    struct Layer
    {
        float *ln1_w, *ln1_b, *attn_w, *attn_b, *proj_w, *proj_b, *ln2_w, *ln2_b, *fc_w, *fc_b, *fcproj_w, *fcproj_b;
    } *layers;
    float *k_cache, *v_cache;               // (L, n_ctx, C): position t of layer l is one row of C
    float *x, *ln, *qkv, *att, *fc, *logits, *p;
};

Model m;
bool ready = false;

// y (OC) = W (OC x IC) @ x + b, four rows at a time, 4-wide SIMD
void matvec(float *y, const float *W, const float *x, const float *b, int OC, int IC)
{
    for (int o = 0; o < OC; o++)
    {
        const float *w = W + static_cast<long>(o) * IC;
        v128_t acc0 = wasm_f32x4_splat(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        int i = 0;
        for (; i + 16 <= IC; i += 16)
        {
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(w + i), wasm_v128_load(x + i)));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(w + i + 4), wasm_v128_load(x + i + 4)));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(w + i + 8), wasm_v128_load(x + i + 8)));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(w + i + 12), wasm_v128_load(x + i + 12)));
        }
        v128_t acc = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3));
        float sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
                    wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
        for (; i < IC; i++)
        {
            sum += w[i] * x[i];
        }
        y[o] = sum + (b ? b[o] : 0.0f);
    }
}

void layernorm(float *out, const float *x, const float *w, const float *b, int C, float eps)
{
    float mean = 0.0f;
    for (int c = 0; c < C; c++)
    {
        mean += x[c];
    }
    mean /= C;
    float var = 0.0f;
    for (int c = 0; c < C; c++)
    {
        float d = x[c] - mean;
        var += d * d;
    }
    float rstd = 1.0f / __builtin_sqrtf(var / C + eps);
    for (int c = 0; c < C; c++)
    {
        out[c] = (x[c] - mean) * rstd * w[c] + b[c];
    }
}

// the k-th largest of a[0 .. n-1] (k from 1, equal values counted), reordering a (quickselect)
float kth_largest(float *a, int n, int k)
{
    int lo = 0, hi = n - 1, want = k - 1;   // the index it would have, sorted largest first
    while (lo < hi)
    {
        float pivot = a[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j)
        {
            while (a[i] > pivot)
            {
                i++;
            }
            while (a[j] < pivot)
            {
                j--;
            }
            if (i <= j)
            {
                float t = a[i];
                a[i] = a[j];
                a[j] = t;
                i++;
                j--;
            }
        }
        if (want <= j)
        {
            hi = j;
        }
        else if (want >= i)
        {
            lo = i;
        }
        else
        {
            return a[want];                 // between j and i every value equals the pivot
        }
    }
    return a[lo];
}

unsigned read_u32(const unsigned char *p)
{
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<unsigned>(p[3]) << 24);
}
} // namespace

extern "C" {

__attribute__((export_name("alloc"))) void *alloc(unsigned bytes)
{
    return grab(bytes);
}

// weights.bin (cuda/weights.h): 0 on success, else what was wrong
__attribute__((export_name("load"))) int load(const unsigned char *file, unsigned bytes)
{
    if (bytes < 48 || read_u32(file) != 0x57545047u)
    {
        return 1;                           // not a weights.bin file
    }
    if (read_u32(file + 4) != 1)
    {
        return 2;                           // another version
    }
    m.V = read_u32(file + 8);
    m.Vp = read_u32(file + 12);
    m.n_ctx = read_u32(file + 16);
    m.L = read_u32(file + 20);
    m.H = read_u32(file + 24);
    m.C = read_u32(file + 28);
    m.eps = __builtin_bit_cast(float, read_u32(file + 32));
    unsigned dtype = read_u32(file + 36);
    unsigned count = read_u32(file + 40);   // the high 32 bits of the uint64 count are 0 here
    int C = m.C;
    unsigned want = m.Vp * C + m.n_ctx * C + m.L * (12 * C * C + 13 * C) + 2 * C;
    if (dtype > 1 || count != want || bytes != 48 + count * (dtype == 1 ? 2 : 4) || C % m.H != 0)
    {
        return 3;                           // sizes do not add up
    }
    m.params = static_cast<float *>(grab(count * 4));
    m.layers = static_cast<Model::Layer *>(grab(m.L * sizeof(Model::Layer)));
    unsigned cache = m.L * m.n_ctx * C;
    m.k_cache = static_cast<float *>(grab(cache * 4));
    m.v_cache = static_cast<float *>(grab(cache * 4));
    m.x = static_cast<float *>(grab(C * 4));
    m.ln = static_cast<float *>(grab(C * 4));
    m.qkv = static_cast<float *>(grab(3 * C * 4));
    m.att = static_cast<float *>(grab(C * 4));
    m.fc = static_cast<float *>(grab(4 * C * 4));
    m.logits = static_cast<float *>(grab(m.Vp * 4));
    m.p = static_cast<float *>(grab((m.n_ctx > m.Vp ? m.n_ctx : m.Vp) * 4));
    if (!m.params || !m.p)
    {
        return 4;                           // out of memory
    }
    const unsigned char *data = file + 48;
    for (unsigned i = 0; i < count; i++)
    {
        m.params[i] = dtype == 1 ? half_to_float(data[2 * i] | (data[2 * i + 1] << 8))
                                 : __builtin_bit_cast(float, read_u32(data + 4 * i));
    }
    float *at = m.params;
    auto take = [&](unsigned n)
    {
        float *p = at;
        at += n;
        return p;
    };
    m.wte = take(m.Vp * C);
    m.wpe = take(m.n_ctx * C);
    for (int l = 0; l < m.L; l++)
    {
        Model::Layer &y = m.layers[l];
        y.ln1_w = take(C);
        y.ln1_b = take(C);
        y.attn_w = take(3 * C * C);
        y.attn_b = take(3 * C);
        y.proj_w = take(C * C);
        y.proj_b = take(C);
        y.ln2_w = take(C);
        y.ln2_b = take(C);
        y.fc_w = take(4 * C * C);
        y.fc_b = take(4 * C);
        y.fcproj_w = take(4 * C * C);
        y.fcproj_b = take(C);
    }
    m.lnf_w = take(C);
    m.lnf_b = take(C);
    ready = true;
    return 0;
}

// 0 V, 1 Vp, 2 n_ctx, 3 L, 4 H, 5 C
__attribute__((export_name("config"))) int config(int what)
{
    const int v[6] = {m.V, m.Vp, m.n_ctx, m.L, m.H, m.C};
    return what >= 0 && what < 6 ? v[what] : -1;
}

// One token at position pos (0 .. n_ctx - 1). Its K and V go into the cache
// at pos; attention sees positions 0 .. pos. Returns the logits (Vp floats).
__attribute__((export_name("step"))) float *step(int token, int pos)
{
    if (!ready || token < 0 || token >= m.V || pos < 0 || pos >= m.n_ctx)
    {
        return nullptr;
    }
    int C = m.C, H = m.H, D = C / H;
    float scale = 1.0f / __builtin_sqrtf(static_cast<float>(D));
    for (int c = 0; c < C; c++)
    {
        m.x[c] = m.wte[token * C + c] + m.wpe[pos * C + c];
    }
    for (int l = 0; l < m.L; l++)
    {
        const Model::Layer &y = m.layers[l];
        float *kc = m.k_cache + static_cast<long>(l) * m.n_ctx * C;
        float *vc = m.v_cache + static_cast<long>(l) * m.n_ctx * C;
        layernorm(m.ln, m.x, y.ln1_w, y.ln1_b, C, m.eps);
        matvec(m.qkv, y.attn_w, m.ln, y.attn_b, 3 * C, C);
        for (int c = 0; c < C; c++)
        {
            kc[pos * C + c] = m.qkv[C + c];
            vc[pos * C + c] = m.qkv[2 * C + c];
        }
        for (int h = 0; h < H; h++)
        {
            const float *q = m.qkv + h * D;
            float mx = -3.4e38f;
            for (int t = 0; t <= pos; t++)
            {
                const float *k = kc + t * C + h * D;
                float s = 0.0f;
                for (int d = 0; d < D; d++)
                {
                    s += q[d] * k[d];
                }
                m.p[t] = s * scale;
                mx = m.p[t] > mx ? m.p[t] : mx;
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++)
            {
                m.p[t] = exp_approx(m.p[t] - mx);
                sum += m.p[t];
            }
            float *out = m.att + h * D;
            for (int d = 0; d < D; d++)
            {
                out[d] = 0.0f;
            }
            for (int t = 0; t <= pos; t++)
            {
                const float *v = vc + t * C + h * D;
                float w = m.p[t] / sum;
                for (int d = 0; d < D; d++)
                {
                    out[d] += w * v[d];
                }
            }
        }
        matvec(m.ln, y.proj_w, m.att, y.proj_b, C, C);
        for (int c = 0; c < C; c++)
        {
            m.x[c] += m.ln[c];
        }
        layernorm(m.ln, m.x, y.ln2_w, y.ln2_b, C, m.eps);
        matvec(m.fc, y.fc_w, m.ln, y.fc_b, 4 * C, C);
        for (int c = 0; c < 4 * C; c++)
        {
            m.fc[c] = gelu(m.fc[c]);
        }
        matvec(m.ln, y.fcproj_w, m.fc, y.fcproj_b, C, 4 * C);
        for (int c = 0; c < C; c++)
        {
            m.x[c] += m.ln[c];
        }
    }
    layernorm(m.ln, m.x, m.lnf_w, m.lnf_b, C, m.eps);
    matvec(m.logits, m.wte, m.ln, nullptr, m.Vp, C);
    return m.logits;
}

// The draw of cuda/kernels/decode.cu on the last logits: temperature 0 is the
// argmax; else l / temperature, keep the top_k largest (ties at the k-th too),
// softmax, and the first id whose running sum passes u * total.
__attribute__((export_name("sample"))) int sample(float temperature, int top_k, float u)
{
    int V = m.V;
    const float *l = m.logits;
    if (temperature <= 0.0f)
    {
        int best = 0;
        for (int j = 1; j < V; j++)
        {
            best = l[j] > l[best] ? j : best;
        }
        return best;
    }
    float tau = -3.4e38f;
    if (top_k > 0 && top_k < V)
    {
        for (int j = 0; j < V; j++)
        {
            m.p[j] = l[j] / temperature;
        }
        tau = kth_largest(m.p, V, top_k);
    }
    float mx = -3.4e38f;
    for (int j = 0; j < V; j++)
    {
        float v = l[j] / temperature;
        mx = v >= tau && v > mx ? v : mx;
    }
    float total = 0.0f;
    int last = 0;
    for (int j = 0; j < V; j++)
    {
        float v = l[j] / temperature;
        m.p[j] = v >= tau ? exp_approx(v - mx) : 0.0f;
        total += m.p[j];
        last = m.p[j] > 0.0f ? j : last;
    }
    float target = u * total, run = 0.0f;
    for (int j = 0; j < V; j++)
    {
        run += m.p[j];
        if (m.p[j] > 0.0f && run > target)
        {
            return j;
        }
    }
    return last;
}

} // extern "C"
