// generator.cu: Generator (see generator.h)

#include "generator.h"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace std;

Generator::Generator(const ModelConfig &c, const vector<float> &params)
    : cfg(c), model(c, 1, c.n_ctx, TrainConfig(), 1.0f)
{
    model.set_params(params);
    size_t cache = static_cast<size_t>(cfg.L) * cfg.n_ctx * cfg.C;     // L * H * n_ctx * D
    k_cache = DeviceArray<floatX>(cache);
    v_cache = DeviceArray<floatX>(cache);
    res = DeviceBuffer(cfg.C);
    logits = DeviceBuffer(cfg.Vp);
    q = DeviceArray<floatX>(cfg.C);
    atty = DeviceArray<floatX>(cfg.C);
    fc = DeviceArray<floatX>(4 * static_cast<size_t>(cfg.C));
    state = DeviceArray<DecodeState>(1);
    prefill_loss = DeviceBuffer(1);
    model.y.zero(model.stream);             // prefill computes a loss nobody reads; its targets just have to be valid
}

void Generator::reserve(int tokens)
{
    if (tokens <= capacity)
    {
        return;
    }
    capacity = tokens;
    history = DeviceInts(tokens);
    rand = DeviceBuffer(tokens);
    graph_sc = {-1.0f, -1};                 // the graph points at the old buffers
}

// The full forward pass over history[n - m .. n) at positions 0 .. m - 1,
// padded with token 0 to a multiple of 64 (causal attention: the padding
// comes after every real row, so no real row sees it).
void Generator::prefill(int m, int n, SampleConfig sc, bool fill)
{
    cudaStream_t s = model.stream;
    int t = (m + 63) / 64 * 64;
    model.set_seq_len(t);
    CUDA_CHECK(cudaMemsetAsync(model.x.ptr, 0, t * sizeof(int), s));
    CUDA_CHECK(cudaMemcpyAsync(model.x.ptr, history.ptr + (n - m), m * sizeof(int), cudaMemcpyDeviceToDevice, s));
    model.forward(false, prefill_loss.ptr);
    if (fill)
    {
        size_t per_layer = static_cast<size_t>(cfg.n_ctx) * cfg.C;
        for (int l = 0; l < cfg.L; l++)
        {
            fill_cache(k_cache.ptr + l * per_layer, v_cache.ptr + l * per_layer, model.layers[l].qkv, m, cfg.C, cfg.H,
                       cfg.n_ctx, s);
        }
    }
    sample_token(state.ptr, history.ptr, model.logits + static_cast<size_t>(m - 1) * cfg.Vp, rand.ptr, cfg.V, sc, m,
                 s);
}

void Generator::decode_step(SampleConfig sc)
{
    cudaStream_t s = model.stream;
    int C = cfg.C, H = cfg.H, D = cfg.D();
    float eps = cfg.ln_eps;
    size_t per_layer = static_cast<size_t>(cfg.n_ctx) * C;
    decode_embed(res.ptr, model.wte, model.wpe, state.ptr, C, s);
    for (int l = 0; l < cfg.L; l++)
    {
        const LayerParams &w = model.lw[l];
        const LayerWeightsX &wx = model.lx[l];
        floatX *kc = k_cache.ptr + l * per_layer;
        floatX *vc = v_cache.ptr + l * per_layer;
        decode_qkv(q.ptr, kc, vc, res.ptr, w.ln1_w, w.ln1_b, eps, wx.attn_w, w.attn_b, state.ptr, C, H, cfg.n_ctx, s);
        decode_attention(atty.ptr, q.ptr, kc, vc, state.ptr, H, D, cfg.n_ctx, s);
        decode_proj_add(res.ptr, atty.ptr, wx.proj_w, w.proj_b, C, C, s);
        decode_fc_gelu(fc.ptr, res.ptr, w.ln2_w, w.ln2_b, eps, wx.fc_w, w.fc_b, 4 * C, C, s);
        decode_proj_add(res.ptr, fc.ptr, wx.fcproj_w, w.fcproj_b, C, 4 * C, s);
    }
    decode_logits(logits.ptr, res.ptr, model.lnf_w, model.lnf_b, eps, model.wte_x, cfg.Vp, C, s);
    sample_token(state.ptr, history.ptr, logits.ptr, rand.ptr, cfg.V, sc, -1, s);
}

void Generator::report(int window_start, int pos, bool from_prefill, int m)
{
    if (!on_logits)
    {
        return;
    }
    vector<float> host(cfg.Vp);
    if (from_prefill)
    {
        ActView v{model.logits + static_cast<size_t>(m - 1) * cfg.Vp, true, static_cast<size_t>(cfg.Vp)};
        host = download_view(v, model.stream);
    }
    else
    {
        CUDA_CHECK(cudaStreamSynchronize(model.stream));
        host = logits.download();
    }
    on_logits(host, window_start, pos);
}

vector<int> Generator::generate(const vector<int> &prompt, int max_new, SampleConfig sc, unsigned long long seed,
                                Mode mode)
{
    int p = static_cast<int>(prompt.size());
    if (p < 1 || p > cfg.n_ctx || max_new < 1)
    {
        fprintf(stderr, "error: the prompt needs 1 to %d tokens and max_new at least 1\n", cfg.n_ctx);
        exit(1);
    }
    reserve(p + max_new);
    cudaStream_t s = model.stream;

    // the random numbers come from the host, so a seed always gives the same sample
    mt19937_64 rng(seed);
    uniform_real_distribution<float> uni(0.0f, 1.0f);
    vector<float> r(max_new);
    for (float &x : r)
    {
        x = uni(rng);
    }
    rand.upload(r.data(), r.size());
    history.upload(prompt.data(), prompt.size());
    DecodeState st = {prompt.back(), 0, p, 0};
    state.upload(&st, 1);

    bool use_graph = mode == Mode::CachedGraph && !on_logits;
    if (use_graph && (graph_sc.temperature != sc.temperature || graph_sc.top_k != sc.top_k))
    {
        graph.capture(s, [&]() { decode_step(sc); });
        graph_sc = sc;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, s));
    int n = p;                              // the host knows every count without asking the GPU
    if (mode == Mode::NoCache)
    {
        while (n - p < max_new)
        {
            int m = min(n, cfg.n_ctx);
            prefill(m, n, sc, false);
            report(n - m, m - 1, true, m);
            n++;
        }
    }
    else
    {
        prefill(p, n, sc, true);
        report(0, p - 1, true, p);
        int window_start = 0, pos = p;
        n++;
        while (n - p < max_new)
        {
            if (pos == cfg.n_ctx)           // the cache is full: keep the last half, recompute it
            {
                int m = cfg.n_ctx / 2;
                prefill(m, n, sc, true);
                window_start = n - m;
                report(window_start, m - 1, true, m);
                pos = m;
            }
            else
            {
                if (use_graph)
                {
                    graph.launch(s);
                }
                else
                {
                    decode_step(sc);
                }
                report(window_start, pos, false, 0);
                pos++;
            }
            n++;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&last_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    model.set_seq_len(model.T_max);

    vector<int> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), history.ptr, n * sizeof(int), cudaMemcpyDeviceToHost));
    return out;
}
