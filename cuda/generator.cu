// generator.cu: Generator (see generator.h)

#include "generator.h"

#include <algorithm>
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
void Generator::prefill(int m, int n, SampleConfig sc, bool fill, bool sample)
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
    if (sample)
    {
        sample_token(state.ptr, history.ptr, model.logits + static_cast<size_t>(m - 1) * cfg.Vp, rand.ptr, cfg.V, sc,
                     m, s);
    }
}

void Generator::decode_layers()
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
}

void Generator::decode_step(SampleConfig sc)
{
    cudaStream_t s = model.stream;
    decode_layers();
    decode_logits(logits.ptr, res.ptr, model.lnf_w, model.lnf_b, cfg.ln_eps, model.wte_x, cfg.Vp, cfg.C, s);
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

// ---------------------------------------------------------------- chat

void Generator::chat_start(int max_new, unsigned long long seed)
{
    if (max_new < 1)
    {
        fprintf(stderr, "error: an answer needs room for at least 1 token\n");
        exit(1);
    }
    chat_max_new = max_new;
    // room for several turns past a full cache; chat_room moves the tail to the front when it fills
    reserve(4 * cfg.n_ctx + max_new + 2);
    chat_rng.seed(seed);
    chat_n = 0;
    chat_pos = 0;
    chat_total = 0;
}

// Only history[n - n_ctx .. n) can matter again (the cache holds at most n_ctx
// positions, a rollover reads the last n_ctx / 2), so when the history would
// overflow, those move to the front. The cache and the positions do not change.
void Generator::chat_room(int needed)
{
    if (chat_n + needed <= capacity)
    {
        return;
    }
    int keep = cfg.n_ctx;
    if (chat_n < 2 * keep || keep + needed > capacity)
    {
        fprintf(stderr, "error: %d more tokens do not fit the chat history of %d\n", needed, capacity);
        exit(1);
    }
    cudaStream_t s = model.stream;
    CUDA_CHECK(cudaMemcpyAsync(history.ptr, history.ptr + (chat_n - keep), keep * sizeof(int),
                               cudaMemcpyDeviceToDevice, s));       // no overlap: chat_n >= 2 * keep
    DecodeState st;
    CUDA_CHECK(cudaMemcpyAsync(&st, state.ptr, sizeof(st), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    st.n = keep;
    CUDA_CHECK(cudaMemcpyAsync(state.ptr, &st, sizeof(st), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));   // st lives on this stack frame
    chat_n = keep;
}

// The pending token goes into the cache (or the cache rolls over with it
// inside), then `token` is the pending one.
void Generator::feed_one(int token)
{
    cudaStream_t s = model.stream;
    if (chat_pos == cfg.n_ctx)
    {
        int m = cfg.n_ctx / 2;
        prefill(m, chat_n, SampleConfig{0.0f, 0}, true, false);
        push_token(state.ptr, history.ptr, token, m, s);
        chat_pos = m;
    }
    else
    {
        decode_layers();
        push_token(state.ptr, history.ptr, token, -1, s);
        chat_pos++;
    }
    chat_n++;
    chat_total++;
}

void Generator::chat_feed(const vector<int> &tokens)
{
    if (tokens.empty())
    {
        return;
    }
    if (chat_max_new == 0)
    {
        fprintf(stderr, "error: chat_feed before chat_start\n");
        exit(1);
    }
    size_t k = 0;
    if (chat_n == 0)                        // a new conversation: the first token just waits
    {
        DecodeState st = {tokens[0], 0, 1, 0};
        history.upload(tokens.data(), 1);
        state.upload(&st, 1);
        chat_n = 1;
        chat_pos = 0;
        chat_total = 1;
        k = 1;
    }
    for (size_t fed = 0; k < tokens.size(); k++, fed++)
    {
        if (fed % cfg.n_ctx == 0)           // room for the next n_ctx at a time, so any length fits
        {
            chat_room(static_cast<int>(min(tokens.size() - k, static_cast<size_t>(cfg.n_ctx))));
        }
        feed_one(tokens[k]);
    }
}

vector<int> Generator::chat_answer(int max_new, SampleConfig sc, int stop_from, int next, Mode mode,
                                   const function<void(int)> &on_token)
{
    if (chat_n == 0 || max_new < 1 || max_new > chat_max_new)
    {
        fprintf(stderr, "error: chat_answer needs a fed conversation and 1 to %d new tokens\n", chat_max_new);
        exit(1);
    }
    cudaStream_t s = model.stream;
    chat_room(max_new + 2);

    // fresh random numbers for this answer, sampled counts from 0 again
    uniform_real_distribution<float> uni(0.0f, 1.0f);
    vector<float> r(max_new);
    for (float &x : r)
    {
        x = uni(chat_rng);
    }
    rand.upload(r.data(), r.size());
    DecodeState st;
    CUDA_CHECK(cudaMemcpyAsync(&st, state.ptr, sizeof(st), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    st.sampled = 0;
    st.ban_from = stop_from;
    CUDA_CHECK(cudaMemcpyAsync(state.ptr, &st, sizeof(st), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    bool use_graph = mode == Mode::CachedGraph;
    if (use_graph && (graph_sc.temperature != sc.temperature || graph_sc.top_k != sc.top_k))
    {
        graph.capture(s, [&]() { decode_step(sc); });
        graph_sc = sc;
    }

    vector<int> out;
    for (int k = 0; k < max_new; k++)
    {
        if (chat_pos == cfg.n_ctx)          // the cache is full: keep the last half, recompute it
        {
            int m = cfg.n_ctx / 2;
            prefill(m, chat_n, sc, true);
            chat_pos = m;
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
            chat_pos++;
        }
        chat_n++;
        chat_total++;
        CUDA_CHECK(cudaMemcpyAsync(&st, state.ptr, sizeof(st), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));   // one token at a time, so it can be printed as it comes
        out.push_back(st.token);
        if (on_token)
        {
            on_token(st.token);
        }
        if (st.token >= stop_from)
        {
            if (st.token != next)
            {
                replace_token(state.ptr, history.ptr, next, s);
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
            return out;
        }
    }
    feed_one('\n');                         // ran out: end the answer the way the training text does
    feed_one(next);
    CUDA_CHECK(cudaStreamSynchronize(s));
    return out;
}
