// test_generate.cu: KV cache generation against the full forward pass.
//
//   ./build/test_generate   data/ref/debug.bin     (fp16 build, tolerance 1e-2)
//   ./build/test_generate32 data/ref/debug.bin     (fp32 build, tolerance 1e-5)
//
//  1. the GPU sampler picks the same token as the CPU version of the draw
//  2. greedy decoding past a full cache: the logits of every cached step
//     match the full forward pass over the same window, at every position,
//     before and after the cache rolls over
//  3. the CUDA Graph replay gives the same tokens as launching the kernels
//  4. a seed reproduces a sample, and no-cache decoding agrees (fp32 build)
//  5. chat: tokens fed one decode step each give what a prefill gives, an
//     answer stops at a tag and leaves <user> in place, and a conversation
//     kept in the cache over many turns (past rollovers and a history
//     compaction) answers exactly like the same conversation fed at once,
//     and an answer never starts with a tag
// The weights are the dump's (any weights work: nothing here needs training).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "dump.h"
#include "generator.h"

using namespace std;

#ifdef FP16
static const float TOL = 1e-2f;
#else
static const float TOL = 1e-5f;
#endif

static int g_failed = 0;

static void check(bool ok, const string &what)
{
    printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        g_failed++;
    }
}

static vector<float> params_from_dump(const Dump &d, const ModelConfig &cfg)
{
    vector<float> params;
    for (const ParamSlot &s : param_layout(cfg))
    {
        const Record &r = d.at(s.name);
        params.insert(params.end(), r.f.begin(), r.f.end());
    }
    return params;
}

static int argmax(const vector<float> &v, int V)
{
    return static_cast<int>(max_element(v.begin(), v.begin() + V) - v.begin());
}

// ---------------------------------------------------------------- 1. the sampler

static void sampler_checks(const ModelConfig &cfg)
{
    printf("\nsampling on the GPU against the same draw on the CPU\n");
    int V = cfg.V, Vp = cfg.Vp;
    mt19937 rng(5);
    normal_distribution<float> normal(0.0f, 3.0f);
    uniform_real_distribution<float> uni(0.0f, 1.0f);
    DeviceBuffer lg(Vp), rnd(1);
    DeviceArray<floatX> lgx(Vp);
    DeviceArray<DecodeState> st(1);
    DeviceInts hist(1);
    const float temps[] = {0.0f, 0.5f, 0.8f, 1.0f, 1.7f};
    const int ks[] = {0, 1, 5, 40, V};
    int same = 0, cases = 0, near = 0, same_x = 0, cases_x = 0;
    for (int rep = 0; rep < 3; rep++)
    {
        for (float temp : temps)
        {
            for (int k : ks)
            {
                vector<float> l(Vp, 0.0f);
                for (int j = 0; j < V; j++)
                {
                    l[j] = normal(rng);
                }
                if (rep == 2)               // ties: every value from a few levels
                {
                    for (int j = 0; j < V; j++)
                    {
                        l[j] = floorf(l[j]);
                    }
                }
                float u = uni(rng);
                SampleConfig sc = {temp, k};
                // the decision must not sit within rounding of a boundary
                int want = sample_host(l.data(), V, sc, u);
                if (sample_host(l.data(), V, sc, fmaxf(0.0f, u - 1e-4f)) != want ||
                    sample_host(l.data(), V, sc, fminf(0.99999f, u + 1e-4f)) != want)
                {
                    near++;
                    continue;
                }
                lg.upload(l.data(), l.size());
                rnd.upload(&u, 1);
                DecodeState s0 = {0, 0, 0, 0};
                st.upload(&s0, 1);
                sample_token(st.ptr, hist.ptr, lg.ptr, rnd.ptr, V, sc, -1, 0);
                vector<int> got = hist.download();
                DecodeState s1;
                CUDA_CHECK(cudaMemcpy(&s1, st.ptr, sizeof(s1), cudaMemcpyDeviceToHost));
                cases++;
                same += got[0] == want && s1.token == want && s1.n == 1 && s1.sampled == 1 && s1.pos == 1;

                // the same draw from floatX logits (the prefill path reads those)
                vector<floatX> lx(Vp);
                vector<float> lr(Vp);
                for (int j = 0; j < Vp; j++)
                {
                    lx[j] = from_float<floatX>(l[j]);
                    lr[j] = to_float(lx[j]);
                }
                int want_x = sample_host(lr.data(), V, sc, u);
                if (sample_host(lr.data(), V, sc, fmaxf(0.0f, u - 1e-4f)) == want_x &&
                    sample_host(lr.data(), V, sc, fminf(0.99999f, u + 1e-4f)) == want_x)
                {
                    lgx.upload(lx.data(), lx.size());
                    st.upload(&s0, 1);
                    sample_token(st.ptr, hist.ptr, lgx.ptr, rnd.ptr, V, sc, 9, 0);
                    CUDA_CHECK(cudaMemcpy(&s1, st.ptr, sizeof(s1), cudaMemcpyDeviceToHost));
                    cases_x++;
                    same_x += s1.token == want_x && s1.pos == 9;
                }
            }
        }
    }
    printf("  %d draws compared, %d skipped (within 1e-4 of a boundary)\n", cases, near);
    check(same == cases, "fp32 logits: GPU and CPU pick the same token, state moves on");
    check(same_x == cases_x, string(FLOATX_NAME) + " logits (prefill path): same token, pos set to 9");
}

// ---------------------------------------------------------------- 2. cached vs full forward

struct Step
{
    vector<float> logits;
    int window_start, pos;
};

// logits of the full forward pass over tokens[start .. start + len), every row
static vector<float> full_forward(Generator &gen, const vector<int> &tokens, int start, int len)
{
    FastGPT &m = gen.model;
    int t = (len + 63) / 64 * 64;
    m.set_seq_len(t);
    vector<int> x(t, 0);
    copy(tokens.begin() + start, tokens.begin() + start + len, x.begin());
    m.x.upload(x.data(), x.size());
    m.y.zero(m.stream);
    DeviceBuffer loss(1);
    m.forward(false, loss.ptr);
    vector<float> out = download_view(m.act("logits"), m.stream);
    m.set_seq_len(m.T_max);
    return out;
}

static void cache_checks(Generator &gen, const vector<int> &prompt)
{
    const ModelConfig &cfg = gen.cfg;
    int max_new = cfg.n_ctx + cfg.n_ctx / 2 + 5;    // rolls over at least once
    printf("\ngreedy, %d new tokens with n_ctx %d: every cached step against the full forward pass\n", max_new,
           cfg.n_ctx);
    vector<Step> steps;
    gen.on_logits = [&](const vector<float> &l, int ws, int pos) { steps.push_back({l, ws, pos}); };
    vector<int> out = gen.generate(prompt, max_new, {0.0f, 0}, 1, Generator::Mode::Cached);
    gen.on_logits = nullptr;
    check(static_cast<int>(out.size()) == static_cast<int>(prompt.size()) + max_new &&
              static_cast<int>(steps.size()) == max_new,
          "one set of logits per new token");

    map<int, int> windows;                  // window start -> tokens it reaches
    for (const Step &s : steps)
    {
        windows[s.window_start] = max(windows[s.window_start], s.pos + 1);
    }
    check(windows.size() >= 2, "the cache rolled over (" + to_string(windows.size()) + " windows)");
    double worst = 0.0;
    int worst_pos = -1, picked_max = 0;
    for (auto &w : windows)
    {
        vector<float> ref = full_forward(gen, out, w.first, w.second);
        for (size_t k = 0; k < steps.size(); k++)
        {
            const Step &s = steps[k];
            if (s.window_start != w.first)
            {
                continue;
            }
            const float *row = ref.data() + static_cast<size_t>(s.pos) * cfg.Vp;
            double scale = 0.0, err = 0.0;
            for (int j = 0; j < cfg.V; j++)
            {
                scale = max(scale, fabs(static_cast<double>(row[j])));
                err = max(err, fabs(static_cast<double>(row[j]) - s.logits[j]));
            }
            if (err / scale > worst)
            {
                worst = err / scale;
                worst_pos = s.pos;
            }
            // the token after this step must be the argmax of these logits
            picked_max += out[prompt.size() + k] == argmax(s.logits, cfg.V);
        }
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "logits match the full forward at every position (worst %.2e at pos %d, tol %.0e)",
             worst, worst_pos, TOL);
    check(worst <= TOL, buf);
    check(picked_max == max_new, "greedy picked the argmax every time");
}

// ---------------------------------------------------------------- 3, 4. graph, seeds, no cache

static void mode_checks(Generator &gen, const vector<int> &prompt)
{
    const ModelConfig &cfg = gen.cfg;
    int max_new = cfg.n_ctx + 20;
    printf("\nsampling (temperature 0.8, top_k 40), %d new tokens\n", max_new);
    SampleConfig sc = {0.8f, 40};
    vector<int> eager = gen.generate(prompt, max_new, sc, 7, Generator::Mode::Cached);
    float ms_eager = gen.last_ms;
    vector<int> graphed = gen.generate(prompt, max_new, sc, 7, Generator::Mode::CachedGraph);
    float ms_graph = gen.last_ms;
    vector<int> again = gen.generate(prompt, max_new, sc, 7, Generator::Mode::CachedGraph);
    vector<int> other = gen.generate(prompt, max_new, sc, 8, Generator::Mode::CachedGraph);
    check(graphed == eager, "the CUDA Graph replay gives the same tokens as plain launches");
    check(again == graphed, "the same seed gives the same sample");
    check(other != graphed, "another seed gives another sample");

    int before = cfg.n_ctx - static_cast<int>(prompt.size());   // the windows are the same until the cache is full
    vector<int> cached = gen.generate(prompt, before, {0.0f, 0}, 1, Generator::Mode::Cached);
    float ms_cached = gen.last_ms;
    vector<int> full = gen.generate(prompt, before, {0.0f, 0}, 1, Generator::Mode::NoCache);
    float ms_full = gen.last_ms;
    int agree = 0;
    while (agree < before && cached[prompt.size() + agree] == full[prompt.size() + agree])
    {
        agree++;
    }
#ifdef FP16
    printf("  no cache vs cache, greedy: the first %d of %d tokens agree (fp16 logits round differently,\n"
           "  so a near tie can go the other way; the logits check above is the real test)\n", agree, before);
#else
    check(agree == before, "greedy without the cache gives the same tokens (" + to_string(agree) + " of " +
                               to_string(before) + ")");
#endif
    printf("  speed here (meaningless on an emulator, real on a GPU): graph %.0f, plain %.0f tok/s;"
           " greedy cached %.0f, no cache %.0f tok/s\n",
           max_new / (ms_graph / 1000.0), max_new / (ms_eager / 1000.0), before / (ms_cached / 1000.0),
           before / (ms_full / 1000.0));
}

// ---------------------------------------------------------------- 5. chat

static DecodeState decode_state(Generator &gen)
{
    DecodeState st;
    CUDA_CHECK(cudaMemcpy(&st, gen.state.ptr, sizeof(st), cudaMemcpyDeviceToHost));
    return st;
}

static int history_at(Generator &gen, int i)
{
    int v = 0;
    CUDA_CHECK(cudaMemcpy(&v, gen.history.ptr + i, sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}

static vector<int> random_tokens(int n, int V, unsigned seed)
{
    mt19937 rng(seed);
    vector<int> t(n);
    for (int &x : t)
    {
        x = static_cast<int>(rng() % 50 == 0 ? rng() % V : rng() % 60);
    }
    return t;
}

static void chat_checks(Generator &gen)
{
    const ModelConfig &cfg = gen.cfg;
    int V = cfg.V, T = cfg.n_ctx;
    const int MAXN = 40;
    int next = V - 1;                       // stands in for <user>
    SampleConfig greedy = {0.0f, 0};
    auto quiet = [](int) {};
    printf("\nchat: one conversation kept in the cache across turns\n");

    // a. a prompt fed one decode step per token continues like a prefilled prompt
    vector<int> p = random_tokens(T / 2, V, 1);
    vector<int> want = gen.generate(p, 10, greedy, 1, Generator::Mode::Cached);
    gen.chat_start(MAXN, 1);
    gen.chat_feed(p);
    vector<int> got = gen.chat_answer(10, greedy, V, next, Generator::Mode::Cached, quiet);
    int agree = 0;
    while (agree < 10 && got[agree] == want[p.size() + agree])
    {
        agree++;
    }
#ifdef FP16
    printf("  fed one step at a time vs prefilled, greedy: the first %d of 10 tokens agree (the prefill's\n"
           "  fp16 logits round differently from the decode GEMV's, so a near tie can go the other way)\n", agree);
#else
    check(agree == 10, "a prompt fed one decode step per token continues like a prefill (" + to_string(agree) +
                           " of 10)");
#endif

    // b. any token >= stop_from ends the answer, and next takes its place
    gen.chat_start(MAXN, 2);
    gen.chat_feed(p);
    int seen = 0;
    vector<int> one = gen.chat_answer(5, {0.8f, 40}, 0, next, Generator::Mode::CachedGraph, [&](int) { seen++; });
    DecodeState st = decode_state(gen);
    check(one.size() == 1 && seen == 1, "the first tag ends the answer, and on_token saw it");
    check(st.token == next && history_at(gen, st.n - 1) == next, "the tag is swapped for <user>, the pending token");

    // c. an answer that runs out gets "\n" and next fed after it
    gen.chat_start(MAXN, 3);
    gen.chat_feed(p);
    DecodeState before = decode_state(gen);
    vector<int> three = gen.chat_answer(3, greedy, V, next, Generator::Mode::Cached, quiet);
    st = decode_state(gen);
    check(three.size() == 3 && st.n == before.n + 5 && st.token == next && history_at(gen, st.n - 2) == '\n',
          "an answer that runs out ends with \"\\n\" and <user> fed after it");

    // d. many turns in the cache against the whole conversation fed at once
    const int TURNS = 8, ANSWER = 25;
    gen.chat_start(MAXN, 4);
    vector<int> all, last_answer;
    for (int k = 0; k < TURNS; k++)
    {
        vector<int> q = random_tokens(13 + 3 * (k % 3), V, 10 + k);
        gen.chat_feed(q);
        all.insert(all.end(), q.begin(), q.end());
        if (k == TURNS - 1)
        {
            break;
        }
        vector<int> a = gen.chat_answer(ANSWER, greedy, V, next, Generator::Mode::CachedGraph, quiet);
        all.insert(all.end(), a.begin(), a.end());
        all.push_back('\n');
        all.push_back(next);
    }
    last_answer = gen.chat_answer(ANSWER, greedy, V, next, Generator::Mode::CachedGraph, quiet);
    int length = gen.chat_length();
    gen.chat_start(MAXN, 5);
    gen.chat_feed(all);
    vector<int> at_once = gen.chat_answer(ANSWER, greedy, V, next, Generator::Mode::CachedGraph, quiet);
    check(length > 4 * T + MAXN + 2, "the conversation (" + to_string(length) + " tokens) outgrew the history " +
                                         "buffer, so it was compacted, and rolled the cache over");
    check(at_once == last_answer, "turn by turn in the cache answers exactly like the whole conversation at once");

    // e. an answer's first token is never a tag, in either mode, greedy or sampled
    gen.chat_start(MAXN, 6);
    gen.chat_feed(p);
    int t0 = gen.chat_answer(1, greedy, V, next, Generator::Mode::Cached, quiet)[0];
    check(t0 > 0, "the greedy first token (" + to_string(t0) + ") leaves ids below it to fall back on");
    for (Generator::Mode mode : {Generator::Mode::Cached, Generator::Mode::CachedGraph})
    {
        gen.chat_start(MAXN, 6);
        gen.chat_feed(p);
        vector<int> a = gen.chat_answer(1, greedy, t0, next, mode, quiet);
        vector<float> l = gen.logits.download();
        int best = static_cast<int>(max_element(l.begin(), l.begin() + t0) - l.begin());
        check(a.size() == 1 && a[0] == best, string(mode == Generator::Mode::Cached ? "plain" : "graph") +
                                                 ": with stop_from at the greedy first token, the answer starts "
                                                 "with the best id below it (" + to_string(a[0]) + ", want " +
                                                 to_string(best) + ")");
    }
    bool sampled_ok = true;
    for (int seed = 0; seed < 20; seed++)
    {
        gen.chat_start(MAXN, 100 + seed);
        gen.chat_feed(p);
        vector<int> a = gen.chat_answer(1, {0.8f, 40}, t0, next, Generator::Mode::CachedGraph, quiet);
        sampled_ok = sampled_ok && a[0] < t0;
    }
    check(sampled_ok, "sampled at temperature 0.8, 20 answers never start with a token >= stop_from");
    gen.chat_start(MAXN, 7);
    gen.chat_feed(p);
    vector<int> two = gen.chat_answer(MAXN, greedy, 1, next, Generator::Mode::CachedGraph, quiet);
    check(two.size() >= 2 && two[0] == 0 && two.back() >= 1,
          "only the first token is held back: with every id but 0 a tag, the answer starts with 0 and ends at a tag");
}

int main(int argc, char *argv[])
{
    string path = argc > 1 ? argv[1] : "data/ref/debug.bin";
    Dump d = read_dump(path);
    ModelConfig cfg = config_from_dump(d);
    printf("test_generate, %s build: %s, L %d H %d C %d n_ctx %d V %d\n", FLOATX_NAME, path.c_str(), cfg.L, cfg.H,
           cfg.C, cfg.n_ctx, cfg.V);
    Generator gen(cfg, params_from_dump(d, cfg));
    sampler_checks(cfg);
    vector<int> prompt = {cfg.V - 1, 10};   // the last id is a tag in a real tokenizer, then "\n"
    cache_checks(gen, prompt);
    mode_checks(gen, prompt);
    chat_checks(gen);
    printf("\n%s\n", g_failed == 0 ? "generation matches" : (to_string(g_failed) + " check(s) failed").c_str());
    return g_failed == 0 ? 0 : 1;
}
