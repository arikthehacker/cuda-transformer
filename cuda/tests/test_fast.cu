// test_fast.cu: the stage 4 model (FastGPT) against the dump, plus the parts
// a dump cannot check: the loss scaler, skipped steps, the batch loader and
// CUDA Graph replay.
//
//   ./build/test_fast   data/ref/debug.bin [more dumps]    (fp16 build)
//   ./build/test_fast32 data/ref/debug.bin                 (fp32 build: stage 3 tolerances)
//
// Every gradient comes out multiplied by the loss scale S (65,536, a power
// of two, so dividing by it is exact). The AdamW check runs on the dump's
// own gradients, the same as test_gpt (see the top of that file for why).

#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "config.h"
#include "dump.h"
#include "gpt_fast.h"
#include "graph.h"
#include "kernels/fast.h"
#include "weights.h"

using namespace std;

#ifdef FP16
// fp16 keeps 11 significant bits (relative rounding 2^-11, about 5e-4) and
// every layer adds its own rounding, so the bar is a percent of the largest value.
static const float FWD = 1e-2f;
static const float BWD = 3e-2f;
#else
static const float FWD = 1e-5f;             // the stage 3 tolerances
static const float BWD = 1e-4f;
#endif
static const float STEP = 1e-4f;            // AdamW is fp32 math in both builds
static const float KEY_BIAS_ABS = 1e-6f;
static const float SCALE = 65536.0f;

static int g_checks_failed = 0;

static void check(bool ok, const string &what)
{
    printf("  %-60s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        g_checks_failed++;
    }
}

static bool starts_with(const string &s, const string &head)
{
    return s.compare(0, head.size(), head) == 0;
}

static bool ends_with(const string &s, const string &tail)
{
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

static vector<float> scaled(vector<float> v, float mul)
{
    for (float &x : v)
    {
        x *= mul;
    }
    return v;
}

static TrainConfig config_from_header(const Dump &d)
{
    TrainConfig tc;
    tc.beta1 = d.beta1;
    tc.beta2 = d.beta2;
    tc.eps = d.eps;
    tc.weight_decay = d.weight_decay;
    return tc;
}

// lr = the dump's lr from step 0 on, no clipping (what the dump's step1 records assume)
static void flat_schedule(FastGPT &model, const Dump &d)
{
    model.sched.lr = d.lr;
    model.sched.min_lr = d.lr;
    model.sched.warmup_steps = 0;
    model.sched.max_steps = 1 << 30;
    model.sched.grad_clip = 1e30f;
}

static void setup(FastGPT &model, const Dump &d)
{
    model.load_params(d);
    upload(d.at("tokens"), model.x.ptr);
    upload(d.at("targets"), model.y.ptr);
    model.reset(SCALE);
}

static vector<float> param_slice(const FastGPT &model, const ParamSlot &s, const float *base)
{
    return download(base + s.offset, s.count);
}

// step1.h.l.attn.c_attn.b: the key bias has a true gradient of 0 (test_gpt.cu)
static void compare_step1(const Dump &d, const FastGPT &model)
{
    for (const ParamSlot &s : model.slots)
    {
        string name = "step1." + s.name;
        const Record &r = d.at(name);
        vector<float> got = param_slice(model, s, model.params.ptr);
        if (ends_with(s.name, "attn.c_attn.b"))
        {
            int C = model.cfg.C;
            vector<float> ref_qv, got_qv, ref_k, got_k;
            for (int c = 0; c < 3 * C; c++)
            {
                bool key = c >= C && c < 2 * C;
                (key ? ref_k : ref_qv).push_back(r.f[c]);
                (key ? got_k : got_qv).push_back(got[c]);
            }
            compare_host(name + " [q, v]", {2 * C}, ref_qv.data(), got_qv.data(), ref_qv.size(), STEP);
            compare_host(name + " [k] abs", {C}, ref_k.data(), got_k.data(), ref_k.size(), 0.0f, KEY_BIAS_ABS);
        }
        else
        {
            compare_host(name, r.dims, r.f.data(), got.data(), got.size(), STEP);
        }
    }
}

// ---------------------------------------------------------------- against the dump

static void dump_checks(const Dump &d, const ModelConfig &cfg, bool flash)
{
    FastOptions opt;
    opt.flash = flash;
    FastGPT model(cfg, d.B, d.T, config_from_header(d), SCALE, opt);
    string mode = string(FLOATX_NAME) + (model.flash ? ", flash attention" : ", unfused attention");
    setup(model, d);
    unordered_set<string> params;
    for (const ParamSlot &s : model.slots)
    {
        params.insert(s.name);
    }

    // ---- forward (eval: the logits stay logits)
    DeviceBuffer loss(1);
    model.forward(false, loss.ptr);
    table_header("forward, every activation record (" + mode + ")");
    vector<string> forward_names;
    for (const string &name : d.order)
    {
        if (params.count(name) || starts_with(name, "grad") || starts_with(name, "step1.") ||
            starts_with(name, "config.") || name == "tokens" || name == "targets")
        {
            continue;
        }
        forward_names.push_back(name);
        if (name == "loss")
        {
            CUDA_CHECK(cudaStreamSynchronize(model.stream));
            compare(d, "loss", loss.ptr, FWD);
            continue;
        }
        ActView a = model.act(name);
        if (a.ptr == nullptr)
        {
            table_note(name, "not kept (fused away or in a scratch buffer)");
            continue;
        }
        const Record &r = d.at(name);
        vector<float> got = download_view(a, model.stream);
        compare_host(name, r.dims, r.f.data(), got.data(), got.size(), FWD);
    }

    // ---- training forward + backward: every gradient is * S
    begin_step(model.state.ptr, model.stream);
    model.grads.zero(model.stream);
    model.forward(true, &model.state.ptr->loss);
    model.backward();
    table_header("backward (" + mode + "): activation gradients kept by the fast model, then every parameter "
                 "gradient (/ S)");
    vector<float> dl = scaled(download_view(model.act("logits"), model.stream), 1.0f / SCALE);
    const Record &rl = d.at("grad.logits");
    compare_host("grad.logits (written over the logits)", rl.dims, rl.f.data(), dl.data(), dl.size(), BWD);
    vector<float> de = scaled(download(model.dres, static_cast<size_t>(model.N) * cfg.C), 1.0f / SCALE);
    const Record &re = d.at("grad.emb");
    compare_host("grad.emb (the residual gradient at the end)", re.dims, re.f.data(), de.data(), de.size(), BWD);
    for (const string &name : d.order)
    {
        if (starts_with(name, "grad.") && params.count(name.substr(5)))
        {
            const Record &r = d.at(name);
            vector<float> g = scaled(download(model.grad(name.substr(5)), r.count()), 1.0f / SCALE);
            compare_host(name, r.dims, r.f.data(), g.data(), g.size(), BWD);
        }
    }

    // ---- the optimizer step on these gradients: unscale, norm, AdamW
    model.optimizer_step();
    StepState st = model.read_state();
    table_header("loss scaler and norm after one step");
    DeviceBuffer norm(1);
    norm.upload(&st.grad_norm, 1);
    compare(d, "grad_norm", norm.ptr, BWD);
    vector<float> g = download(model.grad("wte"), d.at("grad.wte").count());
    compare_host("grad.wte after unscale (in place, / S)", d.at("grad.wte").dims, d.at("grad.wte").f.data(), g.data(),
                 g.size(), BWD);
    check(st.found_inf == 0 && st.skipped == 0, "finite gradients: not skipped");
    check(st.step == 1 && st.adam_t == 1, "step 1, AdamW t 1");
    check(st.loss_scale == SCALE && st.good_steps == 1, "S unchanged, 1 good step");
    float want = d.at("loss").f[0];
    check(fabsf(st.loss - want) <= FWD * fabsf(want), "the step's mean loss matches the dump");

    // ---- AdamW alone, on the dump's gradients (* S, so the unscale runs too)
    setup(model, d);
    flat_schedule(model, d);
    for (const ParamSlot &s : model.slots)
    {
        vector<float> gs = scaled(d.at("grad." + s.name).f, SCALE);
        CUDA_CHECK(cudaMemcpy(model.grads.ptr + s.offset, gs.data(), gs.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    begin_step(model.state.ptr, model.stream);
    model.optimizer_step();
    CUDA_CHECK(cudaStreamSynchronize(model.stream));
    table_header("AdamW step 1 (one launch for every tensor), every step1.* record");
    compare_step1(d, model);
#ifdef FP16
    vector<float> p32 = model.params.download();
    vector<floatX> px = model.params_x.download();
    size_t off = 0;
    for (size_t i = 0; i < p32.size(); i++)
    {
        if (__half2float(px[i]) != __half2float(__float2half(p32[i])))
        {
            off++;
        }
    }
    check(off == 0, "the fp16 weight copy is exactly fp16(params) after the step");
#endif
}

// ---------------------------------------------------------------- the scaler

static void scaler_checks(const Dump &d, const ModelConfig &cfg)
{
    printf("\nloss scaler: skipped steps and growth\n");
    FastGPT model(cfg, d.B, d.T, config_from_header(d), SCALE);
    setup(model, d);
    flat_schedule(model, d);
    vector<float> before = model.params.download();

    // an inf in one gradient: the whole step must be skipped
    begin_step(model.state.ptr, model.stream);
    model.grads.zero(model.stream);
    model.forward(true, &model.state.ptr->loss);
    model.backward();
    CUDA_CHECK(cudaStreamSynchronize(model.stream));
    float inf = INFINITY;
    CUDA_CHECK(cudaMemcpy(model.grads.ptr + 12345 % model.grads.count, &inf, sizeof(float), cudaMemcpyHostToDevice));
    model.optimizer_step();
    StepState st = model.read_state();
    check(st.found_inf == 1 && st.skipped == 1, "an inf gradient is found and the step is skipped");
    check(st.adam_t == 0 && st.step == 1, "AdamW t stays 0, the schedule step moves on");
    check(model.params.download() == before, "the parameters did not move");
    vector<float> mm = model.m.download();
    bool m_zero = true;
    for (float x : mm)
    {
        m_zero = m_zero && x == 0.0f;
    }
    check(m_zero, "AdamW m did not move");
#ifdef FP16
    check(st.loss_scale == SCALE / 2 && st.good_steps == 0, "S halved");
#else
    check(st.loss_scale == SCALE, "S fixed in the fp32 build");
#endif

    // the next step is normal again
    model.train_step();
    st = model.read_state();
    check(st.found_inf == 0 && st.adam_t == 1 && st.step == 2 && st.skipped == 1, "the next step runs (t 1, step 2)");
    check(model.params.download() != before, "and moves the parameters");

#ifdef FP16
    // growth: after growth_interval finite steps in a row, S doubles
    model.sched.growth_interval = 2;
    model.train_step();
    st = model.read_state();
    check(st.loss_scale == SCALE && st.good_steps == 0, "2 finite steps in a row: S doubles back to 65,536");

    // a real overflow: with S = 2^30, dlogits do not fit in fp16 (max 65,504)
    model.reset(1073741824.0f);
    model.train_step();
    st = model.read_state();
    check(st.found_inf == 1 && st.skipped == 1, "S = 2^30 overflows fp16: found and skipped");
    check(st.loss_scale == 536870912.0f, "S halved to 2^29");
#endif
}

// ---------------------------------------------------------------- graphs

static void graph_checks(const Dump &d, const ModelConfig &cfg)
{
    printf("\nCUDA Graph: 4 captured steps against 4 eager steps\n");
    const int STEPS = 4;
    FastGPT eager(cfg, d.B, d.T, config_from_header(d), SCALE);
    FastGPT graphed(cfg, d.B, d.T, config_from_header(d), SCALE);
    for (FastGPT *m : {&eager, &graphed})
    {
        setup(*m, d);
        flat_schedule(*m, d);
    }
    for (int k = 0; k < STEPS; k++)
    {
        eager.train_step();
    }
    CudaGraph graph;
    graph.capture(graphed.stream, [&]() { graphed.train_step(); });
    StepState st0 = graphed.read_state();
    check(st0.step == 0, "capturing ran nothing");
    for (int k = 0; k < STEPS; k++)
    {
        graph.launch(graphed.stream);
    }
    StepState a = eager.read_state(), b = graphed.read_state();
    check(a.step == STEPS && b.step == STEPS && b.adam_t == STEPS, "both took 4 steps");
    vector<StepState> ha(STEPS), hb(STEPS);
    CUDA_CHECK(cudaMemcpy(ha.data(), eager.history.ptr, STEPS * sizeof(StepState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), graphed.history.ptr, STEPS * sizeof(StepState), cudaMemcpyDeviceToHost));
    printf("  %-6s %-14s %-14s %-12s\n", "step", "loss eager", "loss graph", "grad norm");
    bool same = true, falling = true;
    for (int k = 0; k < STEPS; k++)
    {
        printf("  %-6d %-14.6f %-14.6f %-12.6f\n", hb[k].step, ha[k].loss, hb[k].loss, hb[k].grad_norm);
        same = same && fabsf(ha[k].loss - hb[k].loss) <= 1e-5f * fabsf(ha[k].loss);
        falling = falling && (k == 0 || hb[k].loss < hb[k - 1].loss);
    }
    check(same, "the graph replays give the eager losses");
    check(falling, "the loss falls every step (the same batch, lr from the dump)");
    vector<float> pa = eager.params.download(), pb = graphed.params.download();
    table_header("parameters after 4 steps, graph against eager (atomics may reorder sums on a GPU)");
    compare_host("params", {static_cast<int>(pa.size())}, pa.data(), pb.data(), pa.size(), 1e-5f);
}

// ---------------------------------------------------------------- weights.bin

static void weights_checks(const Dump &d, const ModelConfig &cfg)
{
    printf("\nweights.bin: write, read back, load into a model\n");
    vector<float> params;
    for (const ParamSlot &s : param_layout(cfg))
    {
        const Record &r = d.at(s.name);
        params.insert(params.end(), r.f.begin(), r.f.end());
    }
    const string path = "test_fast_weights.tmp";
    write_weights(path, cfg, params);
    Weights w = read_weights(path);
    check(w.params == params && w.cfg.V == cfg.V && w.cfg.Vp == cfg.Vp && w.cfg.n_ctx == cfg.n_ctx &&
              w.cfg.L == cfg.L && w.cfg.H == cfg.H && w.cfg.C == cfg.C && w.cfg.ln_eps == cfg.ln_eps,
          "fp32: the same sizes and every value bit for bit");
    write_weights(path, cfg, params, true);
    Weights h = read_weights(path);
    bool rounded = h.params.size() == params.size();
    for (size_t i = 0; rounded && i < params.size(); i++)
    {
        rounded = h.params[i] == __half2float(__float2half(params[i]));
    }
    check(rounded, "fp16: every value is the fp16 rounding of the original");
    remove(path.c_str());

    FastGPT model(cfg, d.B, d.T, config_from_header(d), SCALE);
    model.set_params(params);
    upload(d.at("tokens"), model.x.ptr);
    upload(d.at("targets"), model.y.ptr);
    DeviceBuffer loss(1);
    model.forward(false, loss.ptr);
    CUDA_CHECK(cudaStreamSynchronize(model.stream));
    table_header("forward after set_params() from the weights");
    compare(d, "loss", loss.ptr, FWD);
}

// ---------------------------------------------------------------- batches

static void batch_checks()
{
    printf("\nload_batch: every step picks its own batch from GPU memory\n");
    const int B = 3, T = 8, NSTEPS = 5, NTOK = 200;
    vector<int> tokens(NTOK), starts(NSTEPS * B);
    for (int i = 0; i < NTOK; i++)
    {
        tokens[i] = (i * 37) % 101;
    }
    for (int i = 0; i < NSTEPS * B; i++)
    {
        starts[i] = (i * 53) % (NTOK - T - 1);
    }
    DeviceInts dt(NTOK), ds(NSTEPS * B), x(B * T), y(B * T);
    dt.upload(tokens.data(), NTOK);
    ds.upload(starts.data(), NSTEPS * B);
    DeviceArray<StepState> st(1);
    bool ok = true;
    for (int step : {0, 3, 7})              // 7 wraps around to 2
    {
        StepState h = initial_state(1.0f);
        h.step = step;
        st.upload(&h, 1);
        load_batch(x.ptr, y.ptr, dt.ptr, ds.ptr, NSTEPS, st.ptr, B, T, 0);
        vector<int> hx = x.download(), hy = y.download();
        for (int b = 0; b < B; b++)
        {
            int start = starts[(step % NSTEPS) * B + b];
            for (int t = 0; t < T; t++)
            {
                ok = ok && hx[b * T + t] == tokens[start + t] && hy[b * T + t] == tokens[start + t + 1];
            }
        }
    }
    check(ok, "x and y match the host's picks at steps 0, 3 and 7 (wraps)");
}

int main(int argc, char *argv[])
{
    vector<string> paths;
    for (int k = 1; k < argc; k++)
    {
        paths.push_back(argv[k]);
    }
    if (paths.empty())
    {
        paths.push_back("data/ref/debug.bin");
    }
    printf("test_fast, %s build, gemm %s\n", FLOATX_NAME, gemm_impl_name(gemm_impl()));
    for (size_t k = 0; k < paths.size(); k++)
    {
        Dump d = read_dump(paths[k]);
        ModelConfig cfg = config_from_dump(d);
        printf("\n==== %s: L %d H %d C %d B %u T %u\n", paths[k].c_str(), cfg.L, cfg.H, cfg.C, d.B, d.T);
        dump_checks(d, cfg, true);
        if (k == 0)
        {
#ifdef FP16
            dump_checks(d, cfg, false);     // the fp16 build with the stage 3 attention (the before of flash)
#endif
            scaler_checks(d, cfg);
            graph_checks(d, cfg);
            weights_checks(d, cfg);
        }
    }
    batch_checks();
    int bad = table_failures() + g_checks_failed;
    printf("\n%s\n", bad == 0 ? "the fast model matches" : (to_string(bad) + " failure(s)").c_str());
    return bad == 0 ? 0 : 1;
}
