// gpt.cu: layout, config, and the model's forward, backward and update

#include "gpt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "dump.h"
#include "kernels/kernels.h"

using namespace std;

// ---------------------------------------------------------------- config

ModelConfig config_from_dump(const Dump &d)
{
    ModelConfig cfg;
    cfg.V = static_cast<int>(d.V);
    cfg.Vp = static_cast<int>(d.Vp);
    cfg.n_ctx = static_cast<int>(d.n_ctx);
    cfg.L = static_cast<int>(d.L);
    cfg.H = static_cast<int>(d.H);
    cfg.C = static_cast<int>(d.C);
    if (d.has("config.ln_eps"))
    {
        cfg.ln_eps = d.at("config.ln_eps").f[0];
    }
    else
    {
        fprintf(stderr, "warning: no config.ln_eps record, using %g\n", cfg.ln_eps);
    }
    if (cfg.H <= 0 || cfg.C % cfg.H != 0 || cfg.Vp < cfg.V)
    {
        fprintf(stderr, "error: bad sizes in the dump header (C %d, H %d, V %d, Vp %d)\n", cfg.C, cfg.H, cfg.V, cfg.Vp);
        exit(1);
    }
    return cfg;
}

float lr_at(int step, const TrainConfig &tc)
{
    const double pi = 3.14159265358979323846;
    if (step < tc.warmup_steps)
    {
        return static_cast<float>(static_cast<double>(tc.lr) * (step + 1) / tc.warmup_steps);
    }
    if (step >= tc.max_steps)
    {
        return tc.min_lr;
    }
    double progress = static_cast<double>(step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps);
    return static_cast<float>(tc.min_lr + 0.5 * (1.0 + cos(pi * progress)) * (tc.lr - tc.min_lr));
}

// ---------------------------------------------------------------- layout

vector<ParamSlot> param_layout(const ModelConfig &cfg)
{
    vector<ParamSlot> slots;
    size_t offset = 0;
    size_t C = cfg.C;
    auto add = [&](const string &name, size_t count, int ndim)
    {
        slots.push_back({name, offset, count, ndim});
        offset += count;
    };

    add("wte", static_cast<size_t>(cfg.Vp) * C, 2);
    add("wpe", static_cast<size_t>(cfg.n_ctx) * C, 2);
    for (int l = 0; l < cfg.L; l++)
    {
        string p = "h." + to_string(l) + ".";
        add(p + "ln1.w", C, 1);
        add(p + "ln1.b", C, 1);
        add(p + "attn.c_attn.w", 3 * C * C, 2);
        add(p + "attn.c_attn.b", 3 * C, 1);
        add(p + "attn.c_proj.w", C * C, 2);
        add(p + "attn.c_proj.b", C, 1);
        add(p + "ln2.w", C, 1);
        add(p + "ln2.b", C, 1);
        add(p + "mlp.c_fc.w", 4 * C * C, 2);
        add(p + "mlp.c_fc.b", 4 * C, 1);
        add(p + "mlp.c_proj.w", C * 4 * C, 2);
        add(p + "mlp.c_proj.b", C, 1);
    }
    add("ln_f.w", C, 1);
    add("ln_f.b", C, 1);
    return slots;
}

static size_t slot_offset(const vector<ParamSlot> &slots, const string &name)
{
    for (const ParamSlot &s : slots)
    {
        if (s.name == name)
        {
            return s.offset;
        }
    }
    fprintf(stderr, "error: no parameter named %s\n", name.c_str());
    exit(1);
}

LayerParams layer_params(float *base, const vector<ParamSlot> &slots, int layer)
{
    string p = "h." + to_string(layer) + ".";
    LayerParams lp;
    lp.ln1_w = base + slot_offset(slots, p + "ln1.w");
    lp.ln1_b = base + slot_offset(slots, p + "ln1.b");
    lp.attn_w = base + slot_offset(slots, p + "attn.c_attn.w");
    lp.attn_b = base + slot_offset(slots, p + "attn.c_attn.b");
    lp.proj_w = base + slot_offset(slots, p + "attn.c_proj.w");
    lp.proj_b = base + slot_offset(slots, p + "attn.c_proj.b");
    lp.ln2_w = base + slot_offset(slots, p + "ln2.w");
    lp.ln2_b = base + slot_offset(slots, p + "ln2.b");
    lp.fc_w = base + slot_offset(slots, p + "mlp.c_fc.w");
    lp.fc_b = base + slot_offset(slots, p + "mlp.c_fc.b");
    lp.fcproj_w = base + slot_offset(slots, p + "mlp.c_proj.w");
    lp.fcproj_b = base + slot_offset(slots, p + "mlp.c_proj.b");
    return lp;
}

vector<ActSlot> activation_layout(const ModelConfig &cfg, int B, int T)
{
    vector<ActSlot> slots;
    size_t offset = 0;
    int C = cfg.C;
    int H = cfg.H;
    int D = cfg.D();
    auto add = [&](const string &name, vector<int> dims)
    {
        size_t count = 1;
        for (int d : dims)
        {
            count *= static_cast<size_t>(d);
        }
        slots.push_back({name, dims, offset, count});
        offset += count;
    };

    add("emb", {B, T, C});
    for (int l = 0; l < cfg.L; l++)
    {
        string p = "h." + to_string(l) + ".";
        add(p + "ln1", {B, T, C});
        add(p + "ln1.mean", {B, T});
        add(p + "ln1.rstd", {B, T});
        add(p + "qkv", {B, T, 3 * C});
        add(p + "q", {B, H, T, D});
        add(p + "k", {B, H, T, D});
        add(p + "v", {B, H, T, D});
        add(p + "scores", {B, H, T, T});
        add(p + "probs", {B, H, T, T});
        add(p + "y", {B, H, T, D});
        add(p + "atty", {B, T, C});
        add(p + "attn_out", {B, T, C});
        add(p + "res1", {B, T, C});
        add(p + "ln2", {B, T, C});
        add(p + "ln2.mean", {B, T});
        add(p + "ln2.rstd", {B, T});
        add(p + "fc", {B, T, 4 * C});
        add(p + "gelu", {B, T, 4 * C});
        add(p + "mlp_out", {B, T, C});
        add(p + "res2", {B, T, C});
    }
    add("ln_f", {B, T, C});
    add("ln_f.mean", {B, T});
    add("ln_f.rstd", {B, T});
    add("logits", {B, T, cfg.Vp});
    add("losses", {B, T});
    add("loss", {});
    return slots;
}

static size_t total_count(const vector<ParamSlot> &slots)
{
    return slots.empty() ? 0 : slots.back().offset + slots.back().count;
}

static size_t total_count(const vector<ActSlot> &slots)
{
    return slots.empty() ? 0 : slots.back().offset + slots.back().count;
}

// ---------------------------------------------------------------- GPT

GPT::GPT(const ModelConfig &config, int B, int T)
    : cfg(config), max_B(B), max_T(T),
      slots(param_layout(config)),
      act_slots(activation_layout(config, B, T))
{
    assert(B > 0 && T > 0 && T <= cfg.n_ctx);
    size_t n = total_count(slots);
    params = DeviceBuffer(n);
    grads = DeviceBuffer(n);
    m = DeviceBuffer(n);
    v = DeviceBuffer(n);
    params.zero();
    grads.zero();
    m.zero();
    v.zero();
    for (int l = 0; l < cfg.L; l++)
    {
        lw.push_back(layer_params(params.ptr, slots, l));
        lg.push_back(layer_params(grads.ptr, slots, l));
    }

    size_t na = total_count(act_slots);
    acts = DeviceBuffer(na);
    dacts = DeviceBuffer(na);
    acts.zero();
    dacts.zero();
    for (const ActSlot &s : act_slots)
    {
        act[s.name] = acts.ptr + s.offset;
        dact[s.name] = dacts.ptr + s.offset;
    }
}

float *GPT::param(const string &name) const
{
    return params.ptr + slot_offset(slots, name);
}

float *GPT::grad(const string &name) const
{
    return grads.ptr + slot_offset(slots, name);
}

float *GPT::A(const string &name) const
{
    auto it = act.find(name);
    if (it == act.end())
    {
        fprintf(stderr, "error: no activation named %s\n", name.c_str());
        exit(1);
    }
    return it->second;
}

float *GPT::dA(const string &name) const
{
    auto it = dact.find(name);
    if (it == dact.end())
    {
        fprintf(stderr, "error: no activation gradient named %s\n", name.c_str());
        exit(1);
    }
    return it->second;
}

void GPT::load_params(const Dump &d)
{
    for (const ParamSlot &s : slots)
    {
        const Record &r = d.at(s.name);
        if (r.count() != s.count)
        {
            fprintf(stderr, "error: %s has %zu values in the dump, the model expects %zu\n",
                    s.name.c_str(), r.count(), s.count);
            exit(1);
        }
        upload(r, params.ptr + s.offset);
    }
    grads.zero();
    m.zero();
    v.zero();
    step = 0;
}

float GPT::forward(const int *tok, const int *targets, int B, int T)
{
    assert(B <= max_B && T <= max_T);
    int N = B * T;
    int C = cfg.C;
    int H = cfg.H;
    int D = cfg.D();

    encoder_forward(A("emb"), tok, param("wte"), param("wpe"), B, T, C);
    const float *x = A("emb");
    for (int l = 0; l < cfg.L; l++)
    {
        const LayerParams &w = lw[l];
        string p = "h." + to_string(l) + ".";

        layernorm_forward(A(p + "ln1"), A(p + "ln1.mean"), A(p + "ln1.rstd"), x, w.ln1_w, w.ln1_b, N, C, cfg.ln_eps);
        linear_forward(A(p + "qkv"), A(p + "ln1"), w.attn_w, w.attn_b, N, C, 3 * C);
        attention_forward(A(p + "atty"), A(p + "q"), A(p + "k"), A(p + "v"), A(p + "scores"), A(p + "probs"),
                          A(p + "y"), A(p + "qkv"), B, T, H, D);
        linear_forward(A(p + "attn_out"), A(p + "atty"), w.proj_w, w.proj_b, N, C, C);
        residual_forward(A(p + "res1"), x, A(p + "attn_out"), static_cast<long long>(N) * C);

        layernorm_forward(A(p + "ln2"), A(p + "ln2.mean"), A(p + "ln2.rstd"), A(p + "res1"), w.ln2_w, w.ln2_b,
                          N, C, cfg.ln_eps);
        linear_forward(A(p + "fc"), A(p + "ln2"), w.fc_w, w.fc_b, N, C, 4 * C);
        gelu_forward(A(p + "gelu"), A(p + "fc"), static_cast<long long>(N) * 4 * C);
        linear_forward(A(p + "mlp_out"), A(p + "gelu"), w.fcproj_w, w.fcproj_b, N, 4 * C, C);
        residual_forward(A(p + "res2"), A(p + "res1"), A(p + "mlp_out"), static_cast<long long>(N) * C);
        x = A(p + "res2");
    }

    layernorm_forward(A("ln_f"), A("ln_f.mean"), A("ln_f.rstd"), x, param("ln_f.w"), param("ln_f.b"),
                      N, C, cfg.ln_eps);
    // logits (N, Vp) = ln_f (N, C) @ wte^T, tied weights, no bias
    linear_forward(A("logits"), A("ln_f"), param("wte"), nullptr, N, C, cfg.Vp);

    if (targets == nullptr)
    {
        return 0.0f;
    }
    crossentropy_forward(A("losses"), A("logits"), targets, N, cfg.V, cfg.Vp);
    vector<float> losses = download(A("losses"), N);
    double sum = 0.0;
    for (float s : losses)
    {
        sum += s;
    }
    float loss = static_cast<float>(sum / N);
    CUDA_CHECK(cudaMemcpy(A("loss"), &loss, sizeof(float), cudaMemcpyHostToDevice));
    return loss;
}

void GPT::zero_grad()
{
    grads.zero();
    dacts.zero();
}

void GPT::backward(const int *tok, const int *targets, int B, int T)
{
    assert(B <= max_B && T <= max_T);
    int N = B * T;
    int C = cfg.C;
    int H = cfg.H;
    int D = cfg.D();
    string last = "h." + to_string(cfg.L - 1) + ".";

    float one = 1.0f;                       // d loss / d loss, so grad.loss compares like the rest
    CUDA_CHECK(cudaMemcpy(dA("loss"), &one, sizeof(float), cudaMemcpyHostToDevice));

    crossentropy_backward(dA("logits"), A("logits"), targets, N, cfg.V, cfg.Vp);
    // dln_f += dlogits @ wte,  dwte += dlogits^T @ ln_f   (first half of the tied wte)
    linear_backward(dA("ln_f"), grad("wte"), nullptr, dA("logits"), A("ln_f"), param("wte"), N, C, cfg.Vp);
    layernorm_backward(dA(last + "res2"), grad("ln_f.w"), grad("ln_f.b"), dA("ln_f"), A(last + "res2"),
                       param("ln_f.w"), A("ln_f.mean"), A("ln_f.rstd"), N, C);

    long long nc = static_cast<long long>(N) * C;
    for (int l = cfg.L - 1; l >= 0; l--)
    {
        const LayerParams &w = lw[l];
        const LayerParams &g = lg[l];
        string p = "h." + to_string(l) + ".";
        string prev = "h." + to_string(l - 1) + ".";
        const float *x = l == 0 ? A("emb") : A(prev + "res2");
        float *dx = l == 0 ? dA("emb") : dA(prev + "res2");

        // res2 = res1 + mlp_out: the gradient goes to both, unchanged
        accumulate(dA(p + "mlp_out"), dA(p + "res2"), nc);
        accumulate(dA(p + "res1"), dA(p + "res2"), nc);
        linear_backward(dA(p + "gelu"), g.fcproj_w, g.fcproj_b, dA(p + "mlp_out"), A(p + "gelu"), w.fcproj_w,
                        N, 4 * C, C);
        gelu_backward(dA(p + "fc"), A(p + "fc"), dA(p + "gelu"), static_cast<long long>(N) * 4 * C);
        linear_backward(dA(p + "ln2"), g.fc_w, g.fc_b, dA(p + "fc"), A(p + "ln2"), w.fc_w, N, C, 4 * C);
        layernorm_backward(dA(p + "res1"), g.ln2_w, g.ln2_b, dA(p + "ln2"), A(p + "res1"), w.ln2_w,
                           A(p + "ln2.mean"), A(p + "ln2.rstd"), N, C);

        // res1 = x + attn_out
        accumulate(dA(p + "attn_out"), dA(p + "res1"), nc);
        accumulate(dx, dA(p + "res1"), nc);
        linear_backward(dA(p + "atty"), g.proj_w, g.proj_b, dA(p + "attn_out"), A(p + "atty"), w.proj_w, N, C, C);
        attention_backward(dA(p + "qkv"), dA(p + "q"), dA(p + "k"), dA(p + "v"), dA(p + "scores"), dA(p + "probs"),
                           dA(p + "y"), dA(p + "atty"), A(p + "q"), A(p + "k"), A(p + "v"), A(p + "probs"),
                           B, T, H, D);
        linear_backward(dA(p + "ln1"), g.attn_w, g.attn_b, dA(p + "qkv"), A(p + "ln1"), w.attn_w, N, C, 3 * C);
        layernorm_backward(dx, g.ln1_w, g.ln1_b, dA(p + "ln1"), x, w.ln1_w, A(p + "ln1.mean"), A(p + "ln1.rstd"),
                           N, C);
    }

    // second half of the tied wte, and wpe
    encoder_backward(grad("wte"), grad("wpe"), dA("emb"), tok, B, T, C);
}

float GPT::clip(float max_norm)
{
    long long n = static_cast<long long>(grads.count);
    float norm = grad_norm(grads.ptr, n);
    // same as torch.nn.utils.clip_grad_norm_: coef = max_norm / (norm + 1e-6), at most 1
    float coef = max_norm / (norm + 1e-6f);
    if (coef < 1.0f)
    {
        scale(grads.ptr, coef, n);
    }
    return norm;
}

void GPT::update(float lr, const TrainConfig &tc)
{
    step++;
    for (const ParamSlot &s : slots)
    {
        float wd = s.ndim >= 2 ? tc.weight_decay : 0.0f;    // by dimensions, not by name
        adamw_step(params.ptr + s.offset, grads.ptr + s.offset, m.ptr + s.offset, v.ptr + s.offset,
                   static_cast<long long>(s.count), lr, tc.beta1, tc.beta2, tc.eps, wd, step);
    }
}
