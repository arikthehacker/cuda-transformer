// gpt_fast.cu: the stage 4 forward, backward and optimizer step.
//
// GEMM mapping (W is stored (out, in) like nn.Linear, so y = x @ W^T):
//   forward    out (N, OC) = in (N, IC) @ W^T          gemm_act(ep, false, true, N, OC, IC)
//   backward   din (N, IC) = dout (N, OC) @ W          gemm_act(ep, false, false, N, IC, OC)
//              dW (OC, IC) += dout^T @ in              gemm_grad(true, false, OC, IC, N)
//
// What the forward pass keeps per layer: ln1, qkv, atty (+ lse), ln2, fc, and
// the fp32 residual stream res1, res2. attn_out, mlp_out and gelu go through
// shared scratch buffers: res1 / res2 already hold what backward needs from
// the first two, and backward recomputes gelu from fc.

#include "gpt_fast.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "dump.h"
#include "kernels/gemm_common.cuh"
#include "kernels/kernels.h"

using namespace std;

// ---------------------------------------------------------------- memory

// Hands out slices of one allocation, each rounded up to 8 elements so every
// slice starts 16-byte aligned for fp16 (the GEMMs load 128 bits at a time).
template <class T>
struct Slicer
{
    vector<pair<T **, size_t>> wants;

    void want(T *&p, size_t n)
    {
        p = nullptr;
        if (n > 0)
        {
            wants.push_back({&p, n});
        }
    }

    DeviceArray<T> allocate()
    {
        size_t total = 0;
        for (auto &w : wants)
        {
            total += (w.second + 7) / 8 * 8;
        }
        DeviceArray<T> mem(total);
        size_t offset = 0;
        for (auto &w : wants)
        {
            *w.first = mem.ptr + offset;
            offset += (w.second + 7) / 8 * 8;
        }
        return mem;
    }
};

static size_t find_slot(const vector<ParamSlot> &slots, const string &name)
{
    for (size_t k = 0; k < slots.size(); k++)
    {
        if (slots[k].name == name)
        {
            return k;
        }
    }
    fprintf(stderr, "error: no parameter named %s\n", name.c_str());
    exit(1);
}

FastGPT::FastGPT(const ModelConfig &c, int b, int t, const TrainConfig &tc, float loss_scale, FastOptions opt)
    : cfg(c), B(b), T(t), N(b * t), T_max(t), history_len(opt.history_len)
{
#ifdef FP16
    flash = opt.flash;
#else
    flash = false;                          // flash attention runs on the tensor cores, fp16 only
#endif
    int C = cfg.C, H = cfg.H, D = cfg.D();
    if (C % 64 != 0 || cfg.Vp % 64 != 0 || T % 64 != 0 || T > cfg.n_ctx)
    {
        fprintf(stderr, "error: the fast model needs C, Vp and T to be multiples of 64 and T <= n_ctx "
                        "(C %d, Vp %d, T %d, n_ctx %d)\n", C, cfg.Vp, T, cfg.n_ctx);
        exit(1);
    }
    if (flash && D != 16 && D != 32 && D != 64)
    {
        fprintf(stderr, "error: flash attention backward supports head sizes 16, 32 and 64, not %d\n", D);
        exit(1);
    }
    CUDA_CHECK(cudaStreamCreate(&stream));     // blocking: plain cudaMemcpy calls stay in order with it

    slots = param_layout(cfg);
    size_t n = slots.back().offset + slots.back().count;
    params = DeviceBuffer(n);
    grads = DeviceBuffer(n);
    m = DeviceBuffer(n);
    v = DeviceBuffer(n);
#ifdef FP16
    params_x = DeviceArray<floatX>(n);
    floatX *px = params_x.ptr;
#else
    floatX *px = params.ptr;                // fp32 build: the GEMMs read the master weights
#endif
    vector<TensorInfo> info;
    for (const ParamSlot &s : slots)
    {
        assert(s.offset % 8 == 0 && s.count % 8 == 0);
        info.push_back({static_cast<long long>(s.offset), static_cast<long long>(s.count), s.ndim >= 2 ? 1 : 0});
    }
    tensors = DeviceArray<TensorInfo>(info.size());
    tensors.upload(info.data(), info.size());
    for (int l = 0; l < cfg.L; l++)
    {
        lw.push_back(layer_params(params.ptr, slots, l));
        lg.push_back(layer_params(grads.ptr, slots, l));
        string p = "h." + to_string(l) + ".";
        LayerWeightsX wx;
        wx.attn_w = px + slots[find_slot(slots, p + "attn.c_attn.w")].offset;
        wx.proj_w = px + slots[find_slot(slots, p + "attn.c_proj.w")].offset;
        wx.fc_w = px + slots[find_slot(slots, p + "mlp.c_fc.w")].offset;
        wx.fcproj_w = px + slots[find_slot(slots, p + "mlp.c_proj.w")].offset;
        lx.push_back(wx);
    }
    wte_x = px + slots[find_slot(slots, "wte")].offset;
    wte = param("wte");
    wpe = param("wpe");
    lnf_w = param("ln_f.w");
    lnf_b = param("ln_f.b");
    dwte = grad("wte");
    dwpe = grad("wpe");
    dlnf_w = grad("ln_f.w");
    dlnf_b = grad("ln_f.b");

    x = DeviceInts(N);
    y = DeviceInts(N);
    state = DeviceArray<StepState>(1);
    history = DeviceArray<StepState>(history_len);
    sched = {tc.lr, tc.min_lr, tc.warmup_steps, tc.max_steps, tc.beta1, tc.beta2, tc.weight_decay, tc.grad_clip,
             2000, 1};
#ifndef FP16
    sched.dynamic_scale = 0;                // fp32 cannot overflow here; S stays a fixed power of two
#endif
    beta1 = tc.beta1;
    beta2 = tc.beta2;
    eps = tc.eps;

    // ---- activations
    size_t nc = static_cast<size_t>(N) * C;
    size_t bht = static_cast<size_t>(B) * H * T;
    size_t bhtd = bht * D, bhtt = bht * T;
    size_t unfused = flash ? 0 : 1;         // the size of a buffer only one of the two attentions needs
    size_t fused = flash ? 1 : 0;
    Slicer<floatX> sx;
    Slicer<float> sf;
    layers.resize(cfg.L);
    sf.want(emb, nc);
    for (FastLayer &a : layers)
    {
        sx.want(a.ln1, nc);
        sx.want(a.qkv, 3 * nc);
        sx.want(a.atty, nc);
        sx.want(a.ln2, nc);
        sx.want(a.fc, 4 * nc);
        sf.want(a.ln1_mean, N);
        sf.want(a.ln1_rstd, N);
        sf.want(a.ln2_mean, N);
        sf.want(a.ln2_rstd, N);
        sf.want(a.res1, nc);
        sf.want(a.res2, nc);
        sf.want(a.lse, fused * bht);
        sf.want(a.q, unfused * bhtd);
        sf.want(a.k, unfused * bhtd);
        sf.want(a.v, unfused * bhtd);
        sf.want(a.probs, unfused * bhtt);
    }
    sx.want(lnf, nc);
    sf.want(lnf_mean, N);
    sf.want(lnf_rstd, N);
    sx.want(logits, static_cast<size_t>(N) * cfg.Vp);
    sf.want(losses, N);
    sf.want(partial, NORM_BLOCKS);
    sf.want(colsum, 2 * static_cast<size_t>(COLSUM_SPLITS) * 4 * C);  // the widest column sum is c_fc's bias
    sx.want(proj_out, nc);
    sx.want(gelu, 4 * nc);
    sx.want(dfc, 4 * nc);
    sx.want(dln, nc);
    sx.want(datty, nc);
    sx.want(dqkv, 3 * nc);
    sx.want(dres_h, nc);
    sf.want(dres, nc);
    sf.want(Dq, fused * bht);
    sf.want(dq_acc, fused * nc);
    sf.want(scores, unfused * bhtt);
    sf.want(att_y, unfused * bhtd);
    sf.want(dq, unfused * bhtd);
    sf.want(dk, unfused * bhtd);
    sf.want(dv, unfused * bhtd);
    sf.want(dscores, unfused * bhtt);
    sf.want(dprobs, unfused * bhtt);
    sf.want(dy, unfused * bhtd);
#ifdef FP16
    size_t copies = unfused;                // fp32 copies of qkv, atty, dqkv for the stage 3 attention
#else
    size_t copies = 0;                      // floatX is already fp32
#endif
    sf.want(qkv32, copies * 3 * nc);
    sf.want(atty32, copies * nc);
    sf.want(dqkv32, copies * 3 * nc);
    xmem = sx.allocate();
    fmem = sf.allocate();
    reset(loss_scale);
}

FastGPT::~FastGPT()
{
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void FastGPT::set_seq_len(int t)
{
    if (t <= 0 || t % 64 != 0 || t > T_max)
    {
        fprintf(stderr, "error: sequence length %d must be a multiple of 64 and at most %d\n", t, T_max);
        exit(1);
    }
    T = t;
    N = B * t;
}

float *FastGPT::param(const string &name) const
{
    return params.ptr + slots[find_slot(slots, name)].offset;
}

float *FastGPT::grad(const string &name) const
{
    return grads.ptr + slots[find_slot(slots, name)].offset;
}

void FastGPT::reset(float loss_scale)
{
    m.zero(stream);
    v.zero(stream);
    grads.zero(stream);
    StepState st = initial_state(loss_scale);
    CUDA_CHECK(cudaMemcpyAsync(state.ptr, &st, sizeof(st), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));      // st lives on this stack frame
}

void FastGPT::load_params(const Dump &d)
{
    for (const ParamSlot &s : slots)
    {
        const Record &r = d.at(s.name);
        if (r.count() != s.count)
        {
            fprintf(stderr, "error: %s has %zu values in the dump, the model expects %zu\n", s.name.c_str(),
                    r.count(), s.count);
            exit(1);
        }
        upload(r, params.ptr + s.offset);
    }
#ifdef FP16
    cast_to_x(params_x.ptr, params.ptr, static_cast<long long>(params.count), stream);
#endif
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void FastGPT::set_params(const vector<float> &host)
{
    if (host.size() != params.count)
    {
        fprintf(stderr, "error: %zu parameters given, the model has %zu\n", host.size(), params.count);
        exit(1);
    }
    params.upload(host.data(), host.size());
#ifdef FP16
    cast_to_x(params_x.ptr, params.ptr, static_cast<long long>(params.count), stream);
#endif
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------- forward

void FastGPT::forward(bool train, float *loss_out)
{
    int C = cfg.C, Vp = cfg.Vp;
    float eps_ln = cfg.ln_eps;
    cudaStream_t s = stream;

    encoder_fwd(emb, x.ptr, wte, wpe, B, T, C, s);
    layernorm_fwd(layers[0].ln1, layers[0].ln1_mean, layers[0].ln1_rstd, emb, lw[0].ln1_w, lw[0].ln1_b, N, C, eps_ln,
                  s);
    for (int l = 0; l < cfg.L; l++)
    {
        FastLayer &a = layers[l];
        const LayerParams &w = lw[l];
        const LayerWeightsX &wx = lx[l];
        const float *res_in = l == 0 ? emb : layers[l - 1].res2;

        gemm_act(EP_BIAS, false, true, N, 3 * C, C, a.ln1, C, wx.attn_w, C, a.qkv, 3 * C, w.attn_b, nullptr, nullptr,
                 s);
        attention_forward_layer(a);
        gemm_act(EP_BIAS, false, true, N, C, C, a.atty, C, wx.proj_w, C, proj_out, C, w.proj_b, nullptr, nullptr, s);
        residual_layernorm_fwd(a.res1, a.ln2, a.ln2_mean, a.ln2_rstd, res_in, proj_out, w.ln2_w, w.ln2_b, N, C,
                               eps_ln, s);
        gemm_act(EP_BIAS_GELU, false, true, N, 4 * C, C, a.ln2, C, wx.fc_w, C, a.fc, 4 * C, w.fc_b, gelu, nullptr, s);
        gemm_act(EP_BIAS, false, true, N, C, 4 * C, gelu, 4 * C, wx.fcproj_w, 4 * C, proj_out, C, w.fcproj_b, nullptr,
                 nullptr, s);
        // res2 = res1 + mlp_out, then the next LayerNorm (ln1 of layer l + 1, or ln_f) in the same pass
        bool last = l == cfg.L - 1;
        floatX *next = last ? lnf : layers[l + 1].ln1;
        float *next_mean = last ? lnf_mean : layers[l + 1].ln1_mean;
        float *next_rstd = last ? lnf_rstd : layers[l + 1].ln1_rstd;
        const float *next_w = last ? lnf_w : lw[l + 1].ln1_w;
        const float *next_b = last ? lnf_b : lw[l + 1].ln1_b;
        residual_layernorm_fwd(a.res2, next, next_mean, next_rstd, a.res1, proj_out, next_w, next_b, N, C, eps_ln, s);
    }
    gemm_act(EP_STORE, false, true, N, Vp, C, lnf, C, wte_x, C, logits, Vp, nullptr, nullptr, nullptr, s);
    count_targets(state.ptr, y.ptr, N, s);  // N outside SFT, so the mean is the same as before stage 6
    crossentropy_fused(losses, logits, y.ptr, N, cfg.V, Vp, &state.ptr->loss_scale, &state.ptr->n_valid, train, s);
    mean_loss(loss_out, losses, N, &state.ptr->n_valid, s);
}

// The stage 3 attention needs fp32 in and out: in the fp16 build, qkv goes
// through an fp32 copy and atty comes back through one.
void FastGPT::attention_forward_layer(FastLayer &a)
{
    int H = cfg.H, D = cfg.D();
    cudaStream_t s = stream;
    if (flash)
    {
        flash_attn_fwd(a.atty, a.lse, a.qkv, B, T, H, D, s);
        return;
    }
#ifdef FP16
    long long nc = static_cast<long long>(N) * cfg.C;
    cast_to_f(qkv32, a.qkv, 3 * nc, s);
    attention_forward(atty32, a.q, a.k, a.v, scores, a.probs, att_y, qkv32, B, T, H, D, s);
    cast_to_x(a.atty, atty32, nc, s);
#else
    attention_forward(a.atty, a.q, a.k, a.v, scores, a.probs, att_y, a.qkv, B, T, H, D, s);
#endif
}

// datty -> dqkv (written, not added)
void FastGPT::attention_backward_layer(FastLayer &a)
{
    int H = cfg.H, D = cfg.D();
    cudaStream_t s = stream;
    long long nc = static_cast<long long>(N) * cfg.C;
    if (flash)
    {
        flash_attn_bwd(dqkv, Dq, dq_acc, datty, a.qkv, a.atty, a.lse, B, T, H, D, s);
        return;
    }
    size_t bhtd = static_cast<size_t>(B) * H * T * D, bhtt = static_cast<size_t>(B) * H * T * T;
    for (float *z : {dq, dk, dv, dy})       // the stage 3 backward accumulates into all of these
    {
        CUDA_CHECK(cudaMemsetAsync(z, 0, bhtd * sizeof(float), s));
    }
    for (float *z : {dscores, dprobs})
    {
        CUDA_CHECK(cudaMemsetAsync(z, 0, bhtt * sizeof(float), s));
    }
#ifdef FP16
    CUDA_CHECK(cudaMemsetAsync(dqkv32, 0, 3 * nc * sizeof(float), s));
    cast_to_f(atty32, datty, nc, s);        // atty32 is free again by now: it holds datty in fp32
    attention_backward(dqkv32, dq, dk, dv, dscores, dprobs, dy, atty32, a.q, a.k, a.v, a.probs, B, T, H, D, s);
    cast_to_x(dqkv, dqkv32, 3 * nc, s);
#else
    CUDA_CHECK(cudaMemsetAsync(dqkv, 0, 3 * nc * sizeof(float), s));
    attention_backward(dqkv, dq, dk, dv, dscores, dprobs, dy, datty, a.q, a.k, a.v, a.probs, B, T, H, D, s);
#endif
}

// ---------------------------------------------------------------- backward

void FastGPT::backward()
{
    int C = cfg.C, Vp = cfg.Vp;
    cudaStream_t s = stream;
    long long nc = static_cast<long long>(N) * C;

    // logits = ln_f @ wte^T (tied): dln_f = dlogits @ wte, dwte += dlogits^T @ ln_f
    gemm_act(EP_STORE, false, false, N, C, Vp, logits, Vp, wte_x, C, dln, C, nullptr, nullptr, nullptr, s);
    gemm_grad(true, false, Vp, C, N, logits, Vp, lnf, C, dwte, C, s);
    CUDA_CHECK(cudaMemsetAsync(dres, 0, nc * sizeof(float), s));
    layernorm_bwd(dres, dres_h, dlnf_w, dlnf_b, dln, layers[cfg.L - 1].res2, lnf_w, lnf_mean, lnf_rstd, N, C, colsum,
                  s);

    for (int l = cfg.L - 1; l >= 0; l--)
    {
        FastLayer &a = layers[l];
        const LayerParams &w = lw[l];
        const LayerParams &g = lg[l];
        const LayerWeightsX &wx = lx[l];
        const float *res_in = l == 0 ? emb : layers[l - 1].res2;

        // dres is d res2 here. res2 = res1 + mlp_out sends it to mlp_out unchanged.
        // mlp_out = gelu @ Wfp^T + b
        gelu_fwd(gelu, a.fc, 4 * nc, s);
        gemm_grad(true, false, C, 4 * C, N, dres_h, C, gelu, 4 * C, g.fcproj_w, 4 * C, s);
        bias_bwd(g.fcproj_b, dres_h, N, C, colsum, s);
        // dfc = (dmlp_out @ Wfp) * gelu'(fc), in the epilogue
        gemm_act(EP_GELU_GRAD, false, false, N, 4 * C, C, dres_h, C, wx.fcproj_w, 4 * C, dfc, 4 * C, nullptr, nullptr,
                 a.fc, s);
        // fc = ln2 @ Wfc^T + b
        gemm_grad(true, false, 4 * C, C, N, dfc, 4 * C, a.ln2, C, g.fc_w, C, s);
        bias_bwd(g.fc_b, dfc, N, 4 * C, colsum, s);
        gemm_act(EP_STORE, false, false, N, C, 4 * C, dfc, 4 * C, wx.fc_w, C, dln, C, nullptr, nullptr, nullptr, s);
        // ln2 = LayerNorm(res1): dres becomes d res1 (the residual path plus this)
        layernorm_bwd(dres, dres_h, g.ln2_w, g.ln2_b, dln, a.res1, w.ln2_w, a.ln2_mean, a.ln2_rstd, N, C, colsum, s);

        // res1 = res_in + attn_out, attn_out = atty @ Wproj^T + b
        gemm_grad(true, false, C, C, N, dres_h, C, a.atty, C, g.proj_w, C, s);
        bias_bwd(g.proj_b, dres_h, N, C, colsum, s);
        gemm_act(EP_STORE, false, false, N, C, C, dres_h, C, wx.proj_w, C, datty, C, nullptr, nullptr, nullptr, s);
        attention_backward_layer(a);
        // qkv = ln1 @ Wqkv^T + b
        gemm_grad(true, false, 3 * C, C, N, dqkv, 3 * C, a.ln1, C, g.attn_w, C, s);
        bias_bwd(g.attn_b, dqkv, N, 3 * C, colsum, s);
        gemm_act(EP_STORE, false, false, N, C, 3 * C, dqkv, 3 * C, wx.attn_w, C, dln, C, nullptr, nullptr, nullptr, s);
        // ln1 = LayerNorm(res_in): dres becomes d res_in
        layernorm_bwd(dres, dres_h, g.ln1_w, g.ln1_b, dln, res_in, w.ln1_w, a.ln1_mean, a.ln1_rstd, N, C, colsum, s);
    }
    // emb = wte[tok] + wpe[t]
    encoder_bwd(dwte, dwpe, dres, x.ptr, B, T, C, s);
}

// ---------------------------------------------------------------- step

void FastGPT::optimizer_step()
{
    long long n = static_cast<long long>(params.count);
#ifdef FP16
    floatX *px = params_x.ptr;
#else
    floatX *px = nullptr;
#endif
    unscale_and_norm(grads.ptr, n, state.ptr, partial, stream);
    finalize_step(state.ptr, partial, sched, history.ptr, history_len, stream);
    adamw_multi(params.ptr, px, grads.ptr, m.ptr, v.ptr, n, tensors.ptr, static_cast<int>(tensors.count), state.ptr,
                beta1, beta2, eps, stream);
}

void FastGPT::train_step()
{
    begin_step(state.ptr, stream);
    grads.zero(stream);
    forward(true, &state.ptr->loss);
    backward();
    optimizer_step();
}

StepState FastGPT::read_state()
{
    StepState st;
    CUDA_CHECK(cudaMemcpyAsync(&st, state.ptr, sizeof(st), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return st;
}

// ---------------------------------------------------------------- views (tests)

ActView FastGPT::act(const string &name) const
{
    size_t nc = static_cast<size_t>(N) * cfg.C;
    auto fp = [](const float *p, size_t n) { return ActView{p, false, n}; };
    auto hx = [](const floatX *p, size_t n) { return ActView{p, true, n}; };
    if (name == "emb") return fp(emb, nc);
    if (name == "ln_f") return hx(lnf, nc);
    if (name == "ln_f.mean") return fp(lnf_mean, N);
    if (name == "ln_f.rstd") return fp(lnf_rstd, N);
    if (name == "logits") return hx(logits, static_cast<size_t>(N) * cfg.Vp);
    for (int l = 0; l < cfg.L; l++)
    {
        string p = "h." + to_string(l) + ".";
        if (name.compare(0, p.size(), p) != 0)
        {
            continue;
        }
        const FastLayer &a = layers[l];
        string r = name.substr(p.size());
        if (r == "ln1") return hx(a.ln1, nc);
        if (r == "ln1.mean") return fp(a.ln1_mean, N);
        if (r == "ln1.rstd") return fp(a.ln1_rstd, N);
        if (r == "qkv") return hx(a.qkv, 3 * nc);
        if (r == "atty") return hx(a.atty, nc);
        if (r == "res1") return fp(a.res1, nc);
        if (r == "ln2") return hx(a.ln2, nc);
        if (r == "ln2.mean") return fp(a.ln2_mean, N);
        if (r == "ln2.rstd") return fp(a.ln2_rstd, N);
        if (r == "fc") return hx(a.fc, 4 * nc);
        if (r == "res2") return fp(a.res2, nc);
        if (r == "probs" && a.probs != nullptr) return fp(a.probs, static_cast<size_t>(B) * cfg.H * T * T);
    }
    return ActView{};
}

vector<float> download_view(const ActView &a, cudaStream_t s)
{
    vector<float> out(a.count);
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (a.is_x)
    {
        vector<floatX> h(a.count);
        CUDA_CHECK(cudaMemcpy(h.data(), a.ptr, a.count * sizeof(floatX), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.count; i++)
        {
            out[i] = to_float(h[i]);
        }
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(out.data(), a.ptr, a.count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return out;
}
