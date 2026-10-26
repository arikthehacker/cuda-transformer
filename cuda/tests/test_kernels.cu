// test_kernels.cu: each kernel alone, against the dump.
//
// Every kernel gets the dump's inputs, not another kernel's outputs, so one
// bug can't hide behind another. Backward kernels start from the dump's
// grad.* of their output. When a value has two gradient paths, one path is
// isolated by subtraction (see the LayerNorm and wte tests).
//
//   ./build/test_kernels data/ref/debug.bin

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "dump.h"
#include "kernels/kernels.h"
#include "layout.h"

using namespace std;

static const float FWD = 1e-5f;             // tolerances, relative to the largest |value|
static const float BWD = 1e-4f;
static const float STEP = 1e-4f;
static const float KEY_BIAS_ABS = 1e-6f;    // see test_adamw

// ---------------------------------------------------------------- helpers

static DeviceBuffer load(const Dump &d, const string &name)
{
    const Record &r = d.at(name);
    DeviceBuffer b(r.count());
    upload(r, b.ptr);
    return b;
}

static DeviceInts load_ints(const Dump &d, const string &name)
{
    const Record &r = d.at(name);
    DeviceInts b(r.count());
    upload(r, b.ptr);
    return b;
}

static DeviceBuffer zeros(size_t n)
{
    DeviceBuffer b(n);
    b.zero();
    return b;
}

static bool ends_with(const string &s, const string &tail)
{
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// record a minus record b, on the host
static vector<float> difference(const Dump &d, const string &a, const string &b)
{
    const Record &ra = d.at(a);
    const Record &rb = d.at(b);
    vector<float> out(ra.count());
    for (size_t k = 0; k < out.size(); k++)
    {
        out[k] = ra.f[k] - rb.f[k];
    }
    return out;
}

// What encoder_backward should add into dwte: grad.emb scattered by token, in double.
static vector<float> encoder_share_of_wte(const Dump &d, int C)
{
    const Record &demb = d.at("grad.emb");
    const Record &tok = d.at("tokens");
    vector<double> acc(d.at("wte").count(), 0.0);
    for (size_t bt = 0; bt < tok.count(); bt++)
    {
        for (int c = 0; c < C; c++)
        {
            acc[static_cast<size_t>(tok.i[bt]) * C + c] += demb.f[bt * C + c];
        }
    }
    return vector<float>(acc.begin(), acc.end());
}

// ---------------------------------------------------------------- sgemm on small matrices

static void test_sgemm_small()
{
    table_header("sgemm, small matrices: naive vs a double loop on the host, then tiled vs naive");
    struct Case
    {
        int M, N, K, batch;
        float alpha, beta;
    };
    vector<Case> cases = {{1, 1, 1, 1, 2.0f, 0.0f}, {5, 7, 3, 1, 1.0f, 0.0f}, {64, 64, 16, 1, 1.0f, 0.0f},
                          {70, 33, 19, 2, 0.5f, 1.0f}, {130, 65, 129, 3, 1.0f, 1.0f}};
    mt19937 rng(7);
    uniform_real_distribution<float> uni(-1.0f, 1.0f);

    for (const Case &cs : cases)
    {
        for (int ta = 0; ta < 2; ta++)
        {
            for (int tb = 0; tb < 2; tb++)
            {
                int M = cs.M, N = cs.N, K = cs.K;
                // stored shapes, with a few spare columns so lda > the minimum
                int a_rows = ta ? K : M, a_cols = ta ? M : K, lda = a_cols + 3;
                int b_rows = tb ? N : K, b_cols = tb ? K : N, ldb = b_cols + 1;
                int ldc = N + 2;
                long long sA = static_cast<long long>(a_rows) * lda + 5;
                long long sB = static_cast<long long>(b_rows) * ldb + 2;
                long long sC = static_cast<long long>(M) * ldc + 7;
                vector<float> A(sA * cs.batch), Bm(sB * cs.batch), C0(sC * cs.batch);
                for (float &x : A) x = uni(rng);
                for (float &x : Bm) x = uni(rng);
                for (float &x : C0) x = uni(rng);

                // With beta 0, C must never be read: put NaN where outputs go.
                vector<float> expected = C0;
                for (int z = 0; z < cs.batch; z++)
                {
                    for (int i = 0; i < M; i++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            double acc = 0.0;
                            for (int k = 0; k < K; k++)
                            {
                                double av = ta ? A[z * sA + static_cast<long long>(k) * lda + i] : A[z * sA + static_cast<long long>(i) * lda + k];
                                double bv = tb ? Bm[z * sB + static_cast<long long>(j) * ldb + k] : Bm[z * sB + static_cast<long long>(k) * ldb + j];
                                acc += av * bv;
                            }
                            long long ci = z * sC + static_cast<long long>(i) * ldc + j;
                            expected[ci] = static_cast<float>(cs.alpha * acc + (cs.beta != 0.0f ? cs.beta * C0[ci] : 0.0));
                            if (cs.beta == 0.0f)
                            {
                                C0[ci] = NAN;
                            }
                        }
                    }
                }

                DeviceBuffer dA(A.size()), dB(Bm.size()), dCn(C0.size()), dCt(C0.size());
                dA.upload(A.data(), A.size());
                dB.upload(Bm.data(), Bm.size());
                dCn.upload(C0.data(), C0.size());
                dCt.upload(C0.data(), C0.size());
                sgemm_naive(ta, tb, M, N, K, cs.alpha, dA.ptr, lda, sA, dB.ptr, ldb, sB, cs.beta, dCn.ptr, ldc, sC, cs.batch);
                sgemm(ta, tb, M, N, K, cs.alpha, dA.ptr, lda, sA, dB.ptr, ldb, sB, cs.beta, dCt.ptr, ldc, sC, cs.batch);

                char label[96];
                snprintf(label, sizeof(label), "%s%s %dx%dx%d b%d beta %g", ta ? "T" : "N", tb ? "T" : "N",
                         M, N, K, cs.batch, cs.beta);
                vector<int> dims = {static_cast<int>(C0.size())};
                compare_values(string("naive ") + label, dims, expected.data(), expected.size(), dCn.ptr, FWD);
                vector<float> naive = dCn.download();
                compare_values(string("tiled ") + label, dims, naive.data(), naive.size(), dCt.ptr, FWD);
            }
        }
    }
}

// ---------------------------------------------------------------- forward, layer 0

static void test_forward(const Dump &d, const ModelConfig &cfg)
{
    int B = d.B, T = d.T, C = cfg.C, H = cfg.H, D = cfg.D();
    int N = B * T;
    long long nc = static_cast<long long>(N) * C;
    string last = "h." + to_string(cfg.L - 1) + ".";
    table_header("forward kernels (layer 0 unless named), inputs from the dump");

    DeviceInts tok = load_ints(d, "tokens");
    DeviceBuffer wte = load(d, "wte");
    DeviceBuffer wpe = load(d, "wpe");
    DeviceBuffer emb = zeros(nc);
    encoder_forward(emb.ptr, tok.ptr, wte.ptr, wpe.ptr, B, T, C);
    compare(d, "emb", emb.ptr, FWD);

    // ln1
    {
        DeviceBuffer x = load(d, "emb"), w = load(d, "h.0.ln1.w"), b = load(d, "h.0.ln1.b");
        DeviceBuffer out = zeros(nc), mean = zeros(N), rstd = zeros(N);
        layernorm_forward(out.ptr, mean.ptr, rstd.ptr, x.ptr, w.ptr, b.ptr, N, C, cfg.ln_eps);
        compare(d, "h.0.ln1", out.ptr, FWD);
        compare(d, "h.0.ln1.mean", mean.ptr, FWD);
        compare(d, "h.0.ln1.rstd", rstd.ptr, FWD);
    }
    // c_attn
    {
        DeviceBuffer x = load(d, "h.0.ln1"), w = load(d, "h.0.attn.c_attn.w"), b = load(d, "h.0.attn.c_attn.b");
        DeviceBuffer out = zeros(nc * 3);
        linear_forward(out.ptr, x.ptr, w.ptr, b.ptr, N, C, 3 * C);
        compare(d, "h.0.qkv", out.ptr, FWD);
    }
    // attention
    {
        size_t bhtd = static_cast<size_t>(B) * H * T * D;
        size_t bhtt = static_cast<size_t>(B) * H * T * T;
        DeviceBuffer qkv = load(d, "h.0.qkv");
        DeviceBuffer q = zeros(bhtd), k = zeros(bhtd), v = zeros(bhtd), y = zeros(bhtd);
        DeviceBuffer scores = zeros(bhtt), probs = zeros(bhtt), atty = zeros(nc);
        attention_forward(atty.ptr, q.ptr, k.ptr, v.ptr, scores.ptr, probs.ptr, y.ptr, qkv.ptr, B, T, H, D);
        compare(d, "h.0.scores", scores.ptr, FWD);
        compare(d, "h.0.probs", probs.ptr, FWD);
        compare(d, "h.0.atty", atty.ptr, FWD);
    }
    // c_proj, residual
    {
        DeviceBuffer x = load(d, "h.0.atty"), w = load(d, "h.0.attn.c_proj.w"), b = load(d, "h.0.attn.c_proj.b");
        DeviceBuffer out = zeros(nc);
        linear_forward(out.ptr, x.ptr, w.ptr, b.ptr, N, C, C);
        compare(d, "h.0.attn_out", out.ptr, FWD);

        DeviceBuffer a = load(d, "emb"), r = load(d, "h.0.attn_out"), res = zeros(nc);
        residual_forward(res.ptr, a.ptr, r.ptr, nc);
        compare(d, "h.0.res1", res.ptr, FWD);
    }
    // ln2, c_fc, gelu, mlp c_proj, residual
    {
        DeviceBuffer x = load(d, "h.0.res1"), w = load(d, "h.0.ln2.w"), b = load(d, "h.0.ln2.b");
        DeviceBuffer out = zeros(nc), mean = zeros(N), rstd = zeros(N);
        layernorm_forward(out.ptr, mean.ptr, rstd.ptr, x.ptr, w.ptr, b.ptr, N, C, cfg.ln_eps);
        compare(d, "h.0.ln2", out.ptr, FWD);

        DeviceBuffer ln2 = load(d, "h.0.ln2"), fw = load(d, "h.0.mlp.c_fc.w"), fb = load(d, "h.0.mlp.c_fc.b");
        DeviceBuffer fc = zeros(nc * 4);
        linear_forward(fc.ptr, ln2.ptr, fw.ptr, fb.ptr, N, C, 4 * C);
        compare(d, "h.0.fc", fc.ptr, FWD);

        DeviceBuffer fcin = load(d, "h.0.fc"), g = zeros(nc * 4);
        gelu_forward(g.ptr, fcin.ptr, nc * 4);
        compare(d, "h.0.gelu", g.ptr, FWD);

        DeviceBuffer gin = load(d, "h.0.gelu"), pw = load(d, "h.0.mlp.c_proj.w"), pb = load(d, "h.0.mlp.c_proj.b");
        DeviceBuffer mo = zeros(nc);
        linear_forward(mo.ptr, gin.ptr, pw.ptr, pb.ptr, N, 4 * C, C);
        compare(d, "h.0.mlp_out", mo.ptr, FWD);

        DeviceBuffer r1 = load(d, "h.0.res1"), m1 = load(d, "h.0.mlp_out"), res = zeros(nc);
        residual_forward(res.ptr, r1.ptr, m1.ptr, nc);
        compare(d, "h.0.res2", res.ptr, FWD);
    }
    // ln_f, logits, loss
    {
        DeviceBuffer x = load(d, last + "res2"), w = load(d, "ln_f.w"), b = load(d, "ln_f.b");
        DeviceBuffer out = zeros(nc), mean = zeros(N), rstd = zeros(N);
        layernorm_forward(out.ptr, mean.ptr, rstd.ptr, x.ptr, w.ptr, b.ptr, N, C, cfg.ln_eps);
        compare(d, "ln_f", out.ptr, FWD);

        DeviceBuffer lnf = load(d, "ln_f"), logits = zeros(static_cast<size_t>(N) * cfg.Vp);
        linear_forward(logits.ptr, lnf.ptr, wte.ptr, nullptr, N, C, cfg.Vp);
        compare(d, "logits", logits.ptr, FWD);

        DeviceBuffer lg = load(d, "logits"), losses = zeros(N);
        DeviceInts targets = load_ints(d, "targets");
        crossentropy_forward(losses.ptr, lg.ptr, targets.ptr, N, cfg.V, cfg.Vp);
        double sum = 0.0;
        for (float s : losses.download())
        {
            sum += s;
        }
        float loss = static_cast<float>(sum / N);
        DeviceBuffer dl(1);
        dl.upload(&loss, 1);
        compare(d, "loss", dl.ptr, FWD);
    }
}

// ---------------------------------------------------------------- backward, last op first

static void test_backward(const Dump &d, const ModelConfig &cfg)
{
    int B = d.B, T = d.T, C = cfg.C, H = cfg.H, D = cfg.D();
    int N = B * T;
    long long nc = static_cast<long long>(N) * C;
    string last = "h." + to_string(cfg.L - 1) + ".";
    table_header("backward kernels, each starting from the dump's grad.* of its output");

    DeviceInts tok = load_ints(d, "tokens");
    DeviceInts targets = load_ints(d, "targets");
    DeviceBuffer wte = load(d, "wte");

    // crossentropy
    {
        DeviceBuffer lg = load(d, "logits"), dlogits = zeros(static_cast<size_t>(N) * cfg.Vp);
        crossentropy_backward(dlogits.ptr, lg.ptr, targets.ptr, N, cfg.V, cfg.Vp);
        compare(d, "grad.logits", dlogits.ptr, BWD);
    }
    // logits (first half of the tied wte)
    {
        DeviceBuffer dlogits = load(d, "grad.logits"), lnf = load(d, "ln_f");
        DeviceBuffer dlnf = zeros(nc), dwte = zeros(wte.count);
        linear_backward(dlnf.ptr, dwte.ptr, nullptr, dlogits.ptr, lnf.ptr, wte.ptr, N, C, cfg.Vp);
        compare(d, "grad.ln_f", dlnf.ptr, BWD);
        vector<float> enc = encoder_share_of_wte(d, C);
        const Record &gw = d.at("grad.wte");
        vector<float> share(enc.size());
        for (size_t k = 0; k < share.size(); k++)
        {
            share[k] = gw.f[k] - enc[k];
        }
        compare_values("grad.wte, logits share", gw.dims, share.data(), share.size(), dwte.ptr, BWD);
    }
    // ln_f: the last res2 only feeds ln_f, so its whole gradient comes from here
    {
        DeviceBuffer dout = load(d, "grad.ln_f"), x = load(d, last + "res2"), w = load(d, "ln_f.w");
        DeviceBuffer mean = load(d, "ln_f.mean"), rstd = load(d, "ln_f.rstd");
        DeviceBuffer dx = zeros(nc), dw = zeros(C), db = zeros(C);
        layernorm_backward(dx.ptr, dw.ptr, db.ptr, dout.ptr, x.ptr, w.ptr, mean.ptr, rstd.ptr, N, C);
        compare(d, "grad." + last + "res2", dx.ptr, BWD);
        compare(d, "grad.ln_f.w", dw.ptr, BWD);
        compare(d, "grad.ln_f.b", db.ptr, BWD);
    }
    // mlp c_proj
    {
        DeviceBuffer dout = load(d, "grad.h.0.mlp_out"), inp = load(d, "h.0.gelu"), w = load(d, "h.0.mlp.c_proj.w");
        DeviceBuffer dinp = zeros(nc * 4), dw = zeros(w.count), db = zeros(C);
        linear_backward(dinp.ptr, dw.ptr, db.ptr, dout.ptr, inp.ptr, w.ptr, N, 4 * C, C);
        compare(d, "grad.h.0.gelu", dinp.ptr, BWD);
        compare(d, "grad.h.0.mlp.c_proj.w", dw.ptr, BWD);
        compare(d, "grad.h.0.mlp.c_proj.b", db.ptr, BWD);
    }
    // gelu
    {
        DeviceBuffer dout = load(d, "grad.h.0.gelu"), x = load(d, "h.0.fc"), dx = zeros(nc * 4);
        gelu_backward(dx.ptr, x.ptr, dout.ptr, nc * 4);
        compare(d, "grad.h.0.fc", dx.ptr, BWD);
    }
    // c_fc
    {
        DeviceBuffer dout = load(d, "grad.h.0.fc"), inp = load(d, "h.0.ln2"), w = load(d, "h.0.mlp.c_fc.w");
        DeviceBuffer dinp = zeros(nc), dw = zeros(w.count), db = zeros(4 * C);
        linear_backward(dinp.ptr, dw.ptr, db.ptr, dout.ptr, inp.ptr, w.ptr, N, C, 4 * C);
        compare(d, "grad.h.0.ln2", dinp.ptr, BWD);
        compare(d, "grad.h.0.mlp.c_fc.w", dw.ptr, BWD);
        compare(d, "grad.h.0.mlp.c_fc.b", db.ptr, BWD);
    }
    // ln2: res1 feeds ln2 AND the residual, which passes grad.h.0.res2 straight
    // through. So ln2's own dx = grad.h.0.res1 - grad.h.0.res2.
    {
        DeviceBuffer dout = load(d, "grad.h.0.ln2"), x = load(d, "h.0.res1"), w = load(d, "h.0.ln2.w");
        DeviceBuffer mean = load(d, "h.0.ln2.mean"), rstd = load(d, "h.0.ln2.rstd");
        DeviceBuffer dx = zeros(nc), dw = zeros(C), db = zeros(C);
        layernorm_backward(dx.ptr, dw.ptr, db.ptr, dout.ptr, x.ptr, w.ptr, mean.ptr, rstd.ptr, N, C);
        vector<float> own = difference(d, "grad.h.0.res1", "grad.h.0.res2");
        compare_values("ln2 dx = grad.h.0.res1 - grad.h.0.res2", d.at("h.0.res1").dims, own.data(), own.size(), dx.ptr, BWD);
        compare(d, "grad.h.0.ln2.w", dw.ptr, BWD);
        compare(d, "grad.h.0.ln2.b", db.ptr, BWD);
    }
    // attention c_proj
    {
        DeviceBuffer dout = load(d, "grad.h.0.attn_out"), inp = load(d, "h.0.atty"), w = load(d, "h.0.attn.c_proj.w");
        DeviceBuffer dinp = zeros(nc), dw = zeros(w.count), db = zeros(C);
        linear_backward(dinp.ptr, dw.ptr, db.ptr, dout.ptr, inp.ptr, w.ptr, N, C, C);
        compare(d, "grad.h.0.atty", dinp.ptr, BWD);
        compare(d, "grad.h.0.attn.c_proj.w", dw.ptr, BWD);
        compare(d, "grad.h.0.attn.c_proj.b", db.ptr, BWD);
    }
    // attention: q, k, v from the dump's qkv, probs from the dump
    {
        size_t bhtd = static_cast<size_t>(B) * H * T * D;
        size_t bhtt = static_cast<size_t>(B) * H * T * T;
        DeviceBuffer qkv = load(d, "h.0.qkv"), probs = load(d, "h.0.probs"), datty = load(d, "grad.h.0.atty");
        DeviceBuffer q = zeros(bhtd), k = zeros(bhtd), v = zeros(bhtd);
        attention_permute(q.ptr, k.ptr, v.ptr, qkv.ptr, B, T, H, D);
        DeviceBuffer dq = zeros(bhtd), dk = zeros(bhtd), dv = zeros(bhtd), dy = zeros(bhtd);
        DeviceBuffer dscores = zeros(bhtt), dprobs = zeros(bhtt), dqkv = zeros(nc * 3);
        attention_backward(dqkv.ptr, dq.ptr, dk.ptr, dv.ptr, dscores.ptr, dprobs.ptr, dy.ptr, datty.ptr,
                           q.ptr, k.ptr, v.ptr, probs.ptr, B, T, H, D);
        compare(d, "grad.h.0.probs", dprobs.ptr, BWD);
        compare(d, "grad.h.0.scores", dscores.ptr, BWD);
        compare(d, "grad.h.0.qkv", dqkv.ptr, BWD);
    }
    // c_attn
    {
        DeviceBuffer dout = load(d, "grad.h.0.qkv"), inp = load(d, "h.0.ln1"), w = load(d, "h.0.attn.c_attn.w");
        DeviceBuffer dinp = zeros(nc), dw = zeros(w.count), db = zeros(3 * C);
        linear_backward(dinp.ptr, dw.ptr, db.ptr, dout.ptr, inp.ptr, w.ptr, N, C, 3 * C);
        compare(d, "grad.h.0.ln1", dinp.ptr, BWD);
        compare(d, "grad.h.0.attn.c_attn.w", dw.ptr, BWD);
        compare(d, "grad.h.0.attn.c_attn.b", db.ptr, BWD);
    }
    // ln1: emb feeds ln1 AND the residual, which passes grad.h.0.res1 through.
    // So ln1's own dx = grad.emb - grad.h.0.res1.
    {
        DeviceBuffer dout = load(d, "grad.h.0.ln1"), x = load(d, "emb"), w = load(d, "h.0.ln1.w");
        DeviceBuffer mean = load(d, "h.0.ln1.mean"), rstd = load(d, "h.0.ln1.rstd");
        DeviceBuffer dx = zeros(nc), dw = zeros(C), db = zeros(C);
        layernorm_backward(dx.ptr, dw.ptr, db.ptr, dout.ptr, x.ptr, w.ptr, mean.ptr, rstd.ptr, N, C);
        vector<float> own = difference(d, "grad.emb", "grad.h.0.res1");
        compare_values("ln1 dx = grad.emb - grad.h.0.res1", d.at("emb").dims, own.data(), own.size(), dx.ptr, BWD);
        compare(d, "grad.h.0.ln1.w", dw.ptr, BWD);
        compare(d, "grad.h.0.ln1.b", db.ptr, BWD);
    }
    // encoder (second half of the tied wte, and all of wpe)
    {
        DeviceBuffer demb = load(d, "grad.emb");
        DeviceBuffer dwte = zeros(wte.count), dwpe = zeros(d.at("wpe").count());
        encoder_backward(dwte.ptr, dwpe.ptr, demb.ptr, tok.ptr, B, T, C);
        vector<float> enc = encoder_share_of_wte(d, C);
        compare_values("grad.wte, encoder share", d.at("wte").dims, enc.data(), enc.size(), dwte.ptr, BWD);
        compare(d, "grad.wpe", dwpe.ptr, BWD);
    }
}

// ---------------------------------------------------------------- AdamW and the norm

// Compares step1.h.l.attn.c_attn.b in two parts. Columns C .. 2C-1 are the key
// bias: adding the same number to every score in a row does not change the
// softmax, so its true gradient is 0 and the dump holds rounding noise. AdamW
// turns that noise into steps of about lr * g / (|g| + eps), so that slice is
// checked with an absolute tolerance instead.
static void compare_c_attn_bias(const Dump &d, const string &name, const float *device, int C)
{
    const Record &r = d.at(name);
    vector<float> got = download(device, r.count());
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

static void test_adamw(const Dump &d, const ModelConfig &cfg)
{
    table_header("AdamW, t = 1: the dump's params and grad.* -> step1.*");
    for (const ParamSlot &s : param_layout(cfg))
    {
        DeviceBuffer p = load(d, s.name), g = load(d, "grad." + s.name);
        DeviceBuffer m = zeros(s.count), v = zeros(s.count);
        float wd = s.ndim >= 2 ? d.weight_decay : 0.0f;
        adamw_step(p.ptr, g.ptr, m.ptr, v.ptr, static_cast<long long>(s.count), d.lr, d.beta1, d.beta2, d.eps, wd, 1);
        if (ends_with(s.name, "attn.c_attn.b"))
        {
            compare_c_attn_bias(d, "step1." + s.name, p.ptr, cfg.C);
        }
        else
        {
            compare(d, "step1." + s.name, p.ptr, STEP);
        }
    }

    if (d.has("grad_norm"))
    {
        vector<float> all;
        for (const ParamSlot &s : param_layout(cfg))
        {
            const Record &r = d.at("grad." + s.name);
            all.insert(all.end(), r.f.begin(), r.f.end());
        }
        DeviceBuffer g(all.size());
        g.upload(all.data(), all.size());
        float norm = grad_norm(g.ptr, static_cast<long long>(all.size()));
        DeviceBuffer out(1);
        out.upload(&norm, 1);
        compare(d, "grad_norm", out.ptr, FWD);
    }
}

int main(int argc, char *argv[])
{
    string path = argc > 1 ? argv[1] : "data/ref/debug.bin";
    Dump d = read_dump(path);
    ModelConfig cfg = config_from_dump(d);
    printf("%s: V %d Vp %d n_ctx %d L %d H %d C %d, batch B %u T %u\n",
           path.c_str(), cfg.V, cfg.Vp, cfg.n_ctx, cfg.L, cfg.H, cfg.C, d.B, d.T);

    test_sgemm_small();
    test_forward(d, cfg);
    test_backward(d, cfg);
    test_adamw(d, cfg);

    int bad = table_failures();
    printf("\n%s\n", bad == 0 ? "all kernels match the dump" : (to_string(bad) + " row(s) over tolerance").c_str());
    return bad == 0 ? 0 : 1;
}
