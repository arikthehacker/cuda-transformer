// test_gpt.cu: the whole model against the dump.
//
//   ./build/test_gpt data/ref/debug.bin [data/ref/debug_full.bin ...]
//
// For each dump: load_params, forward on the dump's tokens and targets, then
// every forward record by name, in record order. backward, then every grad.*
// record. Then one AdamW step (lr from the header, no clipping, t = 1) and
// every step1.* record.
//
// The AdamW step runs on the dump's grad.* values, not on this model's own
// gradients. Those were already checked above, and at t = 1 the update is
// about lr * g / (|g| + eps): a gradient near 0 that differs from PyTorch by
// float noise can move a parameter a different way, which looks like a bug
// but isn't. Feeding the dump's gradients tests AdamW alone.

#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "config.h"
#include "dump.h"
#include "gpt.h"
#include "kernels/kernels.h"

using namespace std;

static const float FWD = 1e-5f;             // tolerances, relative to the largest |value|
static const float BWD = 1e-4f;
static const float STEP = 1e-4f;
static const float KEY_BIAS_ABS = 1e-6f;

static bool starts_with(const string &s, const string &head)
{
    return s.compare(0, head.size(), head) == 0;
}

static bool ends_with(const string &s, const string &tail)
{
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// step1.h.l.attn.c_attn.b in two parts: the key bias (columns C .. 2C-1) has
// a true gradient of 0, so the dump holds rounding noise there. Absolute
// tolerance for that slice, relative for the rest.
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

static void run(const string &path)
{
    Dump d = read_dump(path);
    ModelConfig cfg = config_from_dump(d);
    int B = static_cast<int>(d.B);
    int T = static_cast<int>(d.T);
    printf("\n==== %s: V %d Vp %d n_ctx %d L %d H %d C %d, batch B %d T %d\n",
           path.c_str(), cfg.V, cfg.Vp, cfg.n_ctx, cfg.L, cfg.H, cfg.C, B, T);

    GPT model(cfg, B, T);
    model.load_params(d);
    unordered_set<string> params;
    for (const ParamSlot &s : model.slots)
    {
        params.insert(s.name);
    }

    DeviceInts tok(static_cast<size_t>(B) * T);
    DeviceInts tgt(static_cast<size_t>(B) * T);
    upload(d.at("tokens"), tok.ptr);
    upload(d.at("targets"), tgt.ptr);

    // ---- forward
    model.forward(tok.ptr, tgt.ptr, B, T);
    table_header("forward, every activation record");
    for (const string &name : d.order)
    {
        bool skip = params.count(name) || name == "tokens" || name == "targets" || starts_with(name, "config.") ||
                    starts_with(name, "grad.") || starts_with(name, "step1.") || name == "grad_norm";
        if (skip)
        {
            continue;
        }
        if (model.act.count(name))
        {
            compare(d, name, model.A(name), FWD);
        }
        else
        {
            table_note(name, "no CUDA buffer has this name");
        }
    }

    // ---- backward. Activation gradients first, in the order backward computes
    // them (loss, logits, ln_f, the last layer ... emb), so the first row over
    // tolerance is where a bug starts. Then the parameter gradients.
    model.zero_grad();
    model.backward(tok.ptr, tgt.ptr, B, T);
    table_header("backward, every grad.* record (activations in backward order, then parameters)");
    vector<string> forward_names;
    for (const string &name : d.order)
    {
        if (model.act.count(name) && !params.count(name))
        {
            forward_names.push_back(name);
        }
    }
    for (size_t k = forward_names.size(); k-- > 0;)
    {
        const string &base = forward_names[k];
        string name = "grad." + base;
        if (!d.has(name))
        {
            continue;
        }
        if (ends_with(base, ".mean") || ends_with(base, ".rstd"))
        {
            table_note(name, "skipped: the CUDA LayerNorm backward goes straight from dout to dx");
        }
        else
        {
            compare(d, name, model.dA(base), BWD);
        }
    }
    for (const string &name : d.order)
    {
        if (starts_with(name, "grad.") && params.count(name.substr(5)))
        {
            compare(d, name, model.grad(name.substr(5)), BWD);
        }
    }
    if (d.has("grad_norm"))
    {
        float norm = grad_norm(model.grads.ptr, static_cast<long long>(model.grads.count));
        DeviceBuffer out(1);
        out.upload(&norm, 1);
        compare(d, "grad_norm", out.ptr, BWD);
    }

    // ---- one AdamW step, on the dump's gradients (see the top of this file)
    for (const ParamSlot &s : model.slots)
    {
        upload(d.at("grad." + s.name), model.grads.ptr + s.offset);
    }
    TrainConfig tc;
    tc.beta1 = d.beta1;
    tc.beta2 = d.beta2;
    tc.eps = d.eps;
    tc.weight_decay = d.weight_decay;
    model.update(d.lr, tc);
    table_header("AdamW step 1, every step1.* record");
    for (const ParamSlot &s : model.slots)
    {
        string name = "step1." + s.name;
        if (ends_with(s.name, "attn.c_attn.b"))
        {
            compare_c_attn_bias(d, name, model.param(s.name), cfg.C);
        }
        else
        {
            compare(d, name, model.param(s.name), STEP);
        }
    }
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
    for (const string &p : paths)
    {
        run(p);
    }
    int bad = table_failures();
    printf("\n%s\n", bad == 0 ? "the whole model matches the dump" : (to_string(bad) + " row(s) over tolerance").c_str());
    return bad == 0 ? 0 : 1;
}
