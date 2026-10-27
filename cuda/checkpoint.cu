// checkpoint.cu: training checkpoints (checkpoint.h)

#include "checkpoint.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "common.h"
#include "gpt_fast.h"

using namespace std;

static const char MAGIC[4] = {'G', 'P', 'T', 'C'};
static const uint32_t VERSION = 2;

bool same_size(const ModelConfig &a, const ModelConfig &b)
{
    return a.V == b.V && a.Vp == b.Vp && a.n_ctx == b.n_ctx && a.L == b.L && a.H == b.H && a.C == b.C &&
           a.ln_eps == b.ln_eps;
}

// ---------------------------------------------------------------- writing

struct Writer
{
    FILE *f;
    bool ok = true;

    void bytes(const void *p, size_t n)
    {
        ok = ok && fwrite(p, 1, n, f) == n;
    }
    void u32(uint32_t v)
    {
        unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                              static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
        bytes(b, 4);
    }
    void i32(int v)
    {
        u32(static_cast<uint32_t>(v));
    }
    void f32(float v)
    {
        uint32_t u;
        memcpy(&u, &v, 4);
        u32(u);
    }
    void u64(uint64_t v)
    {
        u32(static_cast<uint32_t>(v));
        u32(static_cast<uint32_t>(v >> 32));
    }
    void str(const string &s)
    {
        u32(static_cast<uint32_t>(s.size()));
        bytes(s.data(), s.size());
    }
};

void write_checkpoint(const string &path, const Checkpoint &c)
{
    size_t n = c.params.size();
    if (c.m.size() != n || c.v.size() != n)
    {
        fprintf(stderr, "error: checkpoint arrays of %zu, %zu and %zu values\n", n, c.m.size(), c.v.size());
        exit(1);
    }
    string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "error: cannot write %s\n", tmp.c_str());
        exit(1);
    }
    Writer w{f};
    w.bytes(MAGIC, 4);
    w.u32(VERSION);
    const ModelConfig &m = c.cfg;
    for (int v : {m.V, m.Vp, m.n_ctx, m.L, m.H, m.C})
    {
        w.i32(v);
    }
    w.f32(m.ln_eps);
    const RunInfo &r = c.run;
    for (int v : {r.B, r.max_steps, r.eval_every, r.warmup_steps, r.sft})
    {
        w.i32(v);
    }
    w.f32(r.lr);
    w.f32(r.min_lr);
    w.f32(r.best_val);
    w.i32(r.best_step);
    w.u32(sizeof(StepState));
    w.bytes(&c.state, sizeof(StepState));
    w.str(r.data);
    w.str(r.starts);
    w.i32(r.decay_steps);
    w.f32(r.beta2);
    w.str(r.keep);
    w.str(r.branch);
    w.u64(n);
    w.bytes(c.params.data(), n * sizeof(float));
    w.bytes(c.m.data(), n * sizeof(float));
    w.bytes(c.v.data(), n * sizeof(float));
    bool flushed = fflush(f) == 0;
#ifdef _WIN32
    bool synced = _commit(_fileno(f)) == 0;
#else
    bool synced = fsync(fileno(f)) == 0;
#endif
    bool closed = fclose(f) == 0;
    if (!w.ok || !flushed || !synced || !closed)
    {
        fprintf(stderr, "error: writing %s failed (disk full?), the last checkpoint is still %s\n", tmp.c_str(),
                path.c_str());
        remove(tmp.c_str());
        exit(1);
    }
#ifdef _WIN32
    remove(path.c_str());                   // rename() over an existing file fails on Windows; POSIX replaces it in one step
#endif
    if (rename(tmp.c_str(), path.c_str()) != 0)
    {
        fprintf(stderr, "error: cannot rename %s to %s\n", tmp.c_str(), path.c_str());
        exit(1);
    }
}

// ---------------------------------------------------------------- reading

struct Reader
{
    const vector<unsigned char> &buf;
    size_t at = 0;
    bool ok = true;

    bool bytes(void *p, size_t n)
    {
        if (!ok || buf.size() - at < n)
        {
            ok = false;
            return false;
        }
        memcpy(p, buf.data() + at, n);
        at += n;
        return true;
    }
    uint32_t u32()
    {
        unsigned char b[4] = {};
        bytes(b, 4);
        return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
               (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }
    int i32()
    {
        return static_cast<int>(u32());
    }
    float f32()
    {
        uint32_t u = u32();
        float v;
        memcpy(&v, &u, 4);
        return v;
    }
    uint64_t u64()
    {
        uint64_t lo = u32();
        return lo | (static_cast<uint64_t>(u32()) << 32);
    }
    string str()
    {
        uint32_t n = u32();
        if (!ok || buf.size() - at < n)
        {
            ok = false;
            return "";
        }
        string s(reinterpret_cast<const char *>(buf.data() + at), n);
        at += n;
        return s;
    }
};

bool try_read_checkpoint(const string &path, Checkpoint &c, string &why)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
    {
        why = "cannot open it";
        return false;
    }
    vector<unsigned char> buf;
    unsigned char block[1 << 16];
    size_t got = 0;
    while ((got = fread(block, 1, sizeof(block), f)) > 0)
    {
        buf.insert(buf.end(), block, block + got);
    }
    fclose(f);

    Reader r{buf};
    char magic[4] = {};
    r.bytes(magic, 4);
    if (!r.ok || memcmp(magic, MAGIC, 4) != 0)
    {
        why = "not a GPTC checkpoint";
        return false;
    }
    uint32_t version = r.u32();
    if (version != 1 && version != VERSION)
    {
        why = "version " + to_string(version) + ", this program reads versions 1 and 2";
        return false;
    }
    ModelConfig &m = c.cfg;
    m.V = r.i32();
    m.Vp = r.i32();
    m.n_ctx = r.i32();
    m.L = r.i32();
    m.H = r.i32();
    m.C = r.i32();
    m.ln_eps = r.f32();
    RunInfo &run = c.run;
    run.B = r.i32();
    run.max_steps = r.i32();
    run.eval_every = r.i32();
    run.warmup_steps = r.i32();
    run.sft = r.i32();
    run.lr = r.f32();
    run.min_lr = r.f32();
    run.best_val = r.f32();
    run.best_step = r.i32();
    uint32_t state_bytes = r.u32();
    if (r.ok && state_bytes != sizeof(StepState))
    {
        why = "its StepState is " + to_string(state_bytes) + " bytes, this build's is " +
              to_string(sizeof(StepState)) + " (written by a different version of the code)";
        return false;
    }
    r.bytes(&c.state, sizeof(StepState));
    run.data = r.str();
    run.starts = r.str();
    if (version >= 2)
    {
        run.decay_steps = r.i32();
        run.beta2 = r.f32();
        run.keep = r.str();
        run.branch = r.str();
    }
    uint64_t n = r.u64();
    if (!r.ok || (buf.size() - r.at) != 3 * n * sizeof(float))
    {
        why = "it is cut short or too long (" + to_string(buf.size()) + " bytes)";
        return false;
    }
    for (vector<float> *a : {&c.params, &c.m, &c.v})
    {
        a->resize(n);
        r.bytes(a->data(), n * sizeof(float));
    }
    return r.ok;
}

Checkpoint read_checkpoint(const string &path)
{
    Checkpoint c;
    string why;
    if (!try_read_checkpoint(path, c, why))
    {
        fprintf(stderr, "error: %s: %s\n", path.c_str(), why.c_str());
        exit(1);
    }
    return c;
}

// ---------------------------------------------------------------- the model's side

Checkpoint checkpoint_of(FastGPT &model, const RunInfo &run)
{
    Checkpoint c;
    c.cfg = model.cfg;
    c.run = run;
    CUDA_CHECK(cudaStreamSynchronize(model.stream));
    c.params = model.params.download();
    c.m = model.m.download();
    c.v = model.v.download();
    c.state = model.read_state();
    return c;
}

void restore(FastGPT &model, const Checkpoint &c)
{
    if (!same_size(model.cfg, c.cfg) || c.params.size() != model.params.count)
    {
        fprintf(stderr, "error: the checkpoint is a different model size (L %d H %d C %d n_ctx %d V %d)\n", c.cfg.L,
                c.cfg.H, c.cfg.C, c.cfg.n_ctx, c.cfg.V);
        exit(1);
    }
    model.set_params(c.params);             // also the fp16 copy: fp16(params), what AdamW writes every step
    model.m.upload(c.m.data(), c.m.size());
    model.v.upload(c.v.data(), c.v.size());
    model.grads.zero(model.stream);
    CUDA_CHECK(cudaMemcpyAsync(model.state.ptr, &c.state, sizeof(StepState), cudaMemcpyHostToDevice, model.stream));
    CUDA_CHECK(cudaStreamSynchronize(model.stream));
}
