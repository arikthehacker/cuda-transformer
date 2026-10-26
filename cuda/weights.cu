// weights.cu: read and write weights.bin (see weights.h)

#include "weights.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_fp16.h>

#include "layout.h"

using namespace std;

static const uint32_t MAGIC = 0x57545047u;  // "GPTW"
static const uint32_t VERSION = 1;

struct Header                               // exactly the 48 bytes on disk
{
    uint32_t magic, version;
    uint32_t V, Vp, n_ctx, L, H, C;
    float ln_eps;
    uint32_t dtype;
    uint64_t count;
};
static_assert(sizeof(Header) == 48, "weights.bin header must be 48 bytes");

static size_t expected_count(const ModelConfig &cfg)
{
    vector<ParamSlot> slots = param_layout(cfg);
    return slots.back().offset + slots.back().count;
}

Weights read_weights(const string &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
    {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(1);
    }
    Header h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != MAGIC)
    {
        fprintf(stderr, "error: %s is not a weights.bin file (reference/export.py writes them)\n", path.c_str());
        exit(1);
    }
    if (h.version != VERSION || h.dtype > 1)
    {
        fprintf(stderr, "error: %s has version %u dtype %u, this program reads version 1, dtype 0 or 1\n",
                path.c_str(), h.version, h.dtype);
        exit(1);
    }
    Weights w;
    w.cfg.V = static_cast<int>(h.V);
    w.cfg.Vp = static_cast<int>(h.Vp);
    w.cfg.n_ctx = static_cast<int>(h.n_ctx);
    w.cfg.L = static_cast<int>(h.L);
    w.cfg.H = static_cast<int>(h.H);
    w.cfg.C = static_cast<int>(h.C);
    w.cfg.ln_eps = h.ln_eps;
    if (w.cfg.H <= 0 || w.cfg.C % w.cfg.H != 0 || w.cfg.Vp < w.cfg.V)
    {
        fprintf(stderr, "error: bad sizes in %s (C %d, H %d, V %d, Vp %d)\n", path.c_str(), w.cfg.C, w.cfg.H,
                w.cfg.V, w.cfg.Vp);
        exit(1);
    }
    size_t n = expected_count(w.cfg);
    if (h.count != n)
    {
        fprintf(stderr, "error: %s holds %llu values, a model of its sizes has %zu\n", path.c_str(),
                static_cast<unsigned long long>(h.count), n);
        exit(1);
    }
    w.params.resize(n);
    bool ok;
    if (h.dtype == 0)
    {
        ok = fread(w.params.data(), sizeof(float), n, f) == n;
    }
    else
    {
        vector<__half> half(n);
        ok = fread(half.data(), sizeof(__half), n, f) == n;
        for (size_t i = 0; i < n; i++)
        {
            w.params[i] = __half2float(half[i]);
        }
    }
    unsigned char extra;
    if (!ok || fread(&extra, 1, 1, f) != 0)
    {
        fprintf(stderr, "error: %s is the wrong size for %zu values\n", path.c_str(), n);
        exit(1);
    }
    fclose(f);
    return w;
}

void write_weights(const string &path, const ModelConfig &cfg, const vector<float> &params, bool fp16)
{
    size_t n = expected_count(cfg);
    if (params.size() != n)
    {
        fprintf(stderr, "error: writing %zu values, a model of these sizes has %zu\n", params.size(), n);
        exit(1);
    }
    Header h = {MAGIC, VERSION, static_cast<uint32_t>(cfg.V), static_cast<uint32_t>(cfg.Vp),
                static_cast<uint32_t>(cfg.n_ctx), static_cast<uint32_t>(cfg.L), static_cast<uint32_t>(cfg.H),
                static_cast<uint32_t>(cfg.C), cfg.ln_eps, fp16 ? 1u : 0u, static_cast<uint64_t>(n)};
    string tmp = path + ".tmp";             // a crash never leaves half a file under the real name
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "error: cannot write %s\n", tmp.c_str());
        exit(1);
    }
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
    if (fp16)
    {
        vector<__half> half(n);
        for (size_t i = 0; i < n; i++)
        {
            half[i] = __float2half(params[i]);
        }
        ok = ok && fwrite(half.data(), sizeof(__half), n, f) == n;
    }
    else
    {
        ok = ok && fwrite(params.data(), sizeof(float), n, f) == n;
    }
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0)
    {
        fprintf(stderr, "error: writing %s failed\n", path.c_str());
        exit(1);
    }
}
