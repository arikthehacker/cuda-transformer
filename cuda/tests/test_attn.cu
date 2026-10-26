// test_attn.cu: fused (flash) attention against the dump and against the
// stage 3 unfused fp32 attention, on the dump's own qkv and grad.atty.
//
//   ./build/test_attn data/ref/debug.bin [data/ref/debug_full.bin]   (fp16 build)
//
// The dump's gradients are around 1e-5, which fp16 cannot hold well, so the
// test scales grad.atty by a power of two first (what loss scaling does in
// training) and divides the results by the same number.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "dump.h"
#include "kernels/fast.h"
#include "kernels/kernels.h"

using namespace std;

static const float TOL = 1e-2f;

static DeviceArray<floatX> to_x(const vector<float> &v, float mul = 1.0f)
{
    vector<floatX> x(v.size());
    for (size_t i = 0; i < v.size(); i++)
    {
        x[i] = from_float<floatX>(v[i] * mul);
    }
    DeviceArray<floatX> d(x.size());
    d.upload(x.data(), x.size());
    return d;
}

static vector<float> from_x(const floatX *p, size_t n, float mul = 1.0f)
{
    vector<floatX> x(n);
    CUDA_CHECK(cudaMemcpy(x.data(), p, n * sizeof(floatX), cudaMemcpyDeviceToHost));
    vector<float> out(n);
    for (size_t i = 0; i < n; i++)
    {
        out[i] = to_float(x[i]) * mul;
    }
    return out;
}

static void run(const string &path)
{
    Dump d = read_dump(path);
    ModelConfig cfg = config_from_dump(d);
    int B = d.B, T = d.T, H = cfg.H, D = cfg.D(), C = cfg.C;
    long long nc = static_cast<long long>(B) * T * C;
    printf("\n==== %s: B %d T %d H %d D %d\n", path.c_str(), B, T, H, D);
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    const Record &qkv = d.at("h.0.qkv");
    const Record &gatty = d.at("grad.h.0.atty");
    float gmax = 0.0f;
    for (float v : gatty.f)
    {
        gmax = fmaxf(gmax, fabsf(v));
    }
    float S = exp2f(floorf(log2f(64.0f / gmax)));       // largest |datty * S| lands near 64

    // ---- fused forward
    DeviceArray<floatX> dqkv_in = to_x(qkv.f);
    DeviceArray<floatX> atty(nc);
    DeviceBuffer L(static_cast<size_t>(B) * H * T);
    flash_attn_fwd(atty.ptr, L.ptr, dqkv_in.ptr, B, T, H, D, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    table_header("flash attention vs the dump (fp16, tolerance 1e-2)");
    vector<float> got = from_x(atty.ptr, nc);
    compare_host("h.0.atty", d.at("h.0.atty").dims, d.at("h.0.atty").f.data(), got.data(), got.size(), TOL);

    // ---- fused backward
    DeviceArray<floatX> datty = to_x(gatty.f, S);
    DeviceArray<floatX> dqkv(nc * 3);
    DeviceBuffer Dq(static_cast<size_t>(B) * H * T), dq_acc(nc);
    flash_attn_bwd(dqkv.ptr, Dq.ptr, dq_acc.ptr, datty.ptr, dqkv_in.ptr, atty.ptr, L.ptr, B, T, H, D, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    got = from_x(dqkv.ptr, nc * 3, 1.0f / S);
    const Record &gq = d.at("grad.h.0.qkv");
    compare_host("grad.h.0.qkv (all of q, k, v)", gq.dims, gq.f.data(), got.data(), got.size(), TOL);
    // each part on its own, so a bug in one of them shows where it is
    const char *part[3] = {"grad.h.0.qkv [q]", "grad.h.0.qkv [k]", "grad.h.0.qkv [v]"};
    for (int p = 0; p < 3; p++)
    {
        vector<float> ref, mine;
        for (long long bt = 0; bt < static_cast<long long>(B) * T; bt++)
        {
            for (int c = 0; c < C; c++)
            {
                ref.push_back(gq.f[bt * 3 * C + p * C + c]);
                mine.push_back(got[bt * 3 * C + p * C + c]);
            }
        }
        compare_host(part[p], {B, T, C}, ref.data(), mine.data(), ref.size(), TOL);
    }

    // ---- the stage 3 unfused fp32 attention on the same inputs, as a second opinion
    table_header("stage 3 unfused fp32 attention vs the same dump (tolerance 1e-4)");
    size_t bhtd = static_cast<size_t>(B) * H * T * D, bhtt = static_cast<size_t>(B) * H * T * T;
    DeviceBuffer fq(nc * 3);
    fq.upload(qkv.f.data(), qkv.f.size());
    DeviceBuffer q(bhtd), k(bhtd), v(bhtd), y(bhtd), sc(bhtt), pr(bhtt), at(nc);
    attention_forward(at.ptr, q.ptr, k.ptr, v.ptr, sc.ptr, pr.ptr, y.ptr, fq.ptr, B, T, H, D);
    compare(d, "h.0.atty", at.ptr, 1e-4f);
    DeviceBuffer dq(bhtd), dk(bhtd), dv(bhtd), dy(bhtd), ds(bhtt), dp(bhtt), dqkv32(nc * 3), dat(nc);
    for (DeviceBuffer *z : {&dq, &dk, &dv, &dy, &ds, &dp, &dqkv32})
    {
        z->zero();
    }
    dat.upload(gatty.f.data(), gatty.f.size());
    attention_backward(dqkv32.ptr, dq.ptr, dk.ptr, dv.ptr, ds.ptr, dp.ptr, dy.ptr, dat.ptr, q.ptr, k.ptr, v.ptr,
                       pr.ptr, B, T, H, D);
    compare(d, "grad.h.0.qkv", dqkv32.ptr, 1e-4f);
    CUDA_CHECK(cudaStreamDestroy(s));
}

int main(int argc, char *argv[])
{
#ifndef FP16
    printf("test_attn needs the fp16 build (flash attention runs on the tensor cores)\n");
    return 0;
#endif
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
    printf("\n%s\n", bad == 0 ? "flash attention matches" : (to_string(bad) + " row(s) over tolerance").c_str());
    return bad == 0 ? 0 : 1;
}
