// test_gemm.cu: every GEMM implementation against the stage 3 naive sgemm.
//
// Inputs are random numbers that fp16 holds exactly, so their products are
// exact in fp32 and only the order of the sums differs: fp32 outputs must
// match to 1e-5. fp16 outputs are allowed their own rounding (1e-3).
// The first test multiplies the identity by a matrix whose elements are their
// own index, so a wrong fragment mapping shows up as a readable pattern.
//
//   ./build/test_gemm        (fp16 build: mma.sync, wmma, simt; fp32 build: simt)

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "dump.h"
#include "kernels/fast.h"
#include "kernels/gemm_common.cuh"
#include "kernels/kernels.h"

using namespace std;

static float round_x(float v)
{
    return to_float(from_float<floatX>(v));
}

struct Mat
{
    int rows, cols;
    vector<float> h;                // the values (already representable in floatX)
    DeviceArray<floatX> dx;         // floatX copy on the GPU
    DeviceBuffer df;                // fp32 copy on the GPU, for the naive reference

    Mat(int r, int c) : rows(r), cols(c), h(static_cast<size_t>(r) * c), dx(h.size()), df(h.size()) {}

    void upload()
    {
        vector<floatX> x(h.size());
        for (size_t i = 0; i < h.size(); i++)
        {
            x[i] = from_float<floatX>(h[i]);
        }
        dx.upload(x.data(), x.size());
        df.upload(h.data(), h.size());
    }
};

static vector<float> download_x(const floatX *p, size_t n)
{
    vector<floatX> x(n);
    CUDA_CHECK(cudaMemcpy(x.data(), p, n * sizeof(floatX), cudaMemcpyDeviceToHost));
    vector<float> out(n);
    for (size_t i = 0; i < n; i++)
    {
        out[i] = to_float(x[i]);
    }
    return out;
}

// the naive stage 3 sgemm in fp32, C = op(A) op(B)
static vector<float> reference(bool ta, bool tb, int M, int N, int K, Mat &A, Mat &B)
{
    DeviceBuffer C(static_cast<size_t>(M) * N);
    sgemm_naive(ta, tb, M, N, K, 1.0f, A.df.ptr, A.cols, 0, B.df.ptr, B.cols, 0, 0.0f, C.ptr, N, 0, 1);
    return C.download();
}

static mt19937 rng(11);

static void fill_random(Mat &m)
{
    uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (float &v : m.h)
    {
        v = round_x(u(rng));
    }
    m.upload();
}

static string label(const char *impl, const char *what, bool ta, bool tb, int M, int N, int K)
{
    char buf[96];
    snprintf(buf, sizeof(buf), "%-8s %-9s %s%s %dx%dx%d", impl, what, ta ? "T" : "N", tb ? "T" : "N", M, N, K);
    return buf;
}

static void run_impl(GemmImpl impl)
{
    set_gemm_impl(impl);
    const char *name = gemm_impl_name(impl);
    cudaStream_t s = 0;

    // ---- identity x index matrix: C must equal B exactly, element for element
    {
        int M = 64, K = 64, N = 32;
        Mat I(M, K), Bm(K, N);
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                I.h[i * K + k] = i == k ? 1.0f : 0.0f;
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                Bm.h[k * N + n] = static_cast<float>(k * N + n);       // 0 .. 2047, exact in fp16
        I.upload();
        Bm.upload();
        DeviceBuffer C(static_cast<size_t>(M) * N);
        C.zero();
        gemm_grad(false, false, M, N, K, I.dx.ptr, K, Bm.dx.ptr, N, C.ptr, N, s);
        vector<float> got = C.download();
        int bad = 0;
        for (int i = 0; i < M * N && bad < 3; i++)
        {
            if (got[i] != Bm.h[i])
            {
                printf("  identity test: C[%d][%d] = %g, expected %g (index %d)\n", i / N, i % N, got[i], Bm.h[i], i);
                bad++;
            }
        }
        compare_host(label(name, "identity", false, false, M, N, K), {M, N}, Bm.h.data(), got.data(), got.size(), 0.0f);
    }

    // ---- every transpose combination, shapes with ragged edges in M, N and K
    struct Shape
    {
        int M, N, K;
    };
    // the last one has a long K and few tiles, so the weight gradient path splits K over blocks
    vector<Shape> shapes = {{128, 128, 32}, {192, 136, 72}, {64, 320, 96}, {136, 128, 1032}};
    for (const Shape &sh : shapes)
    {
        for (int ta = 0; ta < 2; ta++)
        {
            for (int tb = 0; tb < 2; tb++)
            {
                int M = sh.M, N = sh.N, K = sh.K;
                Mat A(ta ? K : M, ta ? M : K), Bm(tb ? N : K, tb ? K : N);
                fill_random(A);
                fill_random(Bm);
                vector<float> ref = reference(ta, tb, M, N, K, A, Bm);

                // fp32 output, accumulating onto a random C0
                Mat C0(M, N);
                fill_random(C0);
                vector<float> want(ref.size());
                for (size_t i = 0; i < ref.size(); i++)
                {
                    want[i] = C0.h[i] + ref[i];
                }
                C0.df.upload(C0.h.data(), C0.h.size());
                gemm_grad(ta, tb, M, N, K, A.dx.ptr, A.cols, Bm.dx.ptr, Bm.cols, C0.df.ptr, N, s);
                compare_values(label(name, "accum", ta, tb, M, N, K), {M, N}, want.data(), want.size(), C0.df.ptr, 1e-5f);

                // floatX output, plain store
                DeviceArray<floatX> Cx(static_cast<size_t>(M) * N);
                gemm_act(EP_STORE, ta, tb, M, N, K, A.dx.ptr, A.cols, Bm.dx.ptr, Bm.cols, Cx.ptr, N, nullptr, nullptr,
                         nullptr, s);
                vector<float> got = download_x(Cx.ptr, Cx.count);
                compare_host(label(name, "store", ta, tb, M, N, K), {M, N}, ref.data(), got.data(), got.size(),
                             sizeof(floatX) == 2 ? 1e-3f : 1e-5f);
            }
        }
    }

    // ---- the epilogues, on the shapes the model uses them with
    {
        int M = 192, N = 136, K = 72;
        float tol = sizeof(floatX) == 2 ? 2e-3f : 1e-5f;
        // forward linear (NT): + bias, and + bias then GELU
        Mat A(M, K), W(N, K), bias(1, N), aux(M, N);
        fill_random(A);
        fill_random(W);
        fill_random(bias);
        fill_random(aux);
        vector<float> ref = reference(false, true, M, N, K, A, W);
        vector<float> want_bias(ref.size()), want_pre(ref.size()), want_gelu(ref.size());
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                size_t at = static_cast<size_t>(i) * N + j;
                want_bias[at] = ref[at] + bias.h[j];
                float pre = round_x(ref[at] + bias.h[j]);
                want_pre[at] = pre;
                want_gelu[at] = 0.5f * pre * (1.0f + tanhf(0.7978845608028654f * (pre + 0.044715f * pre * pre * pre)));
            }
        }
        DeviceArray<floatX> C(ref.size()), C2(ref.size());
        gemm_act(EP_BIAS, false, true, M, N, K, A.dx.ptr, K, W.dx.ptr, K, C.ptr, N, bias.df.ptr, nullptr, nullptr, s);
        vector<float> got = download_x(C.ptr, C.count);
        compare_host(label(name, "bias", false, true, M, N, K), {M, N}, want_bias.data(), got.data(), got.size(), tol);

        gemm_act(EP_BIAS_GELU, false, true, M, N, K, A.dx.ptr, K, W.dx.ptr, K, C.ptr, N, bias.df.ptr, C2.ptr, nullptr, s);
        got = download_x(C.ptr, C.count);
        compare_host(label(name, "gelu pre", false, true, M, N, K), {M, N}, want_pre.data(), got.data(), got.size(), tol);
        got = download_x(C2.ptr, C2.count);
        compare_host(label(name, "gelu out", false, true, M, N, K), {M, N}, want_gelu.data(), got.data(), got.size(), tol);

        // backward dinp (NN) times gelu'(fc)
        Mat D(M, K), W2(K, N);
        fill_random(D);
        fill_random(W2);
        vector<float> ref2 = reference(false, false, M, N, K, D, W2);
        vector<float> want_gg(ref2.size());
        for (size_t at = 0; at < ref2.size(); at++)
        {
            float x = aux.h[at];
            float th = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
            float d = 0.5f * (1 + th) + 0.5f * x * (1 - th * th) * 0.7978845608028654f * (1 + 3 * 0.044715f * x * x);
            want_gg[at] = ref2[at] * d;
        }
        gemm_act(EP_GELU_GRAD, false, false, M, N, K, D.dx.ptr, K, W2.dx.ptr, N, C.ptr, N, nullptr, nullptr,
                 aux.dx.ptr, s);
        got = download_x(C.ptr, C.count);
        compare_host(label(name, "gelu grad", false, false, M, N, K), {M, N}, want_gg.data(), got.data(), got.size(), tol);
    }
}

int main()
{
    printf("test_gemm, %s build\n", FLOATX_NAME);
    table_header("GEMM implementations vs the naive fp32 sgemm");
#ifdef FP16
    run_impl(GemmImpl::Mma);
    run_impl(GemmImpl::Wmma);
#endif
    run_impl(GemmImpl::Simt);
    int bad = table_failures();
    printf("\n%s\n", bad == 0 ? "every GEMM matches" : (to_string(bad) + " row(s) over tolerance").c_str());
    return bad == 0 ? 0 : 1;
}
