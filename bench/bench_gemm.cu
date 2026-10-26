// bench_gemm.cu: the three GEMMs of stage 4 against cuBLAS, at the exact
// shapes one training step of the model runs (fp16 in, fp32 accumulate).
//
//   ./build/bench_gemm                    N 4096 rows, C 384, Vp 4160, L 6
//   ./build/bench_gemm -N 4096 -C 384 -V 4160 -L 6 -r 100 -o bench/results_gemm.md
//
// For every shape: cuBLAS (cublasGemmEx, CUDA_R_16F in, CUBLAS_COMPUTE_32F)
// and each of mma.sync, wmma and simt do the same work, with the same output
// type: fp16 activations with beta 0, or fp32 weight gradients with beta 1
// (accumulate). Each is checked against cuBLAS first, then timed: 10 warmup
// runs, then -r runs between two cudaEvents, the average.
//
// cuBLAS is column-major and everything here is row-major. A row-major C
// (M x N) is a column-major C^T, and C^T = op(B)^T op(A)^T, so cuBLAS gets B
// first, A second, and m = N, n = M.

#include <getopt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "common.h"
#include "kernels/fast.h"

using namespace std;

#ifndef FP16
#error "bench_gemm needs the fp16 build (make build/bench_gemm passes -DFP16)"
#endif

#define CUBLAS_CHECK(call)                                                                   \
    do                                                                                       \
    {                                                                                        \
        cublasStatus_t st_ = (call);                                                         \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                                    \
        {                                                                                    \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n    %s\n", static_cast<int>(st_), __FILE__, __LINE__, #call); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

struct Shape
{
    string name;
    int M, N, K;
    bool transA, transB;
    bool grad;              // fp32 output, C += (a weight gradient)
    int per_step;           // how many times one training step runs it
};

static vector<Shape> model_shapes(int rows, int C, int Vp, int L)
{
    // forward:  out (rows, OC) = in (rows, IC) @ W^T          transB
    // dinp:     din (rows, IC) = dout (rows, OC) @ W          no transposes
    // dW:       dW (OC, IC) += dout^T (OC, rows) @ in         transA
    struct Layer
    {
        const char *name;
        int IC, OC, count;
    } layers[] = {{"c_attn", C, 3 * C, L}, {"attn c_proj", C, C, L}, {"c_fc", C, 4 * C, L},
                  {"mlp c_proj", 4 * C, C, L}, {"logits", C, Vp, 1}};
    vector<Shape> out;
    for (const Layer &l : layers)
    {
        out.push_back({string(l.name) + " fwd", rows, l.OC, l.IC, false, true, false, l.count});
    }
    for (const Layer &l : layers)
    {
        out.push_back({string(l.name) + " dinp", rows, l.IC, l.OC, false, false, false, l.count});
    }
    for (const Layer &l : layers)
    {
        out.push_back({string(l.name) + " dW", l.OC, l.IC, rows, true, false, true, l.count});
    }
    return out;
}

static void run_ours(GemmImpl impl, const Shape &s, const __half *A, const __half *B, void *C, cudaStream_t st)
{
    set_gemm_impl(impl);
    int lda = s.transA ? s.M : s.K;
    int ldb = s.transB ? s.K : s.N;
    if (s.grad)
    {
        gemm_grad(s.transA, s.transB, s.M, s.N, s.K, A, lda, B, ldb, static_cast<float *>(C), s.N, st);
    }
    else
    {
        gemm_act(0, s.transA, s.transB, s.M, s.N, s.K, A, lda, B, ldb, static_cast<__half *>(C), s.N, nullptr, nullptr,
                 nullptr, st);
    }
}

static void run_cublas(cublasHandle_t h, const Shape &s, const __half *A, const __half *B, void *C)
{
    int lda = s.transA ? s.M : s.K;
    int ldb = s.transB ? s.K : s.N;
    float alpha = 1.0f, beta = s.grad ? 1.0f : 0.0f;
    CUBLAS_CHECK(cublasGemmEx(h, s.transB ? CUBLAS_OP_T : CUBLAS_OP_N, s.transA ? CUBLAS_OP_T : CUBLAS_OP_N, s.N, s.M,
                              s.K, &alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, &beta, C,
                              s.grad ? CUDA_R_32F : CUDA_R_16F, s.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <class F>
static float time_ms(F fn, int reps, cudaStream_t st)
{
    for (int k = 0; k < 10; k++)
    {
        fn();
    }
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, st));
    for (int k = 0; k < reps; k++)
    {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(b, st));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / reps;
}

// max |ours - cuBLAS| / max |cuBLAS|, on one run from zeroed outputs
static double check(GemmImpl impl, cublasHandle_t h, const Shape &s, const __half *A, const __half *B, cudaStream_t st)
{
    size_t n = static_cast<size_t>(s.M) * s.N;
    size_t bytes = n * (s.grad ? sizeof(float) : sizeof(__half));
    void *c1, *c2;
    CUDA_CHECK(cudaMalloc(&c1, bytes));
    CUDA_CHECK(cudaMalloc(&c2, bytes));
    CUDA_CHECK(cudaMemsetAsync(c1, 0, bytes, st));
    CUDA_CHECK(cudaMemsetAsync(c2, 0, bytes, st));
    run_ours(impl, s, A, B, c1, st);
    run_cublas(h, s, A, B, c2);
    CUDA_CHECK(cudaStreamSynchronize(st));
    vector<float> x(n), y(n);
    if (s.grad)
    {
        CUDA_CHECK(cudaMemcpy(x.data(), c1, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y.data(), c2, bytes, cudaMemcpyDeviceToHost));
    }
    else
    {
        vector<__half> hx(n), hy(n);
        CUDA_CHECK(cudaMemcpy(hx.data(), c1, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), c2, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++)
        {
            x[i] = __half2float(hx[i]);
            y[i] = __half2float(hy[i]);
        }
    }
    double err = 0.0, scale = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        err = fmax(err, fabs(static_cast<double>(x[i]) - y[i]));
        scale = fmax(scale, fabs(static_cast<double>(y[i])));
    }
    CUDA_CHECK(cudaFree(c1));
    CUDA_CHECK(cudaFree(c2));
    return scale > 0.0 ? err / scale : err;
}

int main(int argc, char *argv[])
{
    int rows = 4096, C = 384, Vp = 4160, L = 6, reps = 100;
    string md_path = "bench/results_gemm.md";
    int opt = 0;
    while ((opt = getopt(argc, argv, "N:C:V:L:r:o:h")) != -1)
    {
        switch (opt)
        {
        case 'N':
            rows = atoi(optarg);
            break;
        case 'C':
            C = atoi(optarg);
            break;
        case 'V':
            Vp = atoi(optarg);
            break;
        case 'L':
            L = atoi(optarg);
            break;
        case 'r':
            reps = atoi(optarg);
            break;
        case 'o':
            md_path = optarg;
            break;
        default:
            printf("Usage: %s [-N rows] [-C n_embd] [-V padded vocab] [-L layers] [-r reps] [-o results.md]\n", argv[0]);
            return opt == 'h' ? 0 : 1;
        }
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("bench_gemm on %s: rows %d, C %d, Vp %d, L %d, %d timed runs each\n\n", prop.name, rows, C, Vp, L, reps);

    vector<Shape> shapes = model_shapes(rows, C, Vp, L);
    size_t most = 0;
    for (const Shape &s : shapes)
    {
        most = max(most, static_cast<size_t>(s.M) * s.K);
        most = max(most, static_cast<size_t>(s.K) * s.N);
    }
    // inputs: normal(0, 1) halves, like LayerNorm outputs and scaled gradients
    mt19937 rng(1);
    normal_distribution<float> normal(0.0f, 1.0f);
    vector<__half> host(most);
    for (__half &v : host)
    {
        v = __float2half(normal(rng));
    }
    __half *A, *B;
    void *Cbuf;
    CUDA_CHECK(cudaMalloc(&A, most * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&B, most * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(A, host.data(), most * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, host.data(), most * sizeof(__half), cudaMemcpyHostToDevice));
    size_t out_bytes = 0;
    for (const Shape &s : shapes)
    {
        out_bytes = max(out_bytes, static_cast<size_t>(s.M) * s.N * sizeof(float));
    }
    CUDA_CHECK(cudaMalloc(&Cbuf, out_bytes));
    CUDA_CHECK(cudaMemset(Cbuf, 0, out_bytes));

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetStream(h, st));

    const GemmImpl impls[] = {GemmImpl::Mma, GemmImpl::Wmma, GemmImpl::Simt};
    const char *names[] = {"mma.sync", "wmma", "simt"};
    FILE *md = fopen(md_path.c_str(), "w");
    if (md)
    {
        fprintf(md, "| GEMM | M x N x K | cuBLAS | mma.sync | wmma | simt |\n|---|---|---|---|---|---|\n");
    }
    printf("%-18s %-18s %9s %9s %16s %16s %16s\n", "shape", "M x N x K", "cuBLAS", "TFLOPS", "mma.sync", "wmma", "simt");
    printf("%-18s %-18s %9s %9s %16s %16s %16s\n", "", "", "ms", "", "ms (% cuBLAS)", "ms (% cuBLAS)", "ms (% cuBLAS)");
    double step_ms[4] = {0, 0, 0, 0};       // cuBLAS, mma, wmma, simt: GEMM time of one training step
    bool all_ok = true;
    for (const Shape &s : shapes)
    {
        double flops = 2.0 * s.M * s.N * s.K;
        float t_cublas = time_ms([&]() { run_cublas(h, s, A, B, Cbuf); }, reps, st);
        step_ms[0] += t_cublas * s.per_step;
        char dims[48];
        snprintf(dims, sizeof(dims), "%d x %d x %d", s.M, s.N, s.K);
        printf("%-18s %-18s %9.4f %9.1f", s.name.c_str(), dims, t_cublas, flops / (t_cublas * 1e-3) / 1e12);
        if (md)
        {
            fprintf(md, "| %s | %s | %.1f TFLOPS |", s.name.c_str(), dims, flops / (t_cublas * 1e-3) / 1e12);
        }
        for (int k = 0; k < 3; k++)
        {
            double err = check(impls[k], h, s, A, B, st);
            float t = time_ms([&]() { run_ours(impls[k], s, A, B, Cbuf, st); }, reps, st);
            step_ms[k + 1] += t * s.per_step;
            bool ok = err < 1e-2;
            all_ok = all_ok && ok;
            printf(" %8.4f (%4.0f%%)%s", t, 100.0 * t_cublas / t, ok ? " " : "!");
            if (md)
            {
                fprintf(md, " %.1f TFLOPS, %.0f%% |", flops / (t * 1e-3) / 1e12, 100.0 * t_cublas / t);
            }
        }
        printf("\n");
        if (md)
        {
            fprintf(md, "\n");
        }
    }
    printf("\nGEMM time of one training step (every shape times how often a step runs it):\n");
    printf("  cuBLAS %.2f ms", step_ms[0]);
    for (int k = 0; k < 3; k++)
    {
        printf(" | %s %.2f ms (%.0f%% of cuBLAS speed)", names[k], step_ms[k + 1], 100.0 * step_ms[0] / step_ms[k + 1]);
    }
    printf("\n%s\n", all_ok ? "every result matches cuBLAS (max error under 1% of the largest value)"
                             : "! marks a result more than 1% away from cuBLAS");
    if (md)
    {
        fprintf(md, "| one training step | | %.2f ms | %.2f ms (%.0f%%) | %.2f ms (%.0f%%) | %.2f ms (%.0f%%) |\n",
                step_ms[0], step_ms[1], 100.0 * step_ms[0] / step_ms[1], step_ms[2], 100.0 * step_ms[0] / step_ms[2],
                step_ms[3], 100.0 * step_ms[0] / step_ms[3]);
        fclose(md);
        printf("table -> %s\n", md_path.c_str());
    }
    CUBLAS_CHECK(cublasDestroy(h));
    CUDA_CHECK(cudaStreamDestroy(st));
    return all_ok ? 0 : 1;
}
