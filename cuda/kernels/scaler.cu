// scaler.cu: dynamic loss scaling, the gradient norm and the learning rate,
// all decided on the GPU (no host sync, so a CUDA Graph can replay the step).
//
// Same rules as torch.amp.GradScaler with its defaults:
//   S starts at 65,536. A step whose gradients hold an inf or NaN is skipped
//   (AdamW does nothing, its t does not move) and S is halved. After 2,000
//   finite steps in a row S doubles. S is a power of two the whole time, so
//   multiplying by S and by 1/S is exact in fp32.

#include <cassert>
#include <cmath>

#include "common.h"
#include "fast.h"
#include "reduce.cuh"

using namespace std;

static const int BLOCK = 256;

StepState initial_state(float scale)
{
    StepState st = {};
    st.loss_scale = scale;
    st.inv_scale = 1.0f / scale;
    st.clip_coef = 1.0f;
    st.decay = 1.0f;
    return st;
}

__global__ void begin_step_kernel(StepState *st)
{
    st->found_inf = 0;
    st->inv_scale = 1.0f / st->loss_scale;
}

// 4 values per thread, grid stride. The flag is a plain store: every thread
// that sees a bad value writes the same 1, so the order does not matter.
__global__ void unscale_norm_kernel(float *grads, long long n4, StepState *st, float *partial)
{
    float inv = st->inv_scale;
    float sum = 0.0f;
    bool bad = false;
    float4 *g4 = reinterpret_cast<float4 *>(grads);
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride)
    {
        float4 g = g4[i];
        g.x *= inv;
        g.y *= inv;
        g.z *= inv;
        g.w *= inv;
        g4[i] = g;
        bad = bad || !isfinite(g.x) || !isfinite(g.y) || !isfinite(g.z) || !isfinite(g.w);
        sum += g.x * g.x + g.y * g.y + g.z * g.z + g.w * g.w;
    }
    if (bad)
    {
        st->found_inf = 1;
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = sum;
    }
}

// The same schedule as lr_at() in gpt.cu and reference/train.py, in double
__device__ double lr_schedule(int step, const ScheduleConfig &cfg)
{
    if (step < cfg.warmup_steps)
    {
        return static_cast<double>(cfg.lr) * (step + 1) / cfg.warmup_steps;
    }
    if (step >= cfg.max_steps)
    {
        return cfg.min_lr;
    }
    int span = cfg.max_steps - cfg.warmup_steps;
    double progress = static_cast<double>(step - cfg.warmup_steps) / (span > 1 ? span : 1);
    double span_lr = static_cast<double>(cfg.lr) - cfg.min_lr;
    return cfg.min_lr + 0.5 * (1.0 + cos(3.14159265358979323846 * progress)) * span_lr;
}

// One block of NORM_BLOCKS threads. The partial sums are added in double in
// a fixed tree, so the norm comes out the same every run.
__global__ void __launch_bounds__(NORM_BLOCKS) finalize_kernel(StepState *st, const float *partial, ScheduleConfig cfg,
                                                               StepState *history, int history_len)
{
    __shared__ double tree[NORM_BLOCKS];
    int tid = threadIdx.x;
    tree[tid] = partial[tid];
    __syncthreads();
    for (int half = NORM_BLOCKS / 2; half > 0; half /= 2)
    {
        if (tid < half)
        {
            tree[tid] += tree[tid + half];
        }
        __syncthreads();
    }
    if (tid != 0)
    {
        return;
    }

    StepState s = *st;
    float norm = static_cast<float>(sqrt(tree[0]));
    s.grad_norm = norm;
    s.clip_coef = fminf(1.0f, cfg.grad_clip / (norm + 1e-6f));     // torch: clamp(max_norm / (norm + 1e-6), max=1)
    double lr = lr_schedule(s.step, cfg);
    s.lr = static_cast<float>(lr);
    if (s.found_inf)
    {
        s.skipped++;
        s.good_steps = 0;
        if (cfg.dynamic_scale)
        {
            s.loss_scale *= 0.5f;
        }
    }
    else
    {
        s.adam_t++;
        double bc1 = 1.0 - pow(static_cast<double>(cfg.beta1), static_cast<double>(s.adam_t));
        double bc2 = 1.0 - pow(static_cast<double>(cfg.beta2), static_cast<double>(s.adam_t));
        s.step_size = static_cast<float>(lr / bc1);
        s.bc2_sqrt = static_cast<float>(sqrt(bc2));
        s.decay = static_cast<float>(1.0 - lr * cfg.weight_decay);
        s.good_steps++;
        if (cfg.dynamic_scale && s.good_steps >= cfg.growth_interval)
        {
            s.loss_scale *= 2.0f;
            s.good_steps = 0;
        }
    }
    if (history != nullptr)
    {
        history[s.step % history_len] = s;
    }
    s.step++;
    *st = s;
}

void begin_step(StepState *st, cudaStream_t s)
{
    begin_step_kernel<<<1, 1, 0, s>>>(st);
    CUDA_CHECK_LAUNCH();
}

void unscale_and_norm(float *grads, long long n, StepState *st, float *partial, cudaStream_t s)
{
    assert(n > 0 && n % 4 == 0);
    unscale_norm_kernel<<<NORM_BLOCKS, BLOCK, 0, s>>>(grads, n / 4, st, partial);
    CUDA_CHECK_LAUNCH();
}

void finalize_step(StepState *st, const float *partial, ScheduleConfig cfg, StepState *history, int history_len,
                   cudaStream_t s)
{
    assert(history == nullptr || history_len > 0);
    finalize_kernel<<<1, NORM_BLOCKS, 0, s>>>(st, partial, cfg, history, history_len);
    CUDA_CHECK_LAUNCH();
}
