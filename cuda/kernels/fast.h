#pragma once
// fast.h: the stage 4 launchers. Everything takes a stream, so a whole
// training step can be captured into one CUDA Graph (the legacy default
// stream cannot be captured).
//
// Types: floatX (precision.h) is fp16 in the -DFP16 build and fp32 otherwise.
// "fp32" below means float in both builds.

#include <cuda_runtime.h>

#include <cstdint>

#include "precision.h"

struct StepState;           // scaler.cu below

// ---------------------------------------------------------------- GEMM (gemm_*.cu)
// C (M x N) = op(A) @ op(B), row-major, op(A) = A^T if transA (A stored K x M).
// M, N, K multiples of 8, lda/ldb/ldc multiples of 8, pointers 16-byte aligned.
//
// gemm_act:  floatX output with an epilogue (see gemm_common.cuh):
//            ep 0 store, 1 + bias, 2 + bias then gelu into out2, 3 * gelu'(aux)
// gemm_grad: fp32 output, C += acc (parameter gradients). The mma.sync version
//            splits a long K over extra blocks that add into C with atomics.
enum class GemmImpl
{
    Mma,    // inline PTX mma.sync m16n8k8 + ldmatrix (fp16 build only)
    Wmma,   // nvcuda::wmma 16x16x16 (fp16 build only)
    Simt,   // fp32 CUDA cores, the stage 3 tiling (both builds; the fp32 build always uses it)
};
void set_gemm_impl(GemmImpl impl);
GemmImpl gemm_impl();
const char *gemm_impl_name(GemmImpl impl);

void gemm_act(int ep, bool transA, bool transB, int M, int N, int K,
              const floatX *A, int lda, const floatX *B, int ldb, floatX *C, int ldc,
              const float *bias, floatX *out2, const floatX *aux, cudaStream_t s);
void gemm_grad(bool transA, bool transB, int M, int N, int K,
               const floatX *A, int lda, const floatX *B, int ldb, float *C, int ldc, cudaStream_t s);

// the three implementations behind gemm_act / gemm_grad (gemm_mma.cu, gemm_wmma.cu, gemm_simt.cu)
template <int EP, class TC>
void gemm_mma(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
              TC *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s);
template <int EP, class TC>
void gemm_wmma(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
               TC *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s);
template <int EP, class TC>
void gemm_simt(bool transA, bool transB, int M, int N, int K, const floatX *A, int lda, const floatX *B, int ldb,
               TC *C, int ldc, const float *bias, floatX *out2, const floatX *aux, cudaStream_t s);

// ---------------------------------------------------------------- fused.cu
// out (N, C) floatX = LayerNorm(x fp32), also mean and rstd (N) fp32
void layernorm_fwd(floatX *out, float *mean, float *rstd, const float *x, const float *w, const float *b,
                   int N, int C, float eps, cudaStream_t s);
// res (N, C) fp32 = x fp32 + y floatX, and out floatX = LayerNorm(res), in one pass
void residual_layernorm_fwd(float *res, floatX *out, float *mean, float *rstd, const float *x, const floatX *y,
                            const float *w, const float *b, int N, int C, float eps, cudaStream_t s);
// LayerNorm backward into the fp32 residual gradient:
//   dres += dx, dres_h = floatX(dres) (the copy the next GEMMs read), dw += ..., db += ...
// scratch: 2 * COLSUM_SPLITS * C floats (the column sums for dw and db go through it)
void layernorm_bwd(float *dres, floatX *dres_h, float *dw, float *db, const floatX *dout, const float *x,
                   const float *w, const float *mean, const float *rstd, int N, int C, float *scratch, cudaStream_t s);
constexpr int COLSUM_SPLITS = 32;           // row chunks of the column sums (bias and LayerNorm gradients)
// emb (B, T, C) fp32 = wte[tok] + wpe[t]  (fp32 master weights)
void encoder_fwd(float *emb, const int *tok, const float *wte, const float *wpe, int B, int T, int C, cudaStream_t s);
// dwte += scatter(dres), dwpe += sum over b of dres  (atomicAdd into dwte)
void encoder_bwd(float *dwte, float *dwpe, const float *dres, const int *tok, int B, int T, int C, cudaStream_t s);
// dbias (OC) fp32 += column sums of dout (N, OC) floatX, scratch: COLSUM_SPLITS * OC floats
void bias_bwd(float *dbias, const floatX *dout, int N, int OC, float *scratch, cudaStream_t s);
// out = gelu(in), floatX (the backward pass recomputes gelu from fc)
void gelu_fwd(floatX *out, const floatX *in, long long n, cudaStream_t s);
// Cross entropy over the V real columns, one block per row. A target of -1
// is not counted (SFT: only the answers are trained on). count points at the
// number of targets >= 0 in GPU memory (count_targets below), nullptr = N.
// Training: losses[n] fp32, and the logits row is OVERWRITTEN with
//   dlogits = (softmax - onehot) * scale / count, padded columns set to 0,
//   and a row whose target is -1 gets loss 0 and dlogits 0.
//   scale and count are read from GPU memory, so a graph can replay them.
// Eval (grad false): losses only, logits untouched.
void crossentropy_fused(float *losses, floatX *logits, const int *targets, int N, int V, int Vp,
                        const float *scale, const int *count, bool grad, cudaStream_t s);
// out[0] = sum of losses[0 .. N-1] / count (nullptr = N), one block, fixed order
void mean_loss(float *out, const float *losses, int N, const int *count, cudaStream_t s);
// st->n_valid = how many of targets[0 .. N-1] are >= 0, at least 1, so a
// batch with nothing counted gives loss 0 instead of 0 / 0. One block.
void count_targets(StepState *st, const int *targets, int N, cudaStream_t s);
// fp32 -> floatX copy (the weight copy the GEMMs read)
void cast_to_x(floatX *out, const float *in, long long n, cudaStream_t s);
// floatX -> fp32 copy
void cast_to_f(float *out, const floatX *in, long long n, cudaStream_t s);
// The batch of step st->step from tokens on the GPU (uint16, 2 bytes per
// id, so 1B pretraining tokens fit in 2 GB): for row b,
// start = starts[(st->step % nsteps) * B + b], x[b, t] = tokens[start + t],
// y[b, t] = tokens[start + t + 1], or targets[start + t] when there are SFT
// targets (int32, -1 = not counted). Every step's starts are uploaded once,
// so a graph replay picks its own batch.
void load_batch(int *x, int *y, const uint16_t *tokens, const int *targets, const int *starts, int nsteps,
                const StepState *st, int B, int T, cudaStream_t s);

// ---------------------------------------------------------------- scaler.cu
// Everything a step decides lives on the GPU, so a CUDA Graph can replay a
// whole training step with no host work in between.
struct StepState
{
    float loss_scale;       // S, multiplies dlogits at the start of backward
    float inv_scale;        // 1 / S of THIS step (unscale uses it)
    int good_steps;         // finite steps in a row since S last changed
    int found_inf;          // this step's gradients had an inf or NaN: AdamW skips the step
    int skipped;            // steps skipped so far
    int step;               // training steps so far, skipped ones included (drives the lr schedule)
    int adam_t;             // AdamW steps taken (skipped steps do not count, like torch)
    float lr;               // of this step
    float grad_norm;        // of the unscaled gradients, before clipping
    float clip_coef;        // min(1, clip / (norm + 1e-6))
    float step_size;        // lr / (1 - beta1^t)
    float bc2_sqrt;         // sqrt(1 - beta2^t)
    float decay;            // 1 - lr * weight_decay (for the 2D weights)
    float loss;             // mean loss of this step
    int n_valid;            // targets >= 0 in this step's batch (B * T outside SFT), the loss divides by it
};

struct ScheduleConfig       // the numbers finalize needs, same values as TrainConfig
{
    float lr, min_lr;
    int warmup_steps, max_steps;
    float beta1, beta2;
    float weight_decay;
    float grad_clip;
    int growth_interval;    // 2,000: finite steps in a row before S doubles
    int dynamic_scale;      // 0: S stays where it is (the fp32 build)
};

// a fresh state: S = scale, every counter 0
StepState initial_state(float scale);
// start of a step: found_inf = 0, inv_scale = 1 / S
void begin_step(StepState *st, cudaStream_t s);
// grads *= inv_scale in place, flag inf/NaN, per block sums of squares into partial[NORM_BLOCKS]
void unscale_and_norm(float *grads, long long n, StepState *st, float *partial, cudaStream_t s);
// one block: norm, clip coefficient, skip decision, loss scale update, lr and
// bias corrections for this step, then step++. A copy of the finished state
// goes to history[step % history_len] (history may be nullptr), so the host
// can read many steps back at once instead of syncing every step.
void finalize_step(StepState *st, const float *partial, ScheduleConfig cfg, StepState *history, int history_len,
                   cudaStream_t s);
constexpr int NORM_BLOCKS = 512;

// ---------------------------------------------------------------- adamw_multi.cu
struct TensorInfo
{
    long long offset;       // floats from the start of the parameter buffer
    long long count;
    int decay;              // 2D weights: 1, biases and LayerNorm: 0
};
// One launch for every parameter, 4 values per thread. Reads lr, clip_coef,
// bias corrections and the skip flag from st (does nothing when the step is
// skipped). Writes the floatX weight copy too (params_x may be nullptr).
// Every offset and count must be a multiple of 4. tensors is a device array.
constexpr int MAX_TENSORS = 1024;
void adamw_multi(float *params, floatX *params_x, const float *grads, float *m, float *v, long long n,
                 const TensorInfo *tensors, int ntensors, const StepState *st, float beta1, float beta2,
                 float eps, cudaStream_t s);

// ---------------------------------------------------------------- flash_attn.cu (fp16 build)
// Causal attention straight from qkv (B, T, 3C) to atty (B, T, C), never
// writing the (B, H, T, T) scores. L (B, H, T) fp32 = m + log(l) per row, for
// backward. T must be a multiple of 64, D one of 16, 32, 64, 128.
void flash_attn_fwd(floatX *atty, float *L, const floatX *qkv, int B, int T, int H, int D, cudaStream_t s);
// dqkv (B, T, 3C) floatX from datty. Scratch: Dq (B, H, T) fp32 and
// dq_acc (B, T, C) fp32 (dq is added up with atomics, then converted).
void flash_attn_bwd(floatX *dqkv, float *Dq, float *dq_acc, const floatX *datty, const floatX *qkv,
                    const floatX *atty, const float *L, int B, int T, int H, int D, cudaStream_t s);
