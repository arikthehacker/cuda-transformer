#pragma once
// gpt_fast.h: the stage 4 model. Same parameters and math as gpt.h, built
// for speed:
//   - fp32 master weights, gradients, AdamW state, residual stream and
//     LayerNorm statistics; floatX (fp16 in the -DFP16 build) activations,
//     activation gradients and the weight copy the GEMMs read
//   - tensor core GEMMs with bias / GELU / GELU-backward fused into the
//     epilogue, flash attention, fused residual + LayerNorm
//   - dynamic loss scaling, the gradient norm, clipping, the lr schedule and
//     AdamW all decided on the GPU (StepState), so train_step() never waits
//     for the host and can be captured into one CUDA Graph
//
// The buffers are sized for (B, T) when the model is made. set_seq_len()
// runs shorter sequences in the same buffers (generation uses that); a graph
// replays the shape it was captured with. Everything runs on the model's own
// stream.

#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "layout.h"
#include "kernels/fast.h"

struct Dump;

struct LayerWeightsX            // the floatX weight copies the GEMMs read, for layer l
{
    const floatX *attn_w, *proj_w, *fc_w, *fcproj_w;
};

struct FastLayer                // the activations backward needs, for layer l
{
    floatX *ln1, *qkv, *atty, *ln2, *fc;    // (N, C), (N, 3C), (N, C), (N, C), (N, 4C)
    float *ln1_mean, *ln1_rstd, *ln2_mean, *ln2_rstd;   // (N)
    float *res1, *res2;         // (N, C) fp32, the residual stream
    float *lse;                 // (B, H, T) flash attention row log-sum-exp
    float *q, *k, *v, *probs;   // the stage 3 attention's buffers (unfused attention)
};

struct FastOptions
{
    bool flash = true;          // flash attention (fp16 build); false: stage 3's unfused fp32 attention
    int history_len = 1024;     // StepState ring buffer length
};

struct ActView                  // one activation, by its dump name (for the tests)
{
    const void *ptr = nullptr;
    bool is_x = false;          // floatX (true) or fp32
    size_t count = 0;
};

struct FastGPT
{
    ModelConfig cfg;
    int B, T, N;                            // N = B * T rows, T <= T_max
    int T_max;
    bool flash;
    cudaStream_t stream;

    DeviceBuffer params, grads, m, v;       // fp32, the stage 3 layout (layout.h)
    DeviceArray<floatX> params_x;           // fp16 copy of params (fp16 build only)
    std::vector<ParamSlot> slots;
    DeviceArray<TensorInfo> tensors;        // the same slots, for adamw_multi
    std::vector<LayerParams> lw, lg;        // fp32 params and grads per layer
    std::vector<LayerWeightsX> lx;
    const floatX *wte_x;
    float *wte, *wpe, *lnf_w, *lnf_b;       // fp32 params outside the layers
    float *dwte, *dwpe, *dlnf_w, *dlnf_b;   // and their gradients

    DeviceInts x, y;                        // the batch forward() reads, (B, T) each
    DeviceArray<StepState> state;
    DeviceArray<StepState> history;         // finalize_step writes step k into history[k % history_len]
    int history_len;
    ScheduleConfig sched;
    float beta1, beta2, eps;

    // activations (one floatX block, one fp32 block, sliced)
    DeviceArray<floatX> xmem;
    DeviceBuffer fmem;
    float *emb, *lnf_mean, *lnf_rstd, *losses, *partial, *colsum;
    floatX *lnf, *logits;                   // logits holds dlogits after a training forward
    std::vector<FastLayer> layers;
    // scratch shared by all layers
    floatX *proj_out, *gelu, *dfc, *dln, *datty, *dqkv, *dres_h;
    float *dres;                            // (N, C) fp32 gradient of the residual stream
    float *Dq, *dq_acc;                     // flash backward
    float *scores, *att_y, *dq, *dk, *dv, *dscores, *dprobs, *dy;   // unfused attention
    float *qkv32, *atty32, *dqkv32;         // unfused attention in the fp16 build: fp32 copies

    FastGPT(const ModelConfig &cfg, int B, int T, const TrainConfig &tc, float loss_scale, FastOptions opt = {});
    ~FastGPT();
    FastGPT(const FastGPT &) = delete;
    FastGPT &operator=(const FastGPT &) = delete;

    void load_params(const Dump &d);                    // from the dump's param records
    void set_params(const std::vector<float> &host);    // all of params, in layout order
    void reset(float loss_scale);                       // m = v = 0, a fresh StepState
    float *param(const std::string &name) const;
    float *grad(const std::string &name) const;
    void set_seq_len(int t);                // a multiple of 64, at most T_max

    // forward on x (and y): the mean loss goes to loss_out (GPU memory). With
    // train, the logits buffer is overwritten with dlogits * S.
    void forward(bool train, float *loss_out);
    void backward();                        // after forward(true): every gradient * S, accumulated
    void optimizer_step();                  // unscale, norm, finalize, AdamW
    // the whole step: begin_step, zero grads, forward, backward, optimizer step
    void train_step();

    StepState read_state();                 // waits for the stream
    ActView act(const std::string &name) const;

private:
    void attention_forward_layer(FastLayer &a);
    void attention_backward_layer(FastLayer &a);
};

// The fp16 / fp32 copy of a floatX or fp32 buffer, on the host (waits for the stream).
std::vector<float> download_view(const ActView &a, cudaStream_t s);
