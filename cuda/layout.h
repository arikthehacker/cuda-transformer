#pragma once
// layout.h: where every parameter and activation lives in memory.
//
// Parameters sit back to back in ONE buffer, in the dump's order (which is
// also model.named_parameters() order). Gradients, AdamW m and AdamW v each
// get one more buffer with the exact same layout, so the same offset finds a
// parameter, its gradient, m and v.
//
//   wte (Vp, C), wpe (n_ctx, C)
//   for each layer l:
//     h.l.ln1.w (C), h.l.ln1.b (C)
//     h.l.attn.c_attn.w (3C, C), h.l.attn.c_attn.b (3C)
//     h.l.attn.c_proj.w (C, C),  h.l.attn.c_proj.b (C)
//     h.l.ln2.w (C), h.l.ln2.b (C)
//     h.l.mlp.c_fc.w (4C, C),    h.l.mlp.c_fc.b (4C)
//     h.l.mlp.c_proj.w (C, 4C),  h.l.mlp.c_proj.b (C)
//   ln_f.w (C), ln_f.b (C)
//
// Activations sit back to back in one buffer too, one slot per dump name
// (N = B*T rows), and their gradients in a second buffer with the same layout:
//   emb (B,T,C)
//   per layer: ln1 (B,T,C), ln1.mean (B,T), ln1.rstd (B,T), qkv (B,T,3C),
//              q, k, v (B,H,T,D) each [not in the dump, the permuted copies],
//              scores (B,H,T,T), probs (B,H,T,T), y (B,H,T,D) [not in the dump],
//              atty (B,T,C), attn_out (B,T,C), res1 (B,T,C),
//              ln2, ln2.mean, ln2.rstd, fc (B,T,4C), gelu (B,T,4C),
//              mlp_out (B,T,C), res2 (B,T,C)
//   ln_f, ln_f.mean, ln_f.rstd, logits (B,T,Vp), losses (B,T), loss (1)

#include <string>
#include <vector>

#include "config.h"

struct ParamSlot
{
    std::string name;
    size_t offset;              // floats from the start of the buffer
    size_t count;
    int ndim;                   // 2D weights get weight decay, 1D ones do not
};

std::vector<ParamSlot> param_layout(const ModelConfig &cfg);    // in dump order

struct LayerParams              // pointers into the params (or grads) buffer for layer l
{
    float *ln1_w, *ln1_b, *attn_w, *attn_b, *proj_w, *proj_b;
    float *ln2_w, *ln2_b, *fc_w, *fc_b, *fcproj_w, *fcproj_b;
};

LayerParams layer_params(float *base, const std::vector<ParamSlot> &slots, int layer);

struct ActSlot
{
    std::string name;
    std::vector<int> dims;
    size_t offset;              // floats from the start of the buffer
    size_t count;
};

std::vector<ActSlot> activation_layout(const ModelConfig &cfg, int B, int T);
