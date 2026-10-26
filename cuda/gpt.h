#pragma once
// gpt.h: the model. Allocates everything once, then forward, backward, update.

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "config.h"
#include "layout.h"

struct Dump;

struct GPT
{
    ModelConfig cfg;
    int max_B;                              // the buffers hold up to (max_B, max_T)
    int max_T;
    DeviceBuffer params, grads, m, v;       // same layout (layout.h)
    std::vector<ParamSlot> slots;
    std::vector<LayerParams> lw, lg;        // per layer pointers into params and grads

    // Activations and their gradients: one buffer each, sliced by dump name.
    DeviceBuffer acts, dacts;
    std::vector<ActSlot> act_slots;
    std::unordered_map<std::string, float *> act, dact;
    int step = 0;                           // AdamW steps taken

    GPT(const ModelConfig &cfg, int B, int T);
    float *param(const std::string &name) const;
    float *grad(const std::string &name) const;
    float *A(const std::string &name) const;    // activation by dump name
    float *dA(const std::string &name) const;   // its gradient

    void load_params(const Dump &d);           // from the dump's param records
    float forward(const int *tok, const int *targets, int B, int T);   // returns the loss (0 without targets)
    void zero_grad();
    void backward(const int *tok, const int *targets, int B, int T);
    float clip(float max_norm);                // returns the norm before clipping
    void update(float lr, const TrainConfig &tc);
};
