#pragma once
// graph.h: capture a piece of stream work once, replay it with one launch.
//
// A training step of the 6 layer model is close to 200 kernel launches. Each
// costs the CPU a few microseconds; a graph hands the whole list to the GPU
// at once. Everything captured must take its changing inputs from GPU
// memory (the step counter, the loss scale, the batch), never from host
// arguments, because the replay reuses the exact arguments of the capture.

#include <functional>

#include <cuda_runtime.h>

struct CudaGraph
{
    cudaGraphExec_t exec = nullptr;

    CudaGraph() = default;
    ~CudaGraph();
    CudaGraph(const CudaGraph &) = delete;
    CudaGraph &operator=(const CudaGraph &) = delete;

    // runs body with s in capture mode (nothing executes), then instantiates
    void capture(cudaStream_t s, const std::function<void()> &body);
    void launch(cudaStream_t s);
    bool ready() const
    {
        return exec != nullptr;
    }
};
