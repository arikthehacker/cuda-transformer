// graph.cu: CudaGraph

#include "graph.h"

#include "common.h"

using namespace std;

CudaGraph::~CudaGraph()
{
    if (exec != nullptr)
    {
        cudaGraphExecDestroy(exec);
    }
}

void CudaGraph::capture(cudaStream_t s, const function<void()> &body)
{
    if (exec != nullptr)
    {
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
    cudaGraph_t graph;
    // ThreadLocal: other threads may still call the CUDA API (only this one is capturing)
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    g_capturing = true;                     // debug builds: no device syncs while capturing (common.h)
    body();
    g_capturing = false;
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));    // the instantiated copy is all a replay needs
}

void CudaGraph::launch(cudaStream_t s)
{
    CUDA_CHECK(cudaGraphLaunch(exec, s));
#ifdef CUDA_DEBUG
    CUDA_CHECK(cudaStreamSynchronize(s));   // a fault inside the graph shows up at this replay
#endif
}
