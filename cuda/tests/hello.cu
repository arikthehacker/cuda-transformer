// hello.cu: step 1. Proves nvcc, the driver and WSL all work.
//   make cuda-hello

#include <cstdio>

#include "common.h"

using namespace std;

__global__ void hello_kernel(int *out)
{
    out[threadIdx.x] = threadIdx.x;
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU 0: %s, compute capability %d.%d, %.1f GB\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);

    DeviceInts d(32);
    hello_kernel<<<1, 32>>>(d.ptr);
    CUDA_CHECK_LAUNCH();
    vector<int> h = d.download();

    bool ok = true;
    for (int k = 0; k < 32; k++)
    {
        printf("%d ", h[k]);
        ok = ok && h[k] == k;
    }
    printf("\n%s\n", ok ? "ok: 0 .. 31 came back from the GPU" : "WRONG values came back");
    return ok ? 0 : 1;
}
