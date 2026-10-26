#pragma once
// common.h: error checks, ceil_div, DeviceBuffer, Timer

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Wrap EVERY CUDA API call. Prints file:line and the error string, then exits.
#define CUDA_CHECK(call)                                                            \
    do                                                                              \
    {                                                                               \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess)                                                    \
        {                                                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n    %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_), #call);                               \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// After a kernel launch: catches bad launch configs right away. In debug
// builds (make DEBUG=1) it also waits for the kernel, so a crash inside it
// is reported at the right line instead of at some later CUDA call.
#ifdef CUDA_DEBUG
#define CUDA_CHECK_LAUNCH()                     \
    do                                          \
    {                                           \
        CUDA_CHECK(cudaGetLastError());         \
        CUDA_CHECK(cudaDeviceSynchronize());    \
    } while (0)
#else
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#endif

inline int ceil_div(long long a, int b)
{
    return static_cast<int>((a + b - 1) / b);
}

// Owns one cudaMalloc and frees it in the destructor. Move only, no copies.
// It is a template (so tokens can live in a DeviceArray<int>), and templates
// have to be defined in the header.
template <typename T>
struct DeviceArray
{
    T *ptr = nullptr;
    size_t count = 0;                       // elements, not bytes

    DeviceArray() = default;

    explicit DeviceArray(size_t n) : count(n)
    {
        if (n > 0)
        {
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }

    ~DeviceArray()
    {
        if (ptr)
        {
            cudaFree(ptr);                  // no CUDA_CHECK: never exit from a destructor
        }
    }

    DeviceArray(DeviceArray &&o) noexcept : ptr(o.ptr), count(o.count)
    {
        o.ptr = nullptr;
        o.count = 0;
    }

    DeviceArray &operator=(DeviceArray &&o) noexcept
    {
        if (this != &o)
        {
            if (ptr)
            {
                cudaFree(ptr);
            }
            ptr = o.ptr;
            count = o.count;
            o.ptr = nullptr;
            o.count = 0;
        }
        return *this;
    }

    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    void zero(cudaStream_t s = 0)           // cudaMemsetAsync to 0
    {
        if (count > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(ptr, 0, count * sizeof(T), s));
        }
    }

    void upload(const T *host, size_t n)    // first n elements, host -> GPU
    {
        if (n > count)
        {
            fprintf(stderr, "upload of %zu elements into a buffer of %zu\n", n, count);
            exit(1);
        }
        CUDA_CHECK(cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<T> download() const         // whole buffer, GPU -> host
    {
        std::vector<T> host(count);
        if (count > 0)
        {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return host;
    }
};

using DeviceBuffer = DeviceArray<float>;
using DeviceInts = DeviceArray<int>;

// cudaEvent based timer: start(), stop(), ms(). Times GPU work, not launches.
struct Timer
{
    cudaEvent_t begin;
    cudaEvent_t end;

    Timer();
    ~Timer();
    Timer(const Timer &) = delete;
    Timer &operator=(const Timer &) = delete;

    void start(cudaStream_t s = 0);
    void stop(cudaStream_t s = 0);
    float ms();                             // waits for stop(), then milliseconds between them
};
