// common.cu: Timer

#include "common.h"

using namespace std;

Timer::Timer()
{
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
}

Timer::~Timer()
{
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
}

void Timer::start(cudaStream_t s)
{
    CUDA_CHECK(cudaEventRecord(begin, s));
}

void Timer::stop(cudaStream_t s)
{
    CUDA_CHECK(cudaEventRecord(end, s));
}

float Timer::ms()
{
    float result = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&result, begin, end));
    return result;
}
