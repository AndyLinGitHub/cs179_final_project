#include "softplus_add1.h"

// y = softplus(x) + 1
__device__ inline float softplus_add1(float x)
{
    const float t = 20.0f;
    if (x >  t)  return x + 1.0f;
    if (x < -t)  return expf(x) + 1.0f;

    return log1pf(expf(x)) + 1.0f;
}

// Numerically-stable sigmoid
__device__ inline float sigmoid(float x)
{
    // avoids overflow
    if (x >= 0.f) {
        float z = expf(-x);
        return 1.f / (1.f + z); 
    }

    // avoids underflow
    else {
        float z = expf(x);
        return z / (1.f + z); 
    }
}

__global__ void softplus_add1_forward_kernel(const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = softplus_add1(x[i]);
}

// dx = dy * sigmoid(x)
__global__ void softplus_add1_backward_kernel(const float *x, const float *dy, float *dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * sigmoid(x[i]);
}

SoftPlusAdd1::SoftPlusAdd1(cudaStream_t stream) : stream_(stream) {  
}

SoftPlusAdd1::~SoftPlusAdd1() {
    // Since x is passed from outside the class, we should not delete it here.
    // if (x_cache) delete x_cache;

    if (dx) delete dx;
    if (y) delete y;
}

Tensor* SoftPlusAdd1:: forward(Tensor* x) {
    x_cache = x;
    if (!y) y = new Tensor(x->n(), x->c(), x->h(), x->w());

    const int N = x->n()*x->c()*x->h()*x->w();
    const int blocks  = (N + THREAD_NUM - 1) / THREAD_NUM;

    softplus_add1_forward_kernel<<<blocks, THREAD_NUM, 0, stream_>>>(x->data, y->data, N);

    return y;
}

Tensor* SoftPlusAdd1:: backward(Tensor* dy) {
    if (!dx) dx = new Tensor(x_cache->n(), x_cache->c(), x_cache->h(), x_cache->w());

    const int N = x_cache->n()*x_cache->c()*x_cache->h()*x_cache->w();
    const int blocks  = (N + THREAD_NUM - 1) / THREAD_NUM;

    softplus_add1_backward_kernel<<<blocks, THREAD_NUM, 0, stream_>>>(x_cache->data, dy->data, dx->data, N);

    return dx;
}
