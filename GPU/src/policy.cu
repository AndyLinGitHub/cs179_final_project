#include "policy.h"
#include <iostream>

__global__
void add2(const float *a, const float *b, float *out, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}

__global__
void add3(const float *a, const float *b, const float *c,
          float *out, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i] + c[i];
}

__global__ void adam_update_kernel(float* p, float* g, float* m, float* v, size_t n, float lr, float beta1, float beta2, float eps, int t) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gi = g[i];
        float mi = m[i] = beta1 * m[i] + (1 - beta1) * gi;
        float vi = v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        float m_hat = mi / (1 - powf(beta1, t));
        float v_hat = vi / (1 - powf(beta2, t));
        p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        g[i] = 0.0f; // clear grad after update
    }
}

Policy:: Policy(float lr, float beta1, float beta2, float eps) : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {

}

Policy:: ~Policy() {
    if (dx_sum) delete dx_sum;
}

Tensor* Policy:: forward(Tensor* x, cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle, cudaStream_t stream) {
    temp = conv_1.forward(x, cudnn_handle, stream);
    temp = relu_1.forward(temp, cudnn_handle, stream);
    temp = conv_2.forward(temp, cudnn_handle, stream);
    temp = relu_2.forward(temp, cudnn_handle, stream);
    temp = fc_1.forward(temp, cublas_handle, stream);
    temp = relu_3.forward(temp, cudnn_handle, stream);

    alpha_ = fc_2.forward(temp, cublas_handle, stream);
    beta_ = fc_3.forward(temp, cublas_handle, stream);
    value_ = fc_4.forward(temp, cublas_handle, stream);

    alpha_ = spa_1.forward(alpha_, stream);
    beta_ = spa_2.forward(beta_, stream);
    bd.forward(alpha_, beta_, stream);

    return value_;
}

Tensor* Policy:: backward(Tensor* dlogp, Tensor* dh, Tensor* dv, cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle, cudaStream_t stream) {
    bd.backward(dlogp, dh, stream);
    dx_1 = spa_2.backward(bd.db(), stream);
    dx_2 = spa_1.backward(bd.da(), stream);
    dx_3 = fc_4.backward(dv, cublas_handle, stream);
    dx_1 = fc_3.backward(dx_1, cublas_handle, stream);
    dx_2 = fc_2.backward(dx_2, cublas_handle, stream);

    if (!dx_sum) dx_sum = new Tensor(dx_3->n(), dx_3->c(), dx_3->h(), dx_3->w());
    const int total = dx_3->n() * dx_3->c() * dx_3->h() * dx_3->w();
    const int blocks  = (total + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    add3<<<blocks, THREAD_PER_BLOCK, 0, stream>>>(dx_1->data, dx_2->data, dx_3->data, dx_sum->data, total);

    dx_1 = relu_3.backward(dx_sum, cudnn_handle, stream);
    dx_1 = fc_1.backward(dx_1, cublas_handle, stream, true); //True flag for indicating there is a implicit flatten operation before forwarding to this layer
    dx_1 = relu_2.backward(dx_1, cudnn_handle, stream);
    dx_1 = conv_2.backward(dx_1, cudnn_handle, stream);
    dx_1 = relu_1.backward(dx_1, cudnn_handle, stream);
    dx_1 = conv_1.backward(dx_1, cudnn_handle, stream);

    return dx_1;
}

void Policy:: step(cudaStream_t stream) {
    ++t;
    int blocks = 0;
    size_t n = 0;

    param = conv_1.params()[0];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = conv_1.params()[1];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = conv_2.params()[0];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = conv_2.params()[1];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = fc_1.params()[0];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);
    
    param = fc_1.params()[1];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = fc_2.params()[0];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);
    
    param = fc_2.params()[1];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = fc_3.params()[0];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);
    
    param = fc_3.params()[1];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    param = fc_4.params()[0];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);
    
    param = fc_4.params()[1];
    n = param->numel;
    blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
    adam_update_kernel<<<blocks, THREAD_NUM, 0, stream>>>(param->value, param->grad, param->m, param->v, n, lr_, beta1_, beta2_, eps_, t);

    CUDA_CALL(cudaGetLastError());
}
