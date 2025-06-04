#include "fc_layer.h"
#include <cassert>

void add_bias_cudnn(float* d_y,
    cudnnTensorDescriptor_t y_desc,   // (N,M,1,1)
    const float* d_b, int M,
    cudnnHandle_t hCudnn)
{
const float alpha = 1.f, beta = 1.f;       // y = alpha*b + beta*y

cudnnTensorDescriptor_t b_desc;
CUDNN_CALL(cudnnCreateTensorDescriptor(&b_desc));
CUDNN_CALL(cudnnSetTensor4dDescriptor(
b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
1, M, 1, 1));                          // shape (1,M,1,1)

CUDNN_CALL(cudnnAddTensor(
hCudnn, &alpha,
b_desc, d_b,
&beta,
y_desc, d_y));

cudnnDestroyTensorDescriptor(b_desc);
}

FullyConnected::FullyConnected(int in_features, int out_features)
    : in_f_{in_features}, out_f_{out_features},
      W(out_features*in_features),
      b(out_features)
{
    CUBLAS_CALL(cublasCreate(&hBlas_));
}

FullyConnected::~FullyConnected()
{
    if (y) delete y;
    CUBLAS_CALL(cublasDestroy(hBlas_));
}

/* -------------------------- forward --------------------------------------- */
Tensor* FullyConnected::forward(Tensor* x, cudnnHandle_t hCudnn)
{
    x_cache = x;                     // save for backward
    int N = x->n();
    int K = in_f_;
    int M = out_f_;

    // flatten X on-the-fly: (N , C*H*W) => (N , K)
    assert(K == x->c()*x->h()*x->w());

    if (!y) y = new Tensor(N, M, 1, 1);          // output (N , M)

    const float alpha = 1.f, beta = 0.f;
    // cuBLAS is column-major; use A^T * B^T trick to keep memory contiguous
    // y = X * W^T  =>  (M x N)^T  = (W x X^T)
    CUBLAS_CALL(cublasSgemm(
        hBlas_,
        CUBLAS_OP_N,       // A^T  (W is row-major out_f_×in_f_)
        CUBLAS_OP_T,       // B^T  (X is row-major N×K)
        N, M, K,
        &alpha,
        x->data, N,          // lda
        W.value, M,           // ldb
        &beta,
        y->data, N));         // ldc (row-major y)

    /* bias add : broadcast (1,M) over N samples with cuDNN */

    add_bias_cudnn(y->data, y->desc, b.value, M, hCudnn);

    return y;
}

/* -------------------------- backward -------------------------------------- */
Tensor* FullyConnected::backward(Tensor* dy, cudnnHandle_t hCudnn)
{
    assert(x_cache);                    // forward must run first
    int N = dy->n();
    int M = out_f_;
    int K = in_f_;

    /* ---- dW ----  (out_f_ , in_f_)  */
    const float alpha = 1.f, beta = 1.f;
    // dW += dy^T * X
    CUBLAS_CALL(cublasSgemm(
        hBlas_,
        CUBLAS_OP_N,         // (out_f_ , N)
        CUBLAS_OP_T,         // (N , K)
        M, K, N,
        &alpha,
        dy->data,  M,
        x_cache->data, K,
        &beta,
        W.grad, M));

    /* ---- db ----  (1 , out_f_)  = sum_N dy */
    // db += dy^T * ones
    static float* d_ones = nullptr;      // lazy-allocate
    if (!d_ones) {
        CUDA_CALL(cudaMalloc(&d_ones, sizeof(float)*N));
        thrust::device_ptr<float> t(d_ones);
        thrust::fill(t, t+N, 1.f);
    }
    CUBLAS_CALL(cublasSgemv(
        hBlas_,
        CUBLAS_OP_T,          // (M×N) ^T
        M, N,
        &alpha,
        dy->data, M,
        d_ones, 1,
        &beta,
        b.grad, 1));

    /* ---- dx ----  (N , K) = dy * W  */
    Tensor* dx = new Tensor(x_cache->n(), x_cache->c(),
                            x_cache->h(), x_cache->w(), false /*no alloc*/);
    CUDA_CALL(cudaMalloc(&dx->data, x_cache->bytes()));

    CUBLAS_CALL(cublasSgemm(
        hBlas_,
        CUBLAS_OP_N,       // (N , M)
        CUBLAS_OP_N,       // (M , K)
        K, N, M,
        &alpha,
        W.value, K,  // W row-major  (M×K)  but treat as column-maj
        dy->data, M,
        &beta,
        dx->data, K));

    return dx;
}
