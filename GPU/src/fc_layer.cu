#include "fc_layer.h"
#include <cassert>

__global__ void addBias(float* Y, const float* b, int M, int nElem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElem) {
        int col = idx % M;
        Y[idx] += b[col];
    }
}

__global__ void reduceBias(const float* __restrict__ dY,
                           float* __restrict__ db,
                           int N, int M)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 … M-1
    if (col >= M) return;

    float sum = 0.f;
    for (int row = 0; row < N; ++row)
        sum += dY[row * M + col];    // row-major stride M
    db[col] = sum;
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
    CUBLAS_CALL(cublasSgemm(hBlas_,
                             CUBLAS_OP_T,   // op(A) = Aᵀ  -> (M,K)
                             CUBLAS_OP_N,   // op(B) = X   -> (K,N)
                             M, N, K,       // gemm sizes: m, n, k
                             &alpha,
                             W.value, K,         // lda = leading dim = K  (row stride of A)
                             x->data, K,         // ldb = K
                             &beta,
                             y->data, M));
    
    int total = N * M;
    addBias<<<(total + 255) / 256, 256>>>(y->data, b.value, M, total);

    return y;
}

/* -------------------------- backward -------------------------------------- */
Tensor* FullyConnected::backward(Tensor* dy, cudnnHandle_t hCudnn)
{
    assert(x_cache);                    // forward must run first
    int N = dy->n();
    int M = out_f_;
    int K = in_f_;

    const float alpha = 1.0f, beta = 0.0f;

    //dx
    Tensor* dx = new Tensor(x_cache->n(), x_cache->c(), 1, 1);
    CUBLAS_CALL(cublasSgemm(hBlas_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        /*m=*/K,      /*n=*/N, /*k=*/M,
                        &alpha,
                        W.value,  K,          // lda = K
                        dy->data, M,          // ldb = M
                        &beta,
                        dx->data, K));        // ldc = K  (row stride of X)

    //dW
    //Tensor* dWT = new Tensor(in_f_, out_f_, 1, 1);
    CUBLAS_CALL(cublasSgemm(hBlas_,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        /*m=*/K,      /*n=*/M, /*k=*/N,
                        &alpha,
                        x_cache->data,  K,          // Xᵀ  (K,N)
                        dy->data, M,          // (dY)ᵗ  (N,M)
                        &beta,
                        W.grad, K));      // K×M result

    /* transpose tmp (K,M) → dA (M,K) */
    
    //CUBLAS_CALL(cublasSgeam(hBlas_,
    //                        CUBLAS_OP_T, CUBLAS_OP_N,
    //                        /*m=*/K, /*n=*/M,
    //                        &alpha,
    //                        dWT->data, K,
     //                       &beta,
     //                       dWT->data, K,        // unused
     //                       W.grad,  M));       // row-major target
    
    //delete dWT;

    //db
    int threads = 256;
    int blocks  = (M + threads - 1) / threads;
    reduceBias<<<blocks, threads>>>(dy->data, b.grad, N, M);

    return dx;
    
}
