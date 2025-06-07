#include "fc.h"

__global__ void add_bias_kernel(float* y, const float* b, int M, int num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        int col = idx % M;
        y[idx] += b[col];
    }
}

// For backward
__global__ void reduce_bias_kernel(const float* dy, float* db, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 ... M-1
    if (col >= M) return;

    float sum = 0.f;
    for (int row = 0; row < N; ++row)
        sum += dy[row * M + col]; // row-major with stride M

    db[col] = sum;
}

FC:: FC(int in_feature, int out_feature) 
    : in_feature_(in_feature), out_feature_(out_feature),
      W(in_feature*out_feature), b(out_feature){
}

FC:: ~FC(){
    if (dx) delete dx;
    if (y) delete y;
}

Tensor* FC:: forward(Tensor* x, cublasHandle_t handle, cudaStream_t stream) {
    CUBLAS_CALL(cublasSetStream(handle, stream));
    
    x_cache = x;
    const int N = x->n();
    const int K = in_feature_;
    const int M = out_feature_;

    if (!y) y = new Tensor(N, M, 1, 1); // output (N , M)

    const float alpha = 1.f, beta = 0.f;

    // cuBLAS use column major
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W.value, K, x->data, K, &beta, y->data, M));
    
    const int total = N * M;
    const int blocks  = (total + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    add_bias_kernel<<<blocks, THREAD_PER_BLOCK, 0, stream>>>(y->data, b.value, M, total);

    return y;
}

Tensor* FC:: backward(Tensor* dy, cublasHandle_t handle, cudaStream_t stream) {
    CUBLAS_CALL(cublasSetStream(handle, stream));

    int N = dy->n();
    int M = out_feature_;
    int K = in_feature_;

    const float alpha = 1.0f, beta = 0.0f;

    // dx
    if (!dx) dx = new Tensor(x_cache->n(), x_cache->c(), 1, 1);
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, W.value, K, dy->data, M, &beta, dx->data, K));

    // dW
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, M, N, &alpha, x_cache->data, K, dy->data, M, &beta, W.grad, K));   

    // db
    const int blocks = (M + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    reduce_bias_kernel<<<blocks, THREAD_PER_BLOCK, 0, stream>>>(dy->data, b.grad, N, M);

    return dx;
    
}
