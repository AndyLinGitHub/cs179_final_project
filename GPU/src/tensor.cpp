#include "tensor.h"

Tensor::Tensor(int n, int c, int h, int w, bool allocate) : dims{n, c, h, w} {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    if (allocate) {
        size_t bytes_ = static_cast<size_t>(n) * c * h * w * sizeof(float);
        CUDA_CALL(cudaMalloc(&data, bytes_));
    }
}

Tensor::~Tensor() {
    if (data) cudaFree(data);
    cudnnDestroyTensorDescriptor(desc);
}

size_t Tensor::bytes() const {
    return static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3] * sizeof(float);
}

void Tensor::zero() {
    CUDA_CALL(cudaMemset(data, 0, bytes()));
}