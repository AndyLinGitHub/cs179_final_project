#include "parameter.h"

Parameter::Parameter(size_t num) : numel(num) {
    CUDA_CALL(cudaMalloc(&value, num * sizeof(float)));
    CUDA_CALL(cudaMalloc(&grad, num * sizeof(float)));
    CUDA_CALL(cudaMalloc(&m, num * sizeof(float)));
    CUDA_CALL(cudaMalloc(&v, num * sizeof(float)));
    CUDA_CALL(cudaMemset(m, 0, num * sizeof(float)));
    CUDA_CALL(cudaMemset(v, 0, num * sizeof(float)));
}

Parameter::~Parameter() {
    cudaFree(value);
    cudaFree(grad);
    cudaFree(m);
    cudaFree(v);
}