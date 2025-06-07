#include "parameter.h"

Parameter::Parameter(size_t num) : numel(num) {
    CUDA_CALL(cudaMalloc(&value, num * sizeof(float)));
    CUDA_CALL(cudaMalloc(&grad, num * sizeof(float)));
    CUDA_CALL(cudaMalloc(&m, num * sizeof(float)));
    CUDA_CALL(cudaMalloc(&v, num * sizeof(float)));
    CUDA_CALL(cudaMemset(m, 0, num * sizeof(float)));
    CUDA_CALL(cudaMemset(v, 0, num * sizeof(float)));
    
    // Set all weights to 0.01 for debugging
    if (POLICY_DEBUG) {
        std::vector<float> host_x(num, 0.001);
        CUDA_CALL(cudaMemcpy(value, host_x.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }
}

Parameter::~Parameter() {
    cudaFree(value);
    cudaFree(grad);
    cudaFree(m);
    cudaFree(v);
}