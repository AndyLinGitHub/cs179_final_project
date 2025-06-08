#include "parameter.h"

std::vector<float> make_normal_vector(std::size_t n, float mean = 0.0, float variance = 0.1)
{
    const float stddev = std::sqrt(variance);
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{mean, stddev};

    std::vector<float> v(n);
    std::generate(v.begin(), v.end(), [&]{ return dist(gen);});

    return v;
}

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
    else {
        std::vector<float> host_x = make_normal_vector(num, 0.0, 0.001);
        CUDA_CALL(cudaMemcpy(value, host_x.data(), num * sizeof(float), cudaMemcpyHostToDevice));
    }

}

Parameter::~Parameter() {
    cudaFree(value);
    cudaFree(grad);
    cudaFree(m);
    cudaFree(v);
}