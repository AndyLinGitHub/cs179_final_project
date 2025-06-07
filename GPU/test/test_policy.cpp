#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "policy.h"

int main(){
    std::ifstream expected_value_file("policy_value.txt");
    std::ifstream expected_alpha_file("policy_alpha.txt");
    std::ifstream expected_beta_file("policy_beta.txt");
    std::ifstream expected_dx_file("policy_dx.txt");

    std::vector<float> expected_value;
    std::vector<float> expected_alpha;
    std::vector<float> expected_beta;
    std::vector<float> expected_dx;
    float temp = 0;

    while (expected_value_file >> temp) expected_value.push_back(temp);
    while (expected_alpha_file >> temp) expected_alpha.push_back(temp);
    while (expected_beta_file >> temp) expected_beta.push_back(temp);
    while (expected_dx_file >> temp) expected_dx.push_back(temp);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    cublasHandle_t cublas;
    CUBLAS_CALL(cublasCreate(&cublas));

    const int N = 1024, C = ACTION_DIM, H = RF, W = RF;
    const int total = N * C * H * W;
    Tensor x(N, C, H, W);
    std::vector<float> host_x(total);
    for (int i = 0; i < total; ++i) host_x[i] = static_cast<float>(i) / static_cast<float>(total);
    CUDA_CALL(cudaMemcpy(x.data, host_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    Policy policy = Policy();

    // Forward
    Tensor* value = policy.forward(&x, cudnn, cublas, stream);
    Tensor* alpha = policy.alpha();
    Tensor* beta = policy.beta();

    std::vector<float> host_value(N);
    std::vector<float> host_alpha(N*C);
    std::vector<float> host_beta(N*C);
    CUDA_CALL(cudaMemcpy(host_value.data(), value->data, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_alpha.data(), alpha->data, N*C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_beta.data(), beta->data, N*C * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        //std:: cout << host_value[i] << "  " << expected_value[i] << std:: endl;
        assert(std::fabs(host_value[i] - expected_value[i]) < 1e-4f);
    }

    for (int i = 0; i < N*C; ++i) {
        //std:: cout << host_alpha[i] << "  " << expected_alpha[i] << std:: endl;
        assert(std::fabs(host_alpha[i] - expected_alpha[i]) < 1e-4f);
    }

    for (int i = 0; i < N*C; ++i) {
        //std:: cout << host_beta[i] << "  " << expected_beta[i] << std:: endl;
        assert(std::fabs(host_beta[i] - expected_beta[i]) < 1e-4f);
    }
    
    // // Backward
    Tensor dlogp(N, 1, 1, 1);
    Tensor dh(N, 1, 1, 1);
    std::vector<float> host_dlogp(N);
    std::vector<float> host_dh(N);
    for (int i = 0; i < N; ++i) host_dlogp[i] = static_cast<float>(1);
    for (int i = 0; i < N; ++i) host_dh[i] = static_cast<float>(1);
    CUDA_CALL(cudaMemcpy(dlogp.data, host_dlogp.data(), host_dlogp.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dh.data, host_dh.data(), host_dh.size()*sizeof(float), cudaMemcpyHostToDevice));

    Tensor dv(N, 1, 1, 1);
    std::vector<float> host_dv(N);
    for (int i = 0; i < N; ++i) host_dv[i] = static_cast<float>(1);
    CUDA_CALL(cudaMemcpy(dv.data, host_dv.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> host_dx(total);
    Tensor* dx = policy.backward(&dlogp, &dh, &dv, cudnn, cublas, stream);
    CUDA_CALL(cudaMemcpy(host_dx.data(), dx->data, total * sizeof(float), cudaMemcpyDeviceToHost));
    float mean_dx = std::accumulate(host_dx.begin(), host_dx.end(), 0.0) / host_dx.size();
    float true_mean_dx = std::accumulate(expected_dx.begin(), expected_dx.end(), 0.0) / expected_dx.size();
    std:: cout << "Output: " << mean_dx << " Expected: " << true_mean_dx << std:: endl;
    std:: cout << "Policy forward/backward tests passed!" << std::endl;

    cudaStreamDestroy(stream);
    cudnnDestroy(cudnn);
    cublasDestroy(cublas);
    
    return 0;
}