#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include "fc.h"

int main() {
    std::ifstream expected_y_file("fc_y.txt");
    std::ifstream expected_dx_file("fc_dx.txt");
    std::ifstream expected_dW_file("fc_dW.txt");
    std::ifstream expected_db_file("fc_db.txt");

    std::vector<float> expected_y;
    std::vector<float> expected_dx;
    std::vector<float> expected_dW;
    std::vector<float> expected_db;
    float value = 0;

    while (expected_y_file >> value) expected_y.push_back(value);
    while (expected_dx_file >> value) expected_dx.push_back(value);
    while (expected_dW_file >> value) expected_dW.push_back(value);
    while (expected_db_file >> value) expected_db.push_back(value);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t cublas;
    CUBLAS_CALL(cublasCreate(&cublas));

    const int N = 2, C = 8, H = 1, W = 1;
    const int total = N * C * H * W;
    const int in_feature = 8, out_feature = 16;
    const int total_weights = in_feature * out_feature;
    Tensor x(N, C, H, W);

    // Fill input with 0, 1, 2, 3, 4 ...
    std::vector<float> host_x(total);
    for (int i = 0; i < host_x.size(); ++i) host_x[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(x.data, host_x.data(), host_x.size()*sizeof(float), cudaMemcpyHostToDevice));

    FC fc(in_feature, out_feature);

    // Override weights and bias with 0, 1, 2, 3, 4 ...
    std::vector<float> host_weights(total_weights);
    for (int i = 0; i < total_weights; ++i) host_weights[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(fc.params()[0]->value, host_weights.data(), total_weights * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> host_bias(out_feature);
    for (int i = 0; i < out_feature; ++i) host_bias[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(fc.params()[1]->value, host_bias.data(), out_feature * sizeof(float), cudaMemcpyHostToDevice));

    // Forward
    Tensor* y = fc.forward(&x, cublas, stream);
    std::vector<float> host_y(N*out_feature);
    CUDA_CALL(cudaMemcpy(host_y.data(), y->data, host_y.size()*sizeof(float), cudaMemcpyDeviceToHost));
 
    for (size_t i = 0; i < host_y.size(); ++i) {
        assert(std::fabs(host_y[i] - expected_y[i]) < 1e-6f);
    }

    // Backward
    Tensor dy(N, out_feature, 1, 1);
    std::vector<float> host_dy(N*out_feature);
    for (int i = 0; i < host_dy.size(); ++i) host_dy[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(dy.data, host_dy.data(), host_dy.size() * sizeof(float), cudaMemcpyHostToDevice));

    Tensor* dx = fc.backward(&dy, cublas, stream);
    std::vector<float> host_dx(total);
    CUDA_CALL(cudaMemcpy(host_dx.data(), dx->data, total * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> host_dW(total_weights);
    CUDA_CALL(cudaMemcpy(host_dW.data(), fc.params()[0]->grad, total_weights * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> host_db(out_feature);
    CUDA_CALL(cudaMemcpy(host_db.data(), fc.params()[1]->grad, out_feature * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; ++i) {
        assert(std::fabs(host_dx[i] - expected_dx[i]) < 1e-4f);
    }

    for (int i = 0; i < total_weights; ++i) {
        assert(std::fabs(host_dW[i] - expected_dW[i]) < 1e-4f);
    }

    for (int i = 0; i < out_feature; ++i) {
        assert(std::fabs(host_db[i] - expected_db[i]) < 1e-4f);
    }

    std::cout << "FC forward/backward tests passed!" << std::endl;

    cudaStreamDestroy(stream);
    cublasDestroy(cublas);

    return 0;


}