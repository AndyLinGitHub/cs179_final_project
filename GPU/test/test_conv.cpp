#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include "conv.h"

int main() {
    std::ifstream expected_y_file("conv_y.txt");
    std::ifstream expected_dx_file("conv_dx.txt");
    std::ifstream expected_dW_file("conv_dW.txt");
    std::ifstream expected_db_file("conv_db.txt");

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

    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    const int N = 8, C = 2, H = 5, W = 5;
    const int total = N * C * H * W;
    Tensor x(N, C, H, W);

    // Fill input with 0, 1, 2, 3, 4 ...
    std::vector<float> host_x(total);
    for (int i = 0; i < total; ++i) host_x[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(x.data, host_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    const int in_channel = 2, out_channel = 4, k = 3, stride = 1, pad = 0;
    const int total_weights = in_channel * out_channel * k * k;
    Conv conv = Conv(in_channel, out_channel, k, stride, pad);

    // Override weights and bias with 0, 1, 2, 3, 4 ...
    std::vector<float> host_weights(total_weights);
    for (int i = 0; i < total_weights; ++i) host_weights[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(conv.params()[0]->value, host_weights.data(), total_weights * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> host_bias(out_channel);
    for (int i = 0; i < out_channel; ++i) host_bias[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(conv.params()[1]->value, host_bias.data(), out_channel * sizeof(float), cudaMemcpyHostToDevice));

    // Forward
    const int out_N = 8, out_C = 4, out_H = 3, out_W = 3;
    const int out_total = out_N * out_C * out_H * out_W;

    Tensor* y = conv.forward(&x, cudnn, stream);
    std::vector<float> host_y(out_total);
    CUDA_CALL(cudaMemcpy(host_y.data(), y->data, out_total * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < out_total; ++i) {
        assert(std::fabs(host_y[i] - expected_y[i]) < 1e-4f);
    }

    // Backward
    Tensor dy(out_N, out_C, out_H, out_W);
    std::vector<float> host_dy(out_total);
    for (int i = 0; i < out_total; ++i) host_dy[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(dy.data, host_dy.data(), out_total * sizeof(float), cudaMemcpyHostToDevice));

    Tensor* dx = conv.backward(&dy, cudnn, stream);
    std::vector<float> host_dx(total);
    CUDA_CALL(cudaMemcpy(host_dx.data(), dx->data, total * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> host_dW(total_weights);
    CUDA_CALL(cudaMemcpy(host_dW.data(), conv.params()[0]->grad, total_weights * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> host_db(out_channel);
    CUDA_CALL(cudaMemcpy(host_db.data(), conv.params()[1]->grad, out_channel * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; ++i) {
        assert(std::fabs(host_dx[i] - expected_dx[i]) < 1e-4f);
    }
    
    for (int i = 0; i < total_weights; ++i) {
        assert(std::fabs(host_dW[i] - expected_dW[i]) < 1e-4f);
    }

    for (int i = 0; i < out_channel; ++i) {
        assert(std::fabs(host_db[i] - expected_db[i]) < 1e-4f);
    }
   
    std::cout << "Conv forward/backward tests passed!" << std::endl;
    
    cudaStreamDestroy(stream);
    cudnnDestroy(cudnn);
    
    return 0;
}