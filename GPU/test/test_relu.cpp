 #include "relu.h"
 #include <cassert>
 #include <cmath>
 #include <iostream>
 #include <vector>
 
 int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));
 
    const int N = 2, C = 2, H = 5, W = 5;
    const int total = N*C*H*W;
    Tensor x(N, C, H, W);
     
    std::vector<float> host_x(total);
    for (int i = 0; i < total; ++i) host_x[i] = static_cast<float>(i - 50);
    CUDA_CALL(cudaMemcpy(x.data, host_x.data(), host_x.size()*sizeof(float), cudaMemcpyHostToDevice));
 
    Relu relu = Relu(stream);

    // Forward
    Tensor* y = relu.forward(&x, cudnn);
    std::vector<float> host_y(host_x.size());
    CUDA_CALL(cudaMemcpy(host_y.data(), y->data, host_y.size()*sizeof(float), cudaMemcpyDeviceToHost));
 
    float expected_y = 0.0f;
    for (size_t i = 0; i < host_y.size(); ++i) {
        expected_y = std::max(host_x[i], 0.0f);
        assert(std::fabs(host_y[i] - expected_y) == 0);
    }
 
    // Backward
    Tensor dy(N, C, H, W);
    std::vector<float> host_dy(host_x.size());
    for (int i = 0; i < total; ++i) host_dy[i] = static_cast<float>(i - 50);
    CUDA_CALL(cudaMemcpy(dy.data, host_dy.data(), host_dy.size()*sizeof(float),cudaMemcpyHostToDevice));
 
    Tensor* dx = relu.backward(&dy, cudnn);
    std::vector<float> host_dx(host_x.size());
    CUDA_CALL(cudaMemcpy(host_dx.data(), dx->data, host_dx.size()*sizeof(float), cudaMemcpyDeviceToHost));
 
    float expected_dx = 0.0f;
    for (size_t i = 0; i < host_dx.size(); ++i) {
        expected_dx = host_dy[i]* (host_x[i] > 0.0f ? 1.0f : 0.0f);
        assert(std::fabs(host_dx[i] - expected_dx) == 0);
    }
 
     std::cout << "Relu forward/backward tests passed!" << std::endl;
 
     cudaStreamDestroy(stream);
     cudnnDestroy(cudnn);

     return 0;
 }
 