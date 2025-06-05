#include "fc_layer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    const int N = 2, C = 8, H = 1, W = 1;
    Tensor x(N, C, H, W);
    std::vector<float> h_x(16);
    for (int i = 0; i < h_x.size(); ++i) h_x[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(x.data, h_x.data(), h_x.size()*sizeof(float), cudaMemcpyHostToDevice));

    FullyConnected fc(8, 16);

    // Override weights & bias with deterministic pattern
    std::vector<float> h_kernel(128);
    for (int i = 0; i < 128; ++i) h_kernel[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(fc.params()[0]->value, h_kernel.data(), 128 * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> bias = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    CUDA_CALL(cudaMemcpy(fc.params()[1]->value, bias.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));

    Tensor* y = fc.forward(&x, cudnn);
    std::vector<float> h_y(32);
     CUDA_CALL(cudaMemcpy(h_y.data(), y->data, h_y.size()*sizeof(float), cudaMemcpyDeviceToHost));
 
     const std::vector<float> expected_y = {
      140.0, 365.0, 590.0, 815.0, 1040.0, 1265.0, 1490.0, 1715.0, 1940.0, 2165.0, 2390.0, 2615.0, 2840.0, 3065.0, 3290.0, 3515.0, 364.0, 1101.0, 1838.0, 2575.0, 3312.0, 4049.0, 4786.0, 5523.0, 6260.0, 6997.0, 7734.0, 8471.0, 9208.0, 9945.0, 10682.0, 11419.0
     };
     for (size_t i = 0; i < h_y.size(); ++i) {
        //std:: cout << h_y[i] << std:: endl;
        assert(std::fabs(h_y[i] - expected_y[i]) < 1e-6f && "FC forward mismatch");
     }

    Tensor dy(2, 16, 1, 1);
    std::vector<float> h_dy(32, 1.0f);
    CUDA_CALL(cudaMemcpy(dy.data, h_dy.data(), 32 * sizeof(float), cudaMemcpyHostToDevice));

    Tensor* dx = fc.backward(&dy, cudnn);
    std::vector<float> h_gW(128);
    CUDA_CALL(cudaMemcpy(h_gW.data(), fc.params()[0]->grad, 128 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> h_gb(16);
    CUDA_CALL(cudaMemcpy(h_gb.data(), fc.params()[1]->grad, 16 * sizeof(float), cudaMemcpyDeviceToHost));

    // Expected gradients
    const std::vector<float> expected_gW = {
      8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0
   };

    for (int i = 0; i < 128; ++i) {
        assert(std::fabs(h_gW[i] - expected_gW[i]) < 1e-4f && "W.grad mismatch");
    }

    for (int i = 0; i < 16; ++i) {
        assert(std::fabs(h_gb[i] - 2) < 1e-4f && "W.grad mismatch");
    }

    std::vector<float> h_dx(16);
    CUDA_CALL(cudaMemcpy(h_dx.data(), dx->data, 16 * sizeof(float), cudaMemcpyDeviceToHost));
    const std::vector<float> expected_dx = {
      960.0, 976.0, 992.0, 1008.0, 1024.0, 1040.0, 1056.0, 1072.0, 960.0, 976.0, 992.0, 1008.0, 1024.0, 1040.0, 1056.0, 1072.0
    };
    for (int i = 0; i < 16; ++i) {
        assert(std::fabs(h_dx[i] - expected_dx[i]) < 1e-4f && "dx mismatch");
    }
   

    std::cout << "FC forward/backward tests passed!" << std::endl;
    
    cudnnDestroy(cudnn);
    return 0;


}