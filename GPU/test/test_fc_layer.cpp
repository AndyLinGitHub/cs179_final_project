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
        2240.0, 2269.0, 2298.0, 2327.0, 2356.0, 2385.0, 2414.0, 2443.0, 2472.0, 2501.0, 2530.0, 2559.0, 2588.0, 2617.0, 2646.0, 2675.0, 5824.0, 5917.0, 6010.0, 6103.0, 6196.0, 6289.0, 6382.0, 6475.0, 6568.0, 6661.0, 6754.0, 6847.0, 6940.0, 7033.0, 7126.0, 7219.0
     };
     for (size_t i = 0; i < h_y.size(); ++i) {
        std:: cout << h_y[i] << std:: endl;
         //assert(std::fabs(h_y[i] - expected_y[i]) < 1e-6f && "FC forward mismatch");
     }


}