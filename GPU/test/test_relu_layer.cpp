 #include "relu_layer.h"
 #include <cassert>
 #include <cmath>
 #include <iostream>
 #include <vector>
 
 int main() {
     // cuDNN handle ------------------------------------------------------------
     cudnnHandle_t cudnn;
     CUDNN_CALL(cudnnCreate(&cudnn));
 
     // Build input tensor ------------------------------------------------------
     const int N = 2, C = 2, H = 5, W = 5;
     Tensor x(N, C, H, W);
     std::vector<float> h_x(100);
    for (int i = 0; i < 100; ++i) h_x[i] = static_cast<float>(i - 50);
     CUDA_CALL(cudaMemcpy(x.data, h_x.data(), h_x.size()*sizeof(float), cudaMemcpyHostToDevice));
 
     ReluLayer relu; 
     Tensor* y = relu.forward(&x, cudnn);
 
     std::vector<float> h_y(h_x.size());
     CUDA_CALL(cudaMemcpy(h_y.data(), y->data, h_y.size()*sizeof(float), cudaMemcpyDeviceToHost));
 
     const std::vector<float> expected_y = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0
     };
     for (size_t i = 0; i < h_y.size(); ++i) {
         assert(std::fabs(h_y[i] - expected_y[i]) < 1e-6f && "ReLU forward mismatch");
     }
 
     // Backward ----------------------------------------------------------------
     Tensor dY(N, C, H, W);
     std::vector<float> h_dy(h_x.size(), 1.0f);
     CUDA_CALL(cudaMemcpy(dY.data, h_dy.data(), h_dy.size()*sizeof(float),cudaMemcpyHostToDevice));
 
     Tensor* dX = relu.backward(&dY, cudnn);
     std::vector<float> h_dx(h_x.size());
     CUDA_CALL(cudaMemcpy(h_dx.data(), dX->data, h_dx.size()*sizeof(float), cudaMemcpyDeviceToHost));
 
     const std::vector<float> expected_dx = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
     };

     for (size_t i = 0; i < h_dx.size(); ++i) {
         assert(std::fabs(h_dx[i] - expected_dx[i]) < 1e-6f && "ReLU backward mismatch");
     }
 
     std::cout << "ReluLayer forward/backward tests passed!" << std::endl;
 
     cudnnDestroy(cudnn);
     return 0;
 }
 