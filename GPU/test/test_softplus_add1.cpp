 #include <cassert>
 #include <cmath>
 #include <iostream>
 #include <fstream>
 #include <vector>
 #include "softplus_add1.h"

 int main() {
    std::ifstream expected_y_file("softplus_add1_y.txt");
    std::ifstream expected_dx_file("softplus_add1_dx.txt");

    std::vector<float> expected_y;
    std::vector<float> expected_dx;
    float value = 0;

    while (expected_y_file >> value) expected_y.push_back(value);
    while (expected_dx_file >> value) expected_dx.push_back(value);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int N = 1024, C = 4, H = 1, W = 1;
    const int total = N*C*H*W;
    Tensor x(N, C, H, W);

    std::vector<float> host_x(total);
    for (int i = 0; i < total; ++i) host_x[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(x.data, host_x.data(), host_x.size()*sizeof(float), cudaMemcpyHostToDevice));
 
    SoftPlusAdd1 spa1 = SoftPlusAdd1(); 

    // Forward
    Tensor* y = spa1.forward(&x, stream);
 
    std::vector<float> host_y(host_x.size());
    CUDA_CALL(cudaMemcpy(host_y.data(), y->data, host_y.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host_y.size(); ++i) {
        assert(std::fabs(host_y[i] - expected_y[i]) < 1e-5f);
    }
 
    // Backward
    Tensor dy(N, C, H, W);
    std::vector<float> host_dy(host_x.size());
    for (int i = 0; i < total; ++i) host_dy[i] = static_cast<float>(i);

    CUDA_CALL(cudaMemcpy(dy.data, host_dy.data(), host_dy.size()*sizeof(float),cudaMemcpyHostToDevice));
 
    Tensor* dx = spa1.backward(&dy, stream);
    std::vector<float> host_dx(host_x.size());
    CUDA_CALL(cudaMemcpy(host_dx.data(), dx->data, host_dx.size()*sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host_dx.size(); ++i) {
        assert(std::fabs(host_dx[i] - expected_dx[i]) < 1e-5f);
    }

     std::cout << "SoftPlusAdd1 forward/backward tests passed!" << std::endl;
 
     cudaStreamDestroy(stream);

     return 0;
 }
 