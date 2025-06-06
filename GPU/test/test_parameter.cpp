#include <cassert>
#include <iostream>
#include <vector>
#include "parameter.h"

int main() {
    const size_t N = 1024;  
    Parameter p(N);

    // Pointers should not be null after allocation
    assert(p.value != nullptr);
    assert(p.grad != nullptr);
    assert(p.m != nullptr);
    assert(p.v != nullptr);

    // m and v need to be zero
    std::vector<float> host(N);
    CUDA_CALL(cudaMemcpy(host.data(), p.m, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (float f : host) assert(f == 0.0f);
    CUDA_CALL(cudaMemcpy(host.data(), p.v, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (float f : host) assert(f == 0.0f);

    // Test write/read for grad buffer
    std::vector<float> ones(N, 1.0f);
    CUDA_CALL(cudaMemcpy(p.grad, ones.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(host.data(), p.grad, N * sizeof(float), cudaMemcpyDeviceToHost));
    for (float f : host) assert(f == 1.0f);

    std::cout << "Parameter tests passed!" << std::endl;
    
    return 0;
}