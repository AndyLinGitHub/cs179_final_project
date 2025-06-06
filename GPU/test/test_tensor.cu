#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "tensor.h"

__global__ void fill_kernel(float* data, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

int main() {
    int N = 2, C = 3, H = 4, W = 5;
    Tensor t(N, C, H, W);

    const int total = N * C * H * W;
    const float value = 4.2f;

    // fill with known value on GPU
    int threads = 256, blocks = (total + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(t.data, total, value);

    // verify values
    std::vector<float> host(total);
    CUDA_CALL(cudaMemcpy(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    for (float v : host) {
        assert(std::fabs(v - value) < 1e-6f);
    }

    // zero-out
    t.zero();
    CUDA_CALL(cudaMemcpy(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    for (float v : host) {
        assert(v == 0.0f);
    }

    std::cout << "Tensor tests passed!\n";

    return 0;
}
