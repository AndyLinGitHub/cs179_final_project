// test/test_tensor.cpp
#include "tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// ── simple kernel to fill a device array ────────────────────────
__global__ void fill_kernel(float* data, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

int main() {
    constexpr int N = 2, C = 3, H = 4, W = 5;
    Tensor t(N, C, H, W);                     // allocate device memory
    const int total = N * C * H * W;
    const float initVal = 3.14f;

    // 1) fill with known value on GPU
    int threads = 256, blocks = (total + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(t.data, total, initVal);
    CUDA_CALL(cudaDeviceSynchronize());

    // 2) verify host-side that every element == initVal
    std::vector<float> host(total);
    CUDA_CALL(cudaMemcpy(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    for (float v : host) {
        assert(std::fabs(v - initVal) < 1e-6f);
    }

    // 3) zero-out and 4) verify all zeros
    t.zero();
    CUDA_CALL(cudaMemcpy(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    for (float v : host) {
        assert(v == 0.0f);
    }

    std::cout << "Tensor fill / zero tests passed!\n";
    return 0;
}
