#include "parameter.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    const size_t N = 16;   // small tensor length for testing

    {
        Parameter p(N);

        // Pointers should not be null after allocation
        assert(p.value && "value pointer is null");
        assert(p.grad  && "grad pointer is null");
        assert(p.m     && "m pointer is null");
        assert(p.v     && "v pointer is null");

        // m and v are zero‑initialised in the constructor
        std::vector<float> host(N);
        CUDA_CALL(cudaMemcpy(host.data(), p.m, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (float f : host) assert(f == 0.0f && "m not zero‑initialised");
        CUDA_CALL(cudaMemcpy(host.data(), p.v, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (float f : host) assert(f == 0.0f && "v not zero‑initialised");

        // Sanity write/read for grad buffer (set to 1.0f)
        std::vector<float> ones(N, 1.0f);
        CUDA_CALL(cudaMemcpy(p.grad, ones.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> host_back(N, 0.0f);
        CUDA_CALL(cudaMemcpy(host_back.data(), p.grad, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (float f : host_back) assert(f == 1.0f && "grad write/read failed");
    } // p goes out of scope—destructor frees memory (leak‑san if compiled with cuda‑memcheck)

    std::cout << "Parameter basic tests passed!" << std::endl;
    return 0;
}