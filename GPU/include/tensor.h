#include <cstddef>
#include <vector>

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

struct Tensor {
    float* data = nullptr;
    cudnnTensorDescriptor_t desc{};
    std::vector<int> dims; // NCHW

    Tensor(int n, int c, int h, int w, bool allocate = true);
    ~Tensor();

    size_t bytes() const;
    void zero();
    int n() const { return dims[0]; }
    int c() const { return dims[1]; }
    int h() const { return dims[2]; }
    int w() const { return dims[3]; }
};
