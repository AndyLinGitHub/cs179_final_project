#include "tensor.h"
#include "parameter.h"

class Relu {
public:
    Relu(cudaStream_t stream);
    ~Relu();

    Tensor* forward(Tensor* x, cudnnHandle_t handle);
    Tensor* backward(Tensor* dy, cudnnHandle_t handle);
    std::vector<Parameter*> params() { return {}; }

private:
    cudnnActivationDescriptor_t actDesc{};
    cudaStream_t stream_{0};
    
    Tensor* x_cache = nullptr;
    Tensor* dx = nullptr;
    Tensor* y = nullptr;
};