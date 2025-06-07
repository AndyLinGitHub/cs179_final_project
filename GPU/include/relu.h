#include "tensor.h"
#include "parameter.h"

class Relu {
public:
    Relu();
    ~Relu();

    Tensor* forward(Tensor* x, cudnnHandle_t handle, cudaStream_t stream);
    Tensor* backward(Tensor* dy, cudnnHandle_t handle, cudaStream_t stream);
    std::vector<Parameter*> params() { return {}; }

private:
    cudnnActivationDescriptor_t actDesc{};
    
    Tensor* x_cache = nullptr;
    Tensor* dx = nullptr;
    Tensor* y = nullptr;
};