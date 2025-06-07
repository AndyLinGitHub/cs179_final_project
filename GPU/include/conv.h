#include "tensor.h"
#include "parameter.h"

class Conv {
public:
    Conv(int in_channel, int out_channel, int k, int stride, int pad);
    ~Conv();

    Tensor* forward(Tensor* x, cudnnHandle_t handle, cudaStream_t stream);
    Tensor* backward(Tensor* dy, cudnnHandle_t handle, cudaStream_t stream);
    std::vector<Parameter*> params() { return {&W, &b}; }

private:
    int in_channel_, out_channel_, k_, stride_, pad_;

    cudnnFilterDescriptor_t filterDesc{};
    cudnnConvolutionDescriptor_t convDesc{};
    cudnnTensorDescriptor_t biasDesc{};

    Parameter W; // weights [Out Channel, In Channel, k, k]
    Parameter b; // bias [1, Out Channel, 1, 1]

    Tensor* x_cache = nullptr;
    Tensor* dx = nullptr;
    Tensor* y = nullptr;
};