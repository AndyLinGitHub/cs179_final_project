#ifndef CONV_LAYER_H
#define CONV_LAYER_H
#include "layer.h"
#include <cudnn.h>

class ConvLayer : public Layer {
public:
    ConvLayer(int inC, int outC, int k, int stride = 1, int pad = 0);
    ~ConvLayer();

    Tensor* forward(Tensor* x, cudnnHandle_t handle) override;
    Tensor* backward(Tensor* dy, cudnnHandle_t handle) override;
    std::vector<Parameter*> params() override { return {&W, &b}; }

private:
    int inChannels, outChannels, kernelSize, stride, padding;

    cudnnFilterDescriptor_t filterDesc{};
    cudnnConvolutionDescriptor_t convDesc{};
    cudnnTensorDescriptor_t biasDesc{};

    Parameter W; // weights [outC, inC, k, k]
    Parameter b; // bias   [1, outC,1,1]

    Tensor* y = nullptr; // cached output
    Tensor* x_cache = nullptr; // cached input
};

#endif // CONV_LAYER_H