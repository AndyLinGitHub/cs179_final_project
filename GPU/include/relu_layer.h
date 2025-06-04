#ifndef RELU_LAYER_H
#define RELU_LAYER_H
#include "layer.h"

class ReluLayer : public Layer {
public:
    ReluLayer();
    ~ReluLayer();

    Tensor* forward(Tensor* x, cudnnHandle_t handle) override;
    Tensor* backward(Tensor* dy, cudnnHandle_t handle) override;
    std::vector<Parameter*> params() override { return {}; }

private:
    cudnnActivationDescriptor_t actDesc{};
    Tensor* y = nullptr;
    Tensor* x_cache = nullptr;
};

#endif // RELU_LAYER_H