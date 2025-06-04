#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include "tensor.h"
#include "parameter.h"

class Layer {
public:
    virtual Tensor* forward(Tensor* x, cudnnHandle_t handle) = 0;
    virtual Tensor* backward(Tensor* dy, cudnnHandle_t handle) = 0;
    virtual std::vector<Parameter*> params() = 0;
    virtual ~Layer() = default;
};

#endif // LAYER_H