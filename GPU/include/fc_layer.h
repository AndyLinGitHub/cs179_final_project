#pragma once
#include "layer.h"
#include <cublas_v2.h>

class FullyConnected : public Layer {
public:
    FullyConnected(int in_features, int out_features);
    ~FullyConnected() override;

    Tensor* forward (Tensor* x, cudnnHandle_t hCudnn) override;
    Tensor* backward(Tensor* dy, cudnnHandle_t hCudnn) override;
    std::vector<Parameter*> params() override { return { &W, &b }; }

private:
    int in_f_, out_f_;

    Parameter  W;          // (out_f_ , in_f_) -- row major
    Parameter  b;          // (1, out_f_, 1, 1)
    Tensor* y = nullptr; // cached output
    Tensor* x_cache = nullptr; // cached input

    cublasHandle_t hBlas_;
};