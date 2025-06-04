#include "relu_layer.h"

ReluLayer::ReluLayer() {
    CUDNN_CALL(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
}

ReluLayer::~ReluLayer() {
    if (y) delete y;
    cudnnDestroyActivationDescriptor(actDesc);
}

Tensor* ReluLayer::forward(Tensor* x, cudnnHandle_t handle) {
    x_cache = x;
    if (!y) y = new Tensor(x->n(), x->c(), x->h(), x->w());
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnActivationForward(handle, actDesc, &alpha, x->desc, x->data, &beta, y->desc, y->data));
    return y;
}

Tensor* ReluLayer::backward(Tensor* dy, cudnnHandle_t handle) {
    Tensor* dx = new Tensor(x_cache->n(), x_cache->c(), x_cache->h(), x_cache->w());
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnActivationBackward(handle, actDesc, &alpha, y->desc, y->data, dy->desc, dy->data, x_cache->desc, x_cache->data, &beta, dx->desc, dx->data));
    return dx;
}