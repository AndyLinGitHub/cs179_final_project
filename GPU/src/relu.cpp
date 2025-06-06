#include "relu.h"

Relu:: Relu(cudaStream_t stream) : stream_(stream) {
    CUDNN_CALL(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
}

Relu:: ~Relu() {
    // Since x is passed from outside the class, we should not delete it here.
    // if (x_cache) delete x_cache;
    
    if (dx) delete dx;
    if (y) delete y;
    cudnnDestroyActivationDescriptor(actDesc);
}

Tensor* Relu::forward(Tensor* x, cudnnHandle_t handle) {
    CUDNN_CALL(cudnnSetStream(handle, stream_));

    x_cache = x;
    if (!y) y = new Tensor(x->n(), x->c(), x->h(), x->w());

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnActivationForward(handle, actDesc, &alpha, x->desc, x->data, &beta, y->desc, y->data));
    
    return y;
}

Tensor* Relu::backward(Tensor* dy, cudnnHandle_t handle) {
    CUDNN_CALL(cudnnSetStream(handle, stream_));
    
    if (!dx) dx = new Tensor(x_cache->n(), x_cache->c(), x_cache->h(), x_cache->w());
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnActivationBackward(handle, actDesc, &alpha, y->desc, y->data, dy->desc, dy->data, x_cache->desc, x_cache->data, &beta, dx->desc, dx->data));
    
    return dx;
}