#include "tensor.h"
#include "parameter.h"
#include "config.h"

class FC {
public:
    FC(int in_feature, int out_feature);
    ~FC();

    Tensor* forward (Tensor* x, cublasHandle_t handle, cudaStream_t stream);
    Tensor* backward(Tensor* dy, cublasHandle_t handle, cudaStream_t stream, bool flatten = false);
    std::vector<Parameter*> params() { return { &W, &b }; }

private:
    int in_feature_, out_feature_;

    // Row major
    Parameter  W; // (Out Feature , In Feature)
    Parameter  b; // (1, Out Feature, 1, 1)

    Tensor* x_cache = nullptr;
    Tensor* dx = nullptr;
    Tensor* y = nullptr;
};