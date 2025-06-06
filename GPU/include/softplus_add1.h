#include <math_constants.h>

#include "config.h"
#include "tensor.h"
#include "parameter.h"

class SoftPlusAdd1 {
public:
    SoftPlusAdd1(cudaStream_t stream);
    ~SoftPlusAdd1();

    Tensor* forward(Tensor* x);
    Tensor* backward(Tensor* dy);
    std::vector<Parameter*> params() { return {}; }

private:
    cudaStream_t stream_{0};

    Tensor* x_cache = nullptr;
    Tensor* dx = nullptr;
    Tensor* y = nullptr;
};