#include <math_constants.h>

#include "config.h"
#include "tensor.h"
#include "parameter.h"

class SoftPlusAdd1 {
public:
    SoftPlusAdd1();
    ~SoftPlusAdd1();

    Tensor* forward(Tensor* x, cudaStream_t stream);
    Tensor* backward(Tensor* dy, cudaStream_t stream);
    std::vector<Parameter*> params() { return {}; }

private:
    Tensor* x_cache = nullptr;
    Tensor* dx = nullptr;
    Tensor* y = nullptr;
};