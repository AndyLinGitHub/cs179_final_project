#ifndef PARAMETER_H
#define PARAMETER_H
#include <cuda_runtime.h>
#include "helper_cuda.h"

struct Parameter {
    float* value;      // device memory
    float* grad;       // gradient (same size)
    float* m;          // Adam first moment
    float* v;          // Adam second moment
    size_t numel;      // elements count

    Parameter(size_t num);
    ~Parameter();
};

#endif // PARAMETER_H