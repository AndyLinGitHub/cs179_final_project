#pragma once

#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "config.h"

struct Parameter {
    float* value; // device memory
    float* grad; // gradient
    float* m; // Adam first moment
    float* v; // Adam second moment
    size_t numel; // elements count

    Parameter(size_t num);
    ~Parameter();
};