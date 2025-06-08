#include "tensor.h"
#include <algorithm>

__global__ void unfold_circular_kernel(const float* x, float* y, int H, int W, int C, int K);

__global__ void new_image_kernel(const float* in, float* out, int H, int W, int D);

__global__ void loss_function(const float* A, const float* B, float* out, int C, int H, int W);
