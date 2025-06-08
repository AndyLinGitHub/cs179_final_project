# include "utils.cuh"

__global__ void unfold_circular_kernel(const float* x, float* y, int H, int W, int C, int K){
    int patch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (patch_id >= H * W) return;

    int h0 = patch_id / W;
    int w0 = patch_id % W;
    int radius = K / 2; // assumes K is odd

    const int patch_stride = C * K * K; // elements per patch
    const int chan_stride  = K * K;
    //const int row_stride   = W; // inside one channel of x

    // iterate over channels, then kernel rows/cols
    for (int c = 0; c < C; ++c) {
        float* y_patch_c = y + patch_id * patch_stride + c * chan_stride;

        for (int kh = 0; kh < K; ++kh) {
            int in_h = (h0 + kh - radius + H) % H; // wrap vertically

            for (int kw = 0; kw < K; ++kw) {
                int in_w = (w0 + kw - radius + W) % W; // wrap horizontally

                int y_offset = kh * K + kw; // row-major in patch
                int x_offset = ((c * H + in_h) * W) + in_w;

                y_patch_c[y_offset] = x[x_offset];
            }
        }
    }
}

__global__ void new_image_kernel(const float* in, float* out, int H, int W, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N   = C * H * W;
    if (idx >= N) return;

    int c  = idx / (H * W);
    int hw = idx - c * H * W;
    int h  = hw / W;
    int w  = hw - h * W;

    const int i = (h * W + w) * C + c;
    float temp = in[i]*2.0f - 1.0f + out[idx];
    //out[idx] = temp;
    if (temp >= 1.0) out[idx] = 1.0f;
    else if (temp <= 0) out[idx] = 0.0f;
    else out[idx] = temp;
}

__global__ void loss_function(const float* A, const float* B, float* out, int C, int H, int W){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = idx / (H * W);

    float diff = (A[idx] - B[idx])/2;

    const int out_idx = ((h * W) + w) * C + c;
    out[out_idx] = diff;
}

