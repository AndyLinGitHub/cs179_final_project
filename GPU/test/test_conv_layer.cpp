#include "conv_layer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // ── cuDNN handle ──────────────────────────────────────────
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // ── Test objects ─────────────────────────────────────────
    const int N = 2, C = 2, H = 5, W = 5;
    Tensor input(N, C, H, W);

    // Fill input with 1..25 (row‑major)
    std::vector<float> h_input(100);
    for (int i = 0; i < 100; ++i) h_input[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(input.data, h_input.data(), 100 * sizeof(float), cudaMemcpyHostToDevice));

    ConvLayer conv(/*inC=*/2, /*outC=*/4, /*k=*/3, /*stride=*/1, /*pad=*/0);

    // Override weights & bias with deterministic pattern
    std::vector<float> h_kernel(72);
    for (int i = 0; i < 72; ++i) h_kernel[i] = static_cast<float>(i);
    CUDA_CALL(cudaMemcpy(conv.params()[0]->value, h_kernel.data(), 72 * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> bias = {0, 1, 2, 3};
    CUDA_CALL(cudaMemcpy(conv.params()[1]->value, bias.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));

    // ── Forward ──────────────────────────────────────────────
    Tensor* y = conv.forward(&input, cudnn);
    std::vector<float> h_y(72);
    CUDA_CALL(cudaMemcpy(h_y.data(), y->data, 72 * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<float> expected_y = {
        4035.0, 4188.0, 4341.0, 4800.0, 4953.0, 5106.0, 5565.0, 5718.0, 5871.0, 10030.0, 10507.0, 10984.0, 12415.0, 12892.0, 13369.0, 14800.0, 15277.0, 15754.0, 16025.0, 16826.0, 17627.0, 20030.0, 20831.0, 21632.0, 24035.0, 24836.0, 25637.0, 22020.0, 23145.0, 24270.0, 27645.0, 28770.0, 29895.0, 33270.0, 34395.0, 35520.0, 11685.0, 11838.0, 11991.0, 12450.0, 12603.0, 12756.0, 13215.0, 13368.0, 13521.0, 33880.0, 34357.0, 34834.0, 36265.0, 36742.0, 37219.0, 38650.0, 39127.0, 39604.0, 56075.0, 56876.0, 57677.0, 60080.0, 60881.0, 61682.0, 64085.0, 64886.0, 65687.0, 78270.0, 79395.0, 80520.0, 83895.0, 85020.0, 86145.0, 89520.0, 90645.0, 91770.0
    };

    for (int i = 0; i < 72; ++i) {
        //std:: cout << h_y[i] << std:: endl;
        assert(std::fabs(h_y[i] - expected_y[i]) < 1e-4f && "forward output mismatch");
    }

    // ── Backward ─────────────────────────────────────────────
    Tensor dy(2, 4, 3, 3);
    std::vector<float> h_dy(72, 1.0f);
    CUDA_CALL(cudaMemcpy(dy.data, h_dy.data(), 72 * sizeof(float), cudaMemcpyHostToDevice));

    Tensor* dx = conv.backward(&dy, cudnn);
    std::vector<float> h_gW(72);
    CUDA_CALL(cudaMemcpy(h_gW.data(), conv.params()[0]->grad, 72 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> h_gb(4);
    CUDA_CALL(cudaMemcpy(h_gb.data(), conv.params()[1]->grad, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Expected gradients
    const std::vector<float> expected_gW = {
        558.0, 576.0, 594.0, 648.0, 666.0, 684.0, 738.0, 756.0, 774.0, 1008.0, 1026.0, 1044.0, 1098.0, 1116.0, 1134.0, 1188.0, 1206.0, 1224.0, 558.0, 576.0, 594.0, 648.0, 666.0, 684.0, 738.0, 756.0, 774.0, 1008.0, 1026.0, 1044.0, 1098.0, 1116.0, 1134.0, 1188.0, 1206.0, 1224.0, 558.0, 576.0, 594.0, 648.0, 666.0, 684.0, 738.0, 756.0, 774.0, 1008.0, 1026.0, 1044.0, 1098.0, 1116.0, 1134.0, 1188.0, 1206.0, 1224.0, 558.0, 576.0, 594.0, 648.0, 666.0, 684.0, 738.0, 756.0, 774.0, 1008.0, 1026.0, 1044.0, 1098.0, 1116.0, 1134.0, 1188.0, 1206.0, 1224.0
    };

    for (int i = 0; i < 72; ++i) {
        assert(std::fabs(h_gW[i] - expected_gW[i]) < 1e-4f && "W.grad mismatch");
    }

    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(h_gb[i] - 18) < 1e-4f && "W.grad mismatch");
    }

    std::vector<float> h_dx(100);
    CUDA_CALL(cudaMemcpy(h_dx.data(), dx->data, 100 * sizeof(float), cudaMemcpyDeviceToHost));
    const std::vector<float> expected_dx = {
        108.0, 220.0, 336.0, 228.0, 116.0, 228.0, 464.0, 708.0, 480.0, 244.0, 360.0, 732.0, 1116.0, 756.0, 384.0, 252.0, 512.0, 780.0, 528.0, 268.0, 132.0, 268.0, 408.0, 276.0, 140.0, 144.0, 292.0, 444.0, 300.0, 152.0, 300.0, 608.0, 924.0, 624.0, 316.0, 468.0, 948.0, 1440.0, 972.0, 492.0, 324.0, 656.0, 996.0, 672.0, 340.0, 168.0, 340.0, 516.0, 348.0, 176.0, 108.0, 220.0, 336.0, 228.0, 116.0, 228.0, 464.0, 708.0, 480.0, 244.0, 360.0, 732.0, 1116.0, 756.0, 384.0, 252.0, 512.0, 780.0, 528.0, 268.0, 132.0, 268.0, 408.0, 276.0, 140.0, 144.0, 292.0, 444.0, 300.0, 152.0, 300.0, 608.0, 924.0, 624.0, 316.0, 468.0, 948.0, 1440.0, 972.0, 492.0, 324.0, 656.0, 996.0, 672.0, 340.0, 168.0, 340.0, 516.0, 348.0, 176.0
    };

    for (int i = 0; i < 100; ++i) {
        assert(std::fabs(h_dx[i] - expected_dx[i]) < 1e-4f && "dx mismatch");
    }
   

    std::cout << "ConvLayer forward/backward tests passed!" << std::endl;
    
    cudnnDestroy(cudnn);
    return 0;
}