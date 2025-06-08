#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include "policy.h"
#include "utils.cuh"

void save_png_from_chw(const std::vector<float>& chw, int C, int H, int W, const char* path, bool sixteenBit = false) {
    if ((C != 3 && C != 4) || chw.size() != static_cast<size_t>(C*H*W))
        throw std::invalid_argument("save_png_from_chw: bad tensor shape");

    const int bytesPerChan = sixteenBit ? 2 : 1;
    std::vector<uint8_t> hwc(W * H * C * bytesPerChan);

    if (sixteenBit) {
        const float max16 = 65535.f;
        auto* dst16 = reinterpret_cast<uint16_t*>(hwc.data());

        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < C; ++c) {
                    float v = chw[c*H*W + y*W + x];   // CHW index
                    v = std::clamp(v, 0.f, 1.f);
                    dst16[(y*W + x)*C + c] = static_cast<uint16_t>(v * max16 + 0.5f);
                }
    }
    else {
        const float max8 = 255.f;

        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < C; ++c) {
                    float v = chw[c*H*W + y*W + x];
                    v = std::clamp(v, 0.f, 1.f);
                    hwc[(y*W + x)*C + c] = static_cast<uint8_t>(v * max8 + 0.5f);
                }
        }

    const int stride = W * C * bytesPerChan;
    int ok = sixteenBit
    ? stbi_write_png(path, W, H, C,
        hwc.data(), stride)
    : stbi_write_png(path, W, H, C,
        hwc.data(), stride);

    if (!ok) throw std::runtime_error("stbi_write_png failed");
}

int main() {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    cublasHandle_t cublas;
    CUBLAS_CALL(cublasCreate(&cublas));

    const int img_size = 16;
    const int total = ACTION_DIM*img_size*img_size;
    Tensor image(1, ACTION_DIM, img_size, img_size);
    std::vector<float> host_image(total);
    for (int i = 0; i < total; ++i) host_image[i] = static_cast<float>(0);
    CUDA_CALL(cudaMemcpy(image.data, host_image.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    save_png_from_chw(host_image, 4, img_size, img_size, "input.png");

    int N = img_size*img_size;
    Tensor x(N, ACTION_DIM, RF, RF);
    Tensor ground_truth(1, ACTION_DIM, img_size, img_size);
    std::vector<float> host_ground_truth(total);
    for (int i = 0; i < total; ++i) host_ground_truth[i] = static_cast<float>(1); // White image
    CUDA_CALL(cudaMemcpy(ground_truth.data, host_ground_truth.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    Tensor error(N, ACTION_DIM, 1, 1);
    std::vector<float> host_error(N*ACTION_DIM);

    Policy policy = Policy();

    int blocks  = (img_size*img_size*ACTION_DIM + THREAD_NUM - 1) / THREAD_NUM;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 30000; i++){
         unfold_circular_kernel<<<blocks, THREAD_NUM, 0, stream>>>(image.data, x.data, img_size, img_size, ACTION_DIM, RF); // Prepare inputs
         policy.forward(&x, cudnn, cublas, stream); // Forward outputs

         // Generate the new image and calcualte loss
         new_image_kernel<<<blocks, THREAD_NUM, 0, stream>>>(policy.action()->data, image.data, img_size, img_size, ACTION_DIM);
         loss_function<<<blocks, THREAD_NUM, 0, stream>>>(image.data, ground_truth.data, error.data, ACTION_DIM, img_size, img_size);

         policy.backward(&error, cudnn, cublas, stream);
         policy.step(stream);

        if (i%100 == 0) {
            CUDA_CALL(cudaMemcpy(host_error.data(), error.data, host_error.size()*sizeof(float), cudaMemcpyDeviceToHost));
            float mean_error = std::accumulate(host_error.begin(), host_error.end(), 0.0) / host_error.size();
            std::cout << "Loss: " << mean_error << " \n";

            if (mean_error == 0.0) break;
        }
    }
        
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count() / 1000 / 1000;
    std::cout << "Elapsed: " << us << " s\n";
    CUDA_CALL(cudaMemcpy(host_image.data(), image.data, host_image.size()* sizeof(float), cudaMemcpyDeviceToHost));
    float mean_image = std::accumulate(host_image.begin(), host_image.end(), 0.0) / host_image.size();
    // std::cout << "Pixel Mean: " << mean_image << " \n";
    save_png_from_chw(host_image, 4, img_size, img_size, "output.png");

    return 0;
}