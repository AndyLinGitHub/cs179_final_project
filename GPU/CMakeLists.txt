cmake_minimum_required(VERSION 3.24)
project(final_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# ── CUDA & cuDNN/cuBLAS ───────────────────────────────────
find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          /usr/lib
          /usr/local/cuda/lib64
          /usr/lib/x86_64-linux-gnu
          "$ENV{CUDA_PATH}/lib/x64")


# ── Tensor Test ───────────────────────────────────
add_executable(test_tensor test/test_tensor.cu src/tensor.cpp)

target_include_directories(test_tensor
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_tensor
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# ── Parameter Test ───────────────────────────────────
add_executable(test_parameter test/test_parameter.cpp src/parameter.cpp)

target_include_directories(test_parameter
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_parameter
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# ── Conv Layer Test ───────────────────────────────────
add_executable(test_conv_layer test/test_conv_layer.cpp src/conv_layer.cpp src/tensor.cpp src/parameter.cpp)

target_include_directories(test_conv_layer
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_conv_layer
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# ── Relu Layer Test ───────────────────────────────────
add_executable(test_relu_layer test/test_relu_layer.cpp src/relu_layer.cpp src/tensor.cpp src/parameter.cpp)

target_include_directories(test_relu_layer
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_relu_layer
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# ── FC Layer Test ───────────────────────────────────
add_executable(test_fc_layer test/test_fc_layer.cpp src/fc_layer.cu src/tensor.cpp src/parameter.cpp)

target_include_directories(test_fc_layer
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_fc_layer
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)
