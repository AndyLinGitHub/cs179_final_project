cmake_minimum_required(VERSION 3.24)
project(final_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# ── CUDA & cuDNN/cuBLAS ───────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          /usr/lib
          /usr/local/cuda/lib64
          /usr/lib/x86_64-linux-gnu
          "$ENV{CUDA_PATH}/lib/x64")


# Tensor Test
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

# Parameter Test
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

# Relu Test
add_executable(test_relu test/test_relu.cpp src/relu.cpp src/tensor.cpp src/parameter.cpp)
    
target_include_directories(test_relu
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_relu
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# SoftPlus Add 1 Test
add_executable(test_softplus_add1 test/test_softplus_add1.cpp src/softplus_add1.cu src/tensor.cpp src/parameter.cpp)

target_include_directories(test_softplus_add1
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_softplus_add1
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# Conv Test
add_executable(test_conv test/test_conv.cpp src/conv.cpp src/tensor.cpp src/parameter.cpp)

target_include_directories(test_conv
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_conv
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# FC Test
add_executable(test_fc test/test_fc.cpp src/fc.cu src/tensor.cpp src/parameter.cpp)

target_include_directories(test_fc
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_fc
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# Beta Dist Test
add_executable(test_beta_dist test/test_beta_dist.cpp src/beta_dist.cu src/tensor.cpp)

target_include_directories(test_beta_dist
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_beta_dist
    PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${CUDNN_LIBRARY}
)

# Policy Test
add_executable(test_policy test/test_policy.cpp src/policy.cu src/beta_dist.cu src/fc.cu src/conv.cpp src/softplus_add1.cu src/relu.cpp src/tensor.cpp src/parameter.cpp)
target_compile_definitions(test_policy PRIVATE POLICY_DEBUG=1)

target_include_directories(test_policy
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_policy
    PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${CUDNN_LIBRARY}
)
