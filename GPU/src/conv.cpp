#include "conv.h"

Conv:: Conv(int in_channel, int out_channel, int k, int stride, int pad) : 
    in_channel_(in_channel), out_channel_(out_channel), k_(k), stride_(stride), pad_(pad),
    W(static_cast<size_t>(in_channel) * out_channel * k * k), b(static_cast<size_t>(out_channel)){
    
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channel, in_channel, k, k));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&biasDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channel, 1, 1));
}

Conv:: ~Conv() {
    if (dx) delete dx;
    if (y) delete y;

    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyTensorDescriptor(biasDesc);
}

Tensor* Conv:: forward(Tensor* x, cudnnHandle_t handle, cudaStream_t stream) {
    CUDNN_CALL(cudnnSetStream(handle, stream));

    x_cache = x;

    int out_n, out_c, out_h, out_w; // For getting the size of output
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convDesc, x->desc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    if (!y) y = new Tensor(out_n, out_c, out_h, out_w);

    // Select an algorithm
    int algoReturned = 0;
    size_t workspace_bytes = 0;
    const int algoRequest = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionFwdAlgoPerf_t perf[ CUDNN_CONVOLUTION_FWD_ALGO_COUNT ];

    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(handle, x->desc, filterDesc, convDesc, y->desc, algoRequest, &algoReturned, perf));
    algo = perf[0].algo; // fastest is first
    workspace_bytes = perf[0].memory; // workspace size for that algo
    
    // Set the workspace
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x->desc, filterDesc, convDesc, y->desc, algo, &workspace_bytes));
    void* workspace = nullptr;
    if (workspace_bytes) CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle, &alpha, x->desc, x->data, filterDesc, W.value, convDesc, algo, workspace, workspace_bytes, &beta, y->desc, y->data));

    // Add bias
    CUDNN_CALL(cudnnAddTensor(handle, &alpha, biasDesc, b.value, &alpha, y->desc, y->data));

    if (workspace) cudaFree(workspace);

    return y;
}

Tensor* Conv:: backward(Tensor* dy, cudnnHandle_t handle, cudaStream_t stream) {
    CUDNN_CALL(cudnnSetStream(handle, stream));
    
    const float alpha = 1.0f, beta = 0.0f;

    // Bias backward 
    CUDNN_CALL(cudnnConvolutionBackwardBias(handle, &alpha, dy->desc, dy->data, &beta, biasDesc, b.grad));

    // Select an algorithm
    size_t wspace_filter_bytes = 0;
    cudnnConvolutionBwdFilterAlgo_t algo_filter;
    const int algoRequestBF = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    int algoReturnedBF = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perfBF[ CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT ];

    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x_cache->desc, dy->desc, convDesc, filterDesc, algoRequestBF, &algoReturnedBF, perfBF));
    algo_filter = perfBF[0].algo;
    wspace_filter_bytes = perfBF[0].memory;
    
    // Set the workspace
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_cache->desc, dy->desc, convDesc, filterDesc, algo_filter, &wspace_filter_bytes));
    void* wspace_filter = nullptr;
    if (wspace_filter_bytes) CUDA_CALL(cudaMalloc(&wspace_filter, wspace_filter_bytes));

    // Weight backward
    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle, &alpha, x_cache->desc, x_cache->data, dy->desc, dy->data, convDesc, algo_filter, wspace_filter, wspace_filter_bytes, &beta, filterDesc, W.grad));
    if (wspace_filter) cudaFree(wspace_filter);

    // Select an algorithm
    size_t wspace_data_bytes = 0;
    cudnnConvolutionBwdDataAlgo_t algo_data;
    const int algoRequestBD = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    int algoReturnedBD      = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perfBD[ CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT ];
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, dy->desc, convDesc, x_cache->desc, algoRequestBD, &algoReturnedBD, perfBD));
    algo_data = perfBD[0].algo;
    wspace_data_bytes = perfBD[0].memory;

    // Set the workspace
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filterDesc, dy->desc, convDesc, x_cache->desc, algo_data, &wspace_data_bytes));
    void* wspace_data = nullptr;
    if (wspace_data_bytes) CUDA_CALL(cudaMalloc(&wspace_data, wspace_data_bytes));

    // Data backward
    if (!dx) dx =  new Tensor(x_cache->n(), x_cache->c(), x_cache->h(), x_cache->w());
    CUDNN_CALL(cudnnConvolutionBackwardData(handle, &alpha, filterDesc, W.value, dy->desc, dy->data, convDesc, algo_data, wspace_data, wspace_data_bytes, &beta, dx->desc, dx->data));
    if (wspace_data) cudaFree(wspace_data);

    return dx;
}