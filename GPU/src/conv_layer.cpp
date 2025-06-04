#include "conv_layer.h"
#include <cassert>

ConvLayer::ConvLayer(int inC, int outC, int k, int stride_, int pad)
    : inChannels(inC), outChannels(outC), kernelSize(k), stride(stride_), padding(pad),
      W(static_cast<size_t>(outC) * inC * k * k),
      b(static_cast<size_t>(outC)) {
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, k, k));

    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&biasDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outC, 1, 1));
}

ConvLayer::~ConvLayer() {
    if (y) delete y;
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyTensorDescriptor(biasDesc);
}

Tensor* ConvLayer::forward(Tensor* x, cudnnHandle_t handle) {
    x_cache = x; // save for backward

    int n, c, h, w;
    n = x->n(); c = x->c(); h = x->h(); w = x->w();

    int outN, outC, outH, outW;
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convDesc, x->desc, filterDesc, &outN, &outC, &outH, &outW));

    if (!y) y = new Tensor(outN, outC, outH, outW);

    int algoReturned = 0;
    size_t workspace_bytes = 0;
    const int algoRequest = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionFwdAlgoPerf_t perf[ CUDNN_CONVOLUTION_FWD_ALGO_COUNT ];

    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(handle, x->desc, filterDesc, convDesc, y->desc, algoRequest, &algoReturned, perf));
    algo = perf[0].algo; // fastest is first
    workspace_bytes = perf[0].memory; // workspace size for that algo
    
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x->desc, filterDesc, convDesc, y->desc, algo, &workspace_bytes));
    void* workspace = nullptr;
    if (workspace_bytes) CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(handle, &alpha, x->desc, x->data, filterDesc, W.value, convDesc, algo, workspace, workspace_bytes, &beta, y->desc, y->data));

    // add bias
    CUDNN_CALL(cudnnAddTensor(handle, &alpha, biasDesc, b.value, &alpha, y->desc, y->data));

    if (workspace) cudaFree(workspace);
    return y;
}

Tensor* ConvLayer::backward(Tensor* dy, cudnnHandle_t handle) {
    // grads wrt bias
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardBias(handle, &alpha, dy->desc, dy->data, &beta, biasDesc, b.grad));

    // grads wrt filters
    size_t wspace_filter_bytes = 0;
    cudnnConvolutionBwdFilterAlgo_t algo_filter;
    const int algoRequestBF = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    int algoReturnedBF      = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perfBF[ CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT ];

    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle,
        x_cache->desc, dy->desc, convDesc, filterDesc,
        algoRequestBF, &algoReturnedBF, perfBF));

    algo_filter        = perfBF[0].algo;
    wspace_filter_bytes = perfBF[0].memory;
    //CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle, x_cache->desc, dy->desc, convDesc, filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo_filter));
    
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_cache->desc, dy->desc, convDesc, filterDesc, algo_filter, &wspace_filter_bytes));

    void* wspace_filter = nullptr;
    if (wspace_filter_bytes) CUDA_CALL(cudaMalloc(&wspace_filter, wspace_filter_bytes));

    CUDNN_CALL(cudnnConvolutionBackwardFilter(handle, &alpha, x_cache->desc, x_cache->data, dy->desc, dy->data, convDesc, algo_filter, wspace_filter, wspace_filter_bytes, &beta, filterDesc, W.grad));

    if (wspace_filter) cudaFree(wspace_filter);

    // grads wrt input (for upstream layers)
    size_t wspace_data_bytes = 0;
    cudnnConvolutionBwdDataAlgo_t algo_data;
    const int algoRequestBD = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    int algoReturnedBD      = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perfBD[ CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT ];

    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle,
        filterDesc, dy->desc, convDesc, x_cache->desc,
        algoRequestBD, &algoReturnedBD, perfBD));

    algo_data        = perfBD[0].algo;
    wspace_data_bytes = perfBD[0].memory;
    //CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle, filterDesc, dy->desc, convDesc, x_cache->desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo_data));
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filterDesc, dy->desc, convDesc, x_cache->desc, algo_data, &wspace_data_bytes));
    void* wspace_data = nullptr;
    if (wspace_data_bytes) CUDA_CALL(cudaMalloc(&wspace_data, wspace_data_bytes));

    Tensor* dx = new Tensor(x_cache->n(), x_cache->c(), x_cache->h(), x_cache->w());
    CUDNN_CALL(cudnnConvolutionBackwardData(handle, &alpha, filterDesc, W.value, dy->desc, dy->data, convDesc, algo_data, wspace_data, wspace_data_bytes, &beta, dx->desc, dx->data));

    if (wspace_data) cudaFree(wspace_data);
    return dx;
}