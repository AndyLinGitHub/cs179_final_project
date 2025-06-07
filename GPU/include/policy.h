#include "tensor.h"
#include "fc.h"
#include "conv.h"
#include "relu.h"
#include "beta_dist.h"
#include "softplus_add1.h"
#include "config.h"

class Policy {
public:
    Policy();
    ~Policy();

    Tensor* forward(Tensor* x, cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle, cudaStream_t stream);
    Tensor* backward(Tensor* dlogp,  Tensor* dh, Tensor* dv, cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle, cudaStream_t stream);

    Tensor* alpha() { return alpha_; }
    Tensor* beta() { return beta_; }

private:
    Conv conv_1 = Conv(ACTION_DIM, CONV_CHANNEL, CONV_KERNEL, CONV_STRIDE, CONV_PAD);
    Conv conv_2 = Conv(CONV_CHANNEL, CONV_CHANNEL*2, CONV_KERNEL, CONV_STRIDE, CONV_PAD);
    Relu relu_1 = Relu();
    Relu relu_2 = Relu();
    Relu relu_3 = Relu();
    FC fc_1 = FC(FEATURE_DIM, HIDDEN_DIM);
    FC fc_2 = FC(HIDDEN_DIM, ACTION_DIM);
    FC fc_3 = FC(HIDDEN_DIM, ACTION_DIM);
    FC fc_4 = FC(HIDDEN_DIM, 1);
    SoftPlusAdd1 spa_1 = SoftPlusAdd1();
    SoftPlusAdd1 spa_2 = SoftPlusAdd1();
    BetaDist bd = BetaDist();

    Tensor* temp = nullptr;
    Tensor* alpha_ = nullptr;
    Tensor* beta_ = nullptr;
    Tensor* value_ = nullptr;

    Tensor* dx_1 = nullptr;
    Tensor* dx_2 = nullptr;
    Tensor* dx_3 = nullptr;
    Tensor* dx_sum = nullptr;
};
