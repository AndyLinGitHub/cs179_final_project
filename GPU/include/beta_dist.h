#include <math_constants.h>
#include "tensor.h"
#include "config.h"


class BetaDist {
public:
    BetaDist();
    ~BetaDist();

    void forward (Tensor* alpha,  Tensor* beta, cudaStream_t stream);
    void backward(Tensor* e, cudaStream_t stream);

    Tensor* action() { return action_; }
    Tensor* logp() { return logp_; }
    Tensor* entropy() { return entropy_; }
    Tensor* da() { return da_; }
    Tensor* db() { return db_; }


    

private:
    Tensor *action_  = nullptr;
    Tensor *logp_ = nullptr;
    Tensor *entropy_ = nullptr;
    Tensor *da_ = nullptr;
    Tensor *db_ = nullptr;

    Tensor *alpha_cache = nullptr;
    Tensor *beta_cache = nullptr;
};
