#include <math_constants.h>
#include "tensor.h"
#include "config.h"


class BetaDist {
public:
    BetaDist();
    ~BetaDist();

    void forward (Tensor* alpha,  Tensor* beta, cudaStream_t stream);
    void backward(Tensor* dlogp,  Tensor* dh, cudaStream_t stream);

    Tensor* action() { return action_; }
    Tensor* logp() { return logp_; }
    Tensor* entropy() { return entropy_; }
    // Tensor* da_logp() { return da_logp_; }
    // Tensor* db_logp() { return db_logp_; }
    // Tensor* da_h() { return da_h_; }
    // Tensor* db_h() { return db_h_; }
    Tensor* da() { return da_; }
    Tensor* db() { return db_; }


    

private:
    Tensor *action_  = nullptr;
    Tensor *logp_ = nullptr;
    Tensor *entropy_ = nullptr;
    // Tensor *da_logp_ = nullptr;
    // Tensor *db_logp_ = nullptr;
    // Tensor *da_h_ = nullptr;
    // Tensor *db_h_ = nullptr;
    Tensor *da_ = nullptr;
    Tensor *db_ = nullptr;

    Tensor *alpha_cache = nullptr;
    Tensor *beta_cache = nullptr;
};
