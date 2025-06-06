#include "tensor.h"

class BetaPolicyOp {
public:
    BetaPolicyOp(int batch, int dim = 4);

    void forward ( Tensor* d_alpha,  Tensor* d_beta);
    void backward( Tensor* d_dLogp,  Tensor* d_dEnt);

    Tensor* action()  { return d_action_; }
    Tensor* logp()  { return d_logp_; }
    Tensor* entropy()  { return d_entropy_; }
    Tensor* gradAlpha_logp()  { return d_dAlpha_logp; }
    Tensor* gradBeta_logp()  { return d_dBeta_logp; }
    Tensor* gradAlpha_h()  { return d_dAlpha_h; }
    Tensor* gradBeta_h()  { return d_dBeta_h; }

    ~BetaPolicyOp();

private:
    int B_, D_;

    Tensor *d_action_  = nullptr;
    Tensor *d_logp_ = nullptr;
    Tensor *d_entropy_ = nullptr;
    Tensor *d_dAlpha_logp = nullptr;
    Tensor *d_dBeta_logp = nullptr;
    Tensor *d_dAlpha_h = nullptr;
    Tensor *d_dBeta_h = nullptr;

    Tensor *alpha_cache = nullptr;
    Tensor *beta_cache = nullptr;
};
