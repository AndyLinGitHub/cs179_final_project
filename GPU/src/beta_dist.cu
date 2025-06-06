#include "beta_dist.h"
#include "beta_dist_kernel.cuh"

BetaPolicyOp::BetaPolicyOp(int batch, int dim)
    : B_(batch), D_(dim)
{
}

void BetaPolicyOp::forward( Tensor* d_alpha,  Tensor* d_beta)
{
    alpha_cache = d_alpha;
    beta_cache = d_beta;

    if (!d_action_) d_action_ = new Tensor(B_, D_, 1, 1);
    if (!d_logp_) d_logp_ = new Tensor(B_, 1, 1, 1);
    if (!d_entropy_) d_entropy_ = new Tensor(B_, 1, 1, 1);

    int threads = 256, blocks = (B_ + threads - 1) / threads;
    beta4_forward_kernel<<<blocks, threads>>>(
        d_alpha->data, d_beta->data,
        d_action_->data, d_logp_->data, d_entropy_->data,
        42, B_);
}

void BetaPolicyOp::backward( Tensor* d_dLogp,  Tensor* d_dEnt)
{
    if (!d_dAlpha_logp) d_dAlpha_logp = new Tensor(B_, D_, 1, 1);
    if (!d_dBeta_logp) d_dBeta_logp = new Tensor(B_, D_, 1, 1);
    if (!d_dAlpha_h) d_dAlpha_h = new Tensor(B_, D_, 1, 1);
    if (!d_dBeta_h) d_dBeta_h = new Tensor(B_, D_, 1, 1);

    int threads = 256, blocks = (B_ + threads - 1) / threads;
    beta4_backward_kernel<<<blocks, threads>>>(
        alpha_cache->data, beta_cache->data,
        d_action_->data, d_dLogp->data, d_dEnt->data,
        d_dAlpha_logp->data, d_dBeta_logp->data,
        d_dAlpha_h->data, d_dBeta_h->data,
        B_);
}

BetaPolicyOp::~BetaPolicyOp()
{
    if (d_action_) delete d_action_;
    if (d_logp_) delete d_logp_;
    if (d_entropy_) delete d_entropy_;
    if (d_dAlpha_logp) delete d_dAlpha_logp;
    if (d_dBeta_logp) delete d_dBeta_logp;
    if (d_dAlpha_h) delete d_dAlpha_h;
    if (d_dBeta_h) delete d_dBeta_h;
}
