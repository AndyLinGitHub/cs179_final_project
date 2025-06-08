#include "beta_dist.h"
#include <cassert>

__device__ inline float digammaf(float x) {
    float r = 0.f;
    while (x < 5.f) { r -= 1.f / x; x += 1.f; }

    float f = 1.f / (x * x);
    r += logf(x) - .5f/x - f*(1.f/12.f - f*(1.f/120.f - f/252.f));
    
    return r;
}

__device__ inline float trigammaf(float x) {
    float acc = 0.0;

    while (x < 5.0) {
        acc += 1.0 / (x * x);
        x   += 1.0;
    }

    float invx = 1.0 / x;
    float invx2 = invx * invx;
    float series = invx + invx2 * (0.5 + invx  * (1.0/6.0 - invx2 * (1.0/30.0)));

    return acc + series;
}

__device__ float gamma_rsample(curandStatePhilox4_32_10_t& state, float k)
{
    const float d = k - 1.f/3.f;
    const float c = 1.f / sqrtf(9.f*d);

    while (true) {
        float x = curand_normal(&state);
        float v = 1.f + c*x;
        if (v <= 0.f) continue;
        
        v = v*v*v;
        float u = curand_uniform(&state);
        if (u < 1.f - .0331f * x * x * x * x) return d * v;
        if (logf(u) < .5f * x * x + d * (1.f - v + logf(v))) return d * v;
    }
}

__global__ void beta_dist_forward_kernel(const float *alpha, const float *beta, float *action, 
                                         float *logp_sum, float *h_sum, int B, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    curandStatePhilox4_32_10_t rng;
    curand_init(clock64(), idx, 0, &rng);

    float logp = 0.f, h = 0.f;

    #pragma unroll
    for (int i = 0; i < dim; ++i) {
        float a = alpha[dim*idx + i];
        float b = beta[dim*idx + i];

        float g1 = gamma_rsample(rng, a);
        float g2 = gamma_rsample(rng, b);
        float x  = g1 / (g1 + g2);
        action[dim*idx + i] = x;

        float lnB = lgammaf(a) + lgammaf(b) - lgammaf(a + b);
        logp += (a - 1.f) * logf(x) + (b - 1.f) * logf(1.f - x) - lnB;
        h += lnB - (a - 1.f) * digammaf(a) - (b - 1.f) * digammaf(b) + (a + b - 2.f) * digammaf(a + b);
    }

    logp_sum[idx] = logp;
    h_sum[idx] = h;
}

__global__ void beta_dist_backward_kernel(const float *alpha, const float *beta, const float *action, 
                                          const float *e, float *da, float *db, int B, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    float e_ = e[idx];

    #pragma unroll
    for (int i = 0; i < dim; ++i) {
        float a = alpha[dim*idx + i];
        float b = beta [dim*idx + i];
        float x = action[dim*idx + i];

        if (x <= 0.000001f) x = 0.000001f;
        if (x >= 0.999999f) x = 0.999999f;

        float psi_ab = digammaf(a + b);
        float dlogp_da = logf(x) - digammaf(a) + psi_ab;
        float dlogp_db = logf(1.f - x) - digammaf(b) + psi_ab;

        float trig_ab = trigammaf(a + b);
        float dh_da  = -(a - 1.f) * trigammaf(a) + (a + b - 2.f) * trig_ab;
        float dh_db  = -(b - 1.f) * trigammaf(b) + (a + b - 2.f) * trig_ab;

        da[dim*idx + i] = e_*e_*dlogp_da;
        db[dim*idx + i] = e_*e_*dlogp_db;

        assert(!isnan(dlogp_db));
    }
}

BetaDist:: BetaDist() {
}

BetaDist:: ~BetaDist() {
    if (action_) delete action_;
    if (logp_) delete logp_;
    if (entropy_) delete entropy_;
    if (da_) delete da_;
    if (db_) delete db_;
}

void BetaDist:: forward(Tensor* alpha,  Tensor* beta, cudaStream_t stream) {
    alpha_cache = alpha;
    beta_cache = beta;

    const int B = alpha->n();
    const int dim = alpha->c();

    if (!action_) action_ = new Tensor(B, dim, 1, 1);
    if (!logp_) logp_ = new Tensor(B, 1, 1, 1); // Sum of last dimension
    if (!entropy_) entropy_ = new Tensor(B, 1, 1, 1); // Sum of last dimension

    const int blocks  = (B + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;    
    beta_dist_forward_kernel<<<blocks, THREAD_PER_BLOCK, 0, stream>>>(alpha->data, beta->data, action_->data, 
                                                                logp_->data, entropy_->data, B, dim);
}

void BetaDist:: backward(Tensor* e, cudaStream_t stream) {
    const int B = alpha_cache->n();
    const int dim = alpha_cache->c();

    if (!da_) da_ = new Tensor(B, dim, 1, 1);
    if (!db_) db_ = new Tensor(B, dim, 1, 1);

    const int blocks  = (B + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    beta_dist_backward_kernel<<<blocks, THREAD_PER_BLOCK, 0, stream>>>(alpha_cache->data, beta_cache->data, action_->data, 
                                                                       e->data, da_->data, db_->data, B, dim);
}


