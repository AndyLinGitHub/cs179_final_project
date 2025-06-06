#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>   // CUDART_* constants

// ───────────────────────────────────────────── helper maths ──────

// B’(α,β) needs digamma ψ  and trigamma ψ₁.  Cheap ≈1-ulp asymptotics.
__device__ inline float digammaf(float x) {
    float r = 0.f;
    while (x < 5.f) { r -= 1.f / x; x += 1.f; }
    float f = 1.f / (x * x);
    r += logf(x) - .5f/x - f*(1.f/12.f - f*(1.f/120.f - f/252.f));
    return r;
}
/*
__device__ inline float trigammaf(float x) {
    float r = 0.f;
    while (x < 5.f) { r += 1.f / (x * x); x += 1.f; }
    float f = 1.f / (x * x);
    r += .5f/x + f*(1.f/6.f - f*(1.f/30.f + f/42.f));
    return r;
}
*/
__device__ inline float trigammaf(float x)
    {
    // Handle NaNs and negative/zero inputs explicitly if desired
    // (here we just propagate NAN or INF from divisions).
    float acc = 0.0;

    // Step 1: Recurrence to push x to moderately large value
    while (x < 5.0) {
        acc += 1.0 / (x * x);
        x   += 1.0;
    }

    // Step 2: Asymptotic expansion (k = O(1/x)).
    float invx = 1.0 / x;
    float invx2 = invx * invx;         // 1/x²
    // Horner form keeps registers low and uses FMA on sm_80+
    float series = invx
                  + invx2 * (0.5
                  + invx  * (1.0/6.0
                  - invx2 * (1.0/30.0)));

    return acc + series;
}

// Marsaglia & Tsang (2000) gamma(shape>0) generator, pathwise.
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
        if (u < 1.f - .0331f * x * x * x * x)      return d * v;
        if (logf(u) < .5f * x * x + d * (1.f - v + logf(v))) return d * v;
    }
}

// ───────────────────────────── forward kernel ────────────────────

extern "C" __global__
void beta4_forward_kernel(const float *alpha,
                          const float *beta,
                          float       *action,      // (B,4)
                          float       *logp_sum,    // (B)
                          float       *ent_sum,     // (B)
                          unsigned long long seed,
                          int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // four dims fit nicely in registers
    curandStatePhilox4_32_10_t rng;
    curand_init(clock64(), b, 0, &rng);

    float lp = 0.f, ent = 0.f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a  = alpha[4*b + i];
        float b_ = beta [4*b + i];

        // reparameterised sample:  Γ(a,1) / (Γ(a,1)+Γ(b,1))
        float g1 = gamma_rsample(rng, a);
        float g2 = gamma_rsample(rng, b_);
        float x  = g1 / (g1 + g2);          // Beta sample ∈ (0,1)
        action[4*b + i] = x;

        // log-prob component
        float lnB  = lgammaf(a) + lgammaf(b_) - lgammaf(a + b_);
        lp += (a - 1.f) * logf(x) + (b_ - 1.f) * logf(1.f - x) - lnB;

        // entropy component  (wiki formula) :contentReference[oaicite:0]{index=0}
        ent +=  lnB
              - (a - 1.f) * digammaf(a)
              - (b_ - 1.f) * digammaf(b_)
              + (a + b_ - 2.f) * digammaf(a + b_);
    }
    logp_sum[b] = lp;
    ent_sum [b] = ent;
}

// ───────────────────────────── backward kernel ───────────────────

extern "C" __global__
void beta4_backward_kernel(const float *alpha,
                           const float *beta,
                           const float *action,     // stash from fwd
                           const float *grad_lp,    // dL/dlogp_sum   (B)
                           const float *grad_ent,   // dL/dentropy_sum(B)
                           float       *grad_a_out_logp, // (B,4)
                           float       *grad_b_out_logp, // (B,4)
                           float       *grad_a_out_h, // (B,4)
                           float       *grad_b_out_h, // (B,4)
                           int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    float g_lp  = grad_lp [b];
    float g_ent = grad_ent[b];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a  = alpha[4*b + i];
        float b_ = beta [4*b + i];
        float x  = action[4*b + i];

        float psi_ab = digammaf(a + b_);
        float d_logp_da = logf(x)         - digammaf(a) + psi_ab;
        float d_logp_db = logf(1.f - x)   - digammaf(b_) + psi_ab;

        // entropy partials (ψ’ ≡ trigamma) :contentReference[oaicite:1]{index=1}
        float trig_ab = trigammaf(a + b_);
        float d_H_da  = -(a - 1.f) * trigammaf(a) + (a + b_ - 2.f) * trig_ab;
        float d_H_db  = -(b_ - 1.f) * trigammaf(b_) + (a + b_ - 2.f) * trig_ab;

        grad_a_out_logp[4*b + i] = g_lp * d_logp_da; // + g_ent * d_H_da;
        grad_b_out_logp[4*b + i] = g_lp * d_logp_db; // + g_ent * d_H_db;
        grad_a_out_h[4*b + i] = g_ent * d_H_da;
        grad_b_out_h[4*b + i] = g_ent * d_H_db;
    }
}

// ────────────────────────────── C++ wrappers ────────────────────

void beta4_forward(const float *d_alpha,
                   const float *d_beta,
                   float       *d_action,
                   float       *d_logp,
                   float       *d_entropy,
                   int B,
                   cudaStream_t stream = 0)
{
    int threads = 256;
    int blocks  = (B + threads - 1) / threads;
    unsigned long long seed = 0xdeadbeefULL;
    beta4_forward_kernel<<<blocks, threads, 0, stream>>>(
        d_alpha, d_beta, d_action, d_logp, d_entropy, seed, B);
}

void beta4_backward(const float *d_alpha,
                    const float *d_beta,
                    const float *d_action,
                    const float *d_grad_logp,
                    const float *d_grad_ent,
                    float       *d_grad_alpha_logp,
                    float       *d_grad_beta_logp,
                    float       *d_grad_alpha_h,
                    float       *d_grad_beta_h,
                    int B,
                    cudaStream_t stream = 0)
{
    int threads = 256;
    int blocks  = (B + threads - 1) / threads;
    beta4_backward_kernel<<<blocks, threads, 0, stream>>>(
        d_alpha, d_beta, d_action,
        d_grad_logp, d_grad_ent,
        d_grad_alpha_logp, d_grad_beta_logp, 
        d_grad_alpha_h, d_grad_beta_h, 
        B);
}
