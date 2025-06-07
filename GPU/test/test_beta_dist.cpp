 #include "beta_dist.h"
 #include <cassert>
 #include <cmath>
 #include <iostream>
 #include <fstream>
 #include <vector>
 #include <numeric>

 int main() {
    std::ifstream expected_action_file("beta_dist_action.txt");
    std::ifstream expected_logp_file("beta_dist_logp.txt");
    std::ifstream expected_entropy_file("beta_dist_entropy.txt");
    std::ifstream expected_da_logp_file("beta_dist_da_logp.txt");
    std::ifstream expected_db_logp_file("beta_dist_db_logp.txt");
    std::ifstream expected_da_h_file("beta_dist_da_h.txt");
    std::ifstream expected_db_h_file("beta_dist_db_h.txt");

    std::vector<float> expected_action;
    std::vector<float> expected_logp;
    std::vector<float> expected_entropy;
    std::vector<float> expected_da_logp;
    std::vector<float> expected_db_logp;
    std::vector<float> expected_da_h;
    std::vector<float> expected_db_h;

    float value = 0;

    while (expected_action_file >> value) expected_action.push_back(value);
    while (expected_logp_file >> value) expected_logp.push_back(value);
    while (expected_entropy_file >> value) expected_entropy.push_back(value);
    while (expected_da_logp_file >> value) expected_da_logp.push_back(value);
    while (expected_db_logp_file >> value) expected_db_logp.push_back(value);
    while (expected_da_h_file >> value) expected_da_h.push_back(value);
    while (expected_db_h_file >> value) expected_db_h.push_back(value);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int B = 1024, dim = 4;
    const int total = B*dim;
    Tensor alpha(B, dim, 1, 1);
    Tensor beta(B, dim, 1, 1);

    std::vector<float> host_alpha(total);
    std::vector<float> host_beta(total);
    for (int i = 0; i < total; ++i) host_alpha[i] = static_cast<float>(i+1);
    for (int i = 0; i < total; ++i) host_beta[i] = static_cast<float>(i+2);
    CUDA_CALL(cudaMemcpy(alpha.data, host_alpha.data(), host_alpha.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(beta.data, host_beta.data(), host_beta.size()*sizeof(float), cudaMemcpyHostToDevice));

    BetaDist bd = BetaDist();

    // Forward
    bd.forward(&alpha, &beta, stream);
    Tensor* a = bd.action();
    Tensor* l = bd.logp();
    Tensor* e = bd.entropy();

    std::vector<float> host_a(total);
    std::vector<float> host_l(B);
    std::vector<float> host_e(B);
    CUDA_CALL(cudaMemcpy(host_a.data(), a->data, host_a.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_l.data(), l->data, host_l.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_e.data(), e->data, host_e.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float mean_a = std::accumulate(host_a.begin(), host_a.end(), 0.0) / host_a.size();
    float mean_l = std::accumulate(host_l.begin(), host_l.end(), 0.0) / host_l.size();
    float mean_e = std::accumulate(host_e.begin(), host_e.end(), 0.0) / host_e.size();

    float true_mean_a = std::accumulate(expected_action.begin(), expected_action.end(), 0.0) / host_a.size();
    float true_mean_l = std::accumulate(expected_logp.begin(), expected_logp.end(), 0.0) / host_l.size();
    float true_mean_e = std::accumulate(expected_entropy.begin(), expected_entropy.end(), 0.0) / host_e.size();

    assert(std::fabs(mean_a - true_mean_a) < 1e-2f);
    assert(std::fabs(mean_l - true_mean_l) < 5e-1f);
    assert(std::fabs(mean_e - true_mean_e) < 5e-1f);


    // Backward
    Tensor dlogp(B, 1, 1, 1);
    Tensor dh(B, 1, 1, 1);
    std::vector<float> host_dlogp(B);
    std::vector<float> host_dh(B);
    for (int i = 0; i < B; ++i) host_dlogp[i] = static_cast<float>(i+1);
    for (int i = 0; i < B; ++i) host_dh[i] = static_cast<float>(i+1);
    CUDA_CALL(cudaMemcpy(dlogp.data, host_dlogp.data(), host_dlogp.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dh.data, host_dh.data(), host_dh.size()*sizeof(float), cudaMemcpyHostToDevice));
     
    for (int i = 0; i < total; ++i) host_a[i] = static_cast<float>(i+1) / static_cast<float>(total+2);
    CUDA_CALL(cudaMemcpy(a->data, host_a.data(), host_a.size()*sizeof(float), cudaMemcpyHostToDevice));
    bd.backward(&dlogp, &dh, stream);

    Tensor* da_logp = bd.da_logp();
    Tensor* db_logp = bd.db_logp();
    Tensor* da_h = bd.da_h();
    Tensor* db_h = bd.db_h();
    std::vector<float> host_da_logp(total);
    std::vector<float> host_db_logp(total);
    std::vector<float> host_da_h(total);
    std::vector<float> host_db_h(total);

    CUDA_CALL(cudaMemcpy(host_da_logp.data(), da_logp->data, host_da_logp.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_db_logp.data(), db_logp->data, host_db_logp.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_da_h.data(), da_h->data, host_da_h.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_db_h.data(), db_h->data, host_db_h.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float mean_da_logp = std::accumulate(host_da_logp.begin(), host_da_logp.end(), 0.0) / host_da_logp.size();
    float mean_db_logp = std::accumulate(host_db_logp.begin(), host_db_logp.end(), 0.0) / host_db_logp.size();
    float mean_da_h = std::accumulate(host_da_h.begin(), host_da_h.end(), 0.0) / host_da_h.size();
    float mean_db_h = std::accumulate(host_db_h.begin(), host_db_h.end(), 0.0) / host_db_h.size();

    float true_mean_da_logp = std::accumulate(expected_da_logp.begin(), expected_da_logp.end(), 0.0) / expected_da_logp.size();
    float true_mean_db_logp = std::accumulate(expected_db_logp.begin(), expected_db_logp.end(), 0.0) / expected_db_logp.size();
    float true_mean_da_h = std::accumulate(expected_da_h.begin(), expected_da_h.end(), 0.0) / expected_da_h.size();
    float true_mean_db_h = std::accumulate(expected_db_h.begin(), expected_db_h.end(), 0.0) / expected_db_h.size();

    assert(std::fabs(mean_da_logp - true_mean_da_logp) < 5e-1f);
    assert(std::fabs(mean_db_logp - true_mean_db_logp) < 5e-1f);
    assert(std::fabs(mean_da_h - true_mean_da_h) < 5e-1f);
    assert(std::fabs(mean_db_h - true_mean_db_h) < 5e-1f);

    std::cout << "Beta Dist forward/backward tests passed!" << std::endl;

    cudaStreamDestroy(stream);

    return 0;
 }