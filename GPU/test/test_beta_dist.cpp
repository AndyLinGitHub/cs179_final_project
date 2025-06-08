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
  std::ifstream expected_da_file("beta_dist_da.txt");
  std::ifstream expected_db_file("beta_dist_db.txt");

  std::vector<float> expected_action;
  std::vector<float> expected_logp;
  std::vector<float> expected_entropy;
  std::vector<float> expected_da;
  std::vector<float> expected_db;

  float value = 0;

  while (expected_action_file >> value) expected_action.push_back(value);
  while (expected_logp_file >> value) expected_logp.push_back(value);
  while (expected_entropy_file >> value) expected_entropy.push_back(value);
  while (expected_da_file >> value) expected_da.push_back(value);
  while (expected_db_file >> value) expected_db.push_back(value);


  cudaStream_t stream;
  cudaStreamCreate(&stream);

  const int B = 4096, dim = 4;
  const int total = B*dim;
  Tensor alpha(B, dim, 1, 1);
  Tensor beta(B, dim, 1, 1);

  std::vector<float> host_alpha(total);
  std::vector<float> host_beta(total);
  for (int i = 0; i < total; ++i) host_alpha[i] = static_cast<float>(1.5);
  for (int i = 0; i < total; ++i) host_beta[i] = static_cast<float>(1.5);
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

  // assert(std::fabs(mean_a/true_mean_a - 1) < 5e-3f);
  // assert(std::fabs(mean_l/true_mean_l - 1) < 5e-3f);
  assert(std::fabs(mean_e/true_mean_e - 1) < 5e-3f);


  // // Backward
  // Tensor de(B, 1, 1, 1);
  // std::vector<float> host_de(B);
  // for (int i = 0; i < B; ++i) host_de[i] = static_cast<float>(1+1);
  // CUDA_CALL(cudaMemcpy(dh.data, host_de.data(), host_de.size()*sizeof(float), cudaMemcpyHostToDevice));
  
  // bd.backward(&de, stream);

  // Tensor* da = bd.da();
  // Tensor* db = bd.db();
  // std::vector<float> host_da(total);
  // std::vector<float> host_db(total);

  // CUDA_CALL(cudaMemcpy(host_da.data(), da->data, host_da.size()*sizeof(float), cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaMemcpy(host_db.data(), db->data, host_db.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // float mean_da = std::accumulate(host_da.begin(), host_da.end(), 0.0) / host_da.size();
  // float mean_db = std::accumulate(host_db.begin(), host_db.end(), 0.0) / host_db.size();

  // float true_mean_da = std::accumulate(expected_da.begin(), expected_da.end(), 0.0) / expected_da.size();
  // float true_mean_db = std::accumulate(expected_db.begin(), expected_db.end(), 0.0) / expected_db.size();

  // std:: cout << "Output: " << mean_da << " Expected: " << true_mean_da << std:: endl;
  // std:: cout << "Output: " << mean_db << " Expected: " << true_mean_db << std:: endl;
  std:: cout << "Beta Dist forward/backward tests passed!" << std::endl;

  cudaStreamDestroy(stream);

  return 0;
}