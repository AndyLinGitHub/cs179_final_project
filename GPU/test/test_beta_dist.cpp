 #include "beta_dist.h"
 #include <cassert>
 #include <cmath>
 #include <iostream>
 #include <vector>
 #include <numeric>

 int main() {
     // Build input tensor ------------------------------------------------------
     const int N = 256, C = 4, H = 1, W = 1;
     Tensor alpha(N, C, H, W);
     Tensor beta(N, C, H, W);

     std::vector<float> h_alpha(1024);
     std::vector<float> h_beta(1024);
     for (int i = 0; i < 1024; ++i) h_alpha[i] = static_cast<float>(i+1);
     for (int i = 0; i < 1024; ++i) h_beta[i] = static_cast<float>(2*i+1);
     CUDA_CALL(cudaMemcpy(alpha.data, h_alpha.data(), h_alpha.size()*sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMemcpy(beta.data, h_beta.data(), h_beta.size()*sizeof(float), cudaMemcpyHostToDevice));

     BetaPolicyOp bpo(256, 4); 
     bpo.forward(&alpha, &beta);
     Tensor* a = bpo.action();
     Tensor* l = bpo.logp();
     Tensor* e = bpo.entropy();

     std::vector<float> h_a(h_alpha.size());
     std::vector<float> h_l(256);
     std::vector<float> h_e(256);

     CUDA_CALL(cudaMemcpy(h_a.data(), a->data, h_a.size()*sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CALL(cudaMemcpy(h_l.data(), l->data, h_l.size()*sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CALL(cudaMemcpy(h_e.data(), e->data, h_e.size()*sizeof(float), cudaMemcpyDeviceToHost));

     float mean = std::accumulate(h_a.begin(), h_a.end(), 0.0) / h_a.size();
     float mean_l = std::accumulate(h_l.begin(), h_l.end(), 0.0) / h_l.size();
     float mean_e = std::accumulate(h_e.begin(), h_e.end(), 0.0) / h_e.size();
     assert(std::fabs(mean - 0.333) < 1e-2f && "forward output mismatch");
     assert(std::fabs(mean_l - 11.5) < 5e-1f && "forward output mismatch");
     assert(std::fabs(mean_e + 11.5) < 5e-1f && "forward output mismatch");

     Tensor dl(N, 1, H, W);
     Tensor de(N, 1, H, W);
     std::vector<float> h_dl(256);
     std::vector<float> h_de(256);
     for (int i = 0; i < 256; ++i) h_dl[i] = static_cast<float>(1);
     for (int i = 0; i < 256; ++i) h_de[i] = static_cast<float>(1);
     CUDA_CALL(cudaMemcpy(dl.data, h_dl.data(), h_dl.size()*sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMemcpy(de.data, h_de.data(), h_de.size()*sizeof(float), cudaMemcpyHostToDevice));
     
     for (int i = 0; i < 1024; ++i) h_a[i] = static_cast<float>(0.5);
     CUDA_CALL(cudaMemcpy(a->data, h_a.data(), h_a.size()*sizeof(float), cudaMemcpyHostToDevice));
     bpo.backward(&dl, &de);

     Tensor* da = bpo.gradAlpha_logp();
     Tensor* db = bpo.gradBeta_logp();
     Tensor* da2 = bpo.gradAlpha_h();
     Tensor* db2 = bpo.gradBeta_h();
     std::vector<float> h_da(1024);
     std::vector<float> h_db(1024);
     std::vector<float> h_da2(1024);
     std::vector<float> h_db2(1024);

     //CUDA_CALL(cudaMemcpy(h_da.data(), da->data, h_da.size()*sizeof(float), cudaMemcpyDeviceToHost));
     //float sum = std::accumulate(h_da.begin(), h_da.end(), 0.0);
     //for (int i = 0; i < 32; ++i) {
      //std::cout << sum << std::endl;
    //}

     //CUDA_CALL(cudaMemcpy(h_db.data(), db->data, h_db.size()*sizeof(float), cudaMemcpyDeviceToHost));
     //sum = std::accumulate(h_db.begin(), h_db.end(), 0.0);
     //std::cout << sum << std::endl;

     CUDA_CALL(cudaMemcpy(h_da2.data(), da2->data, h_da2.size()*sizeof(float), cudaMemcpyDeviceToHost));
    float sum = std::accumulate(h_da2.begin(), h_da2.end(), 0.0);
    //for (int i = 0; i < 32; ++i) {
      std::cout << sum << std::endl;
    //}

     CUDA_CALL(cudaMemcpy(h_db2.data(), db2->data, h_db2.size()*sizeof(float), cudaMemcpyDeviceToHost));
     sum = std::accumulate(h_db2.begin(), h_db2.end(), 0.0);
     std::cout << sum << std::endl;


 }