# cs179_final_project

## Installation and Usage Instructions
```
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Project Description and Features
- This project implements a convolutional neural-network (CNN) with a β-policy in **C++/CUDA**. All layers are built using **cuDNN**, **cuBLAS**, **cuRAND**, and custom CUDA kernels. Training is carried out on the GPU with an implementation of the **Adam optimizer** and the **REINFORCE** algorithm.

- A companion demo illustrates that both the forward- and backward-passes work as intended. Inspired by classic Neural Cellular Automata, it treats every pixel as an agent that reads its neighbors and predicts its own update. To keep the learning task short, the demo trains a policy network that gradually drives any input image toward solid white. At each frame the network outputs the **α** and **β** parameters of a Beta distribution, samples an action *i*, and updates each pixel with `pixel_i ← pixel_i + (2 · action_i − 1)`. This setup neatly exposes the network layers, gradient flow, and custom Adam-REINFORCE optimizer in action.



## Expected Results
```
# Expect to see the loss decreasing gradually during training. 
./main

# The forward and backward results are expected to be the same as those output from Pytorch. 
python3 ../test_gen/conv_test_gen.py
./test_conv

python3 ../test_gen/fc_test_gen.py
./test_fc

python3 ../test_gen/softplus_test_gen.py 
 ./test_softplus_add1 

# Only test forward
python3 ../test_gen/beta_dist_test_gen.py
./test_beta_dist 
```

## Performance Analysis
- Latency comapres to Pytorch (CPU)
    - FC Forward 100x
    - FC Backward 0.02x
    - Conv Forward 70x
    - Conv Backward	0.05x
    - Softplus Forward 1x
    - Softplus Backward 0.0001x

The CUDA kernels speed up the backward pass, yet the forward pass remains sluggish—most likely because tensors are laid out in row-major rather than column-major order. This choice mis-aligns memory accesses for forward kernels, causing non-coalesced global reads/writes and extra register spilling that throttle performance. Conversely, backward propagation often involves matrix transposes, so a row-major layout can actually align those memory accesses and partially explain the backward-pass gains.

## Potential Improvements
- **Modular, Config-Driven Architecture with Modern Algorithms**
   Replace the hard-coded layer sequence in the forward and backward passes with a fully modular design driven by a concise configuration file (e.g., YAML or JSON). This declarative approach simplifies experimentation, enables automated hyper-parameter searches, and future-proofs the codebase. At the same time, broaden the framework’s reach by adding two industry-standard policy-gradient methods—Proximal Policy Optimization (PPO) and Soft Actor–Critic (SAC).

- **Automated Bottleneck Analysis for Memory Layout**
   Embed a lightweight profiling warm-up that inspects memory-access patterns across the computation graph, pinpoints hotspots, and automatically decides whether each intermediate tensor should be stored in row-major or column-major order. Aligning tensor layouts with cache behavior can accelerate both forward and backward propagation beyond vanilla PyTorch, all without manual tuning.

- **Real-Time Training Dashboard**
   Integrate a live GUI that streams key metrics—loss curves, reward trajectories, policy distributions, and convergence diagnostics—so users can immediately see how target settings and hyper-parameters influence learning dynamics during training.



