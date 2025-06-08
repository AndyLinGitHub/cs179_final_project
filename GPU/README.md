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
```

## Performance Analysis
## Potential Improvements
