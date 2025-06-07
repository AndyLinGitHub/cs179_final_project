import torch
import torch.nn.functional as F

N = 8
C = 2
H = 5
W = 5
total = N*C*H*W

in_channel = 2
out_channel = 4
k = 3
stride = 1
pad = 0
total_weights = in_channel * out_channel * k * k

x = torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(N, C, H, W)
weights = torch.arange(total_weights, dtype=torch.float32, requires_grad=True).reshape(out_channel, in_channel, k, k)
bias = torch.arange(out_channel, dtype=torch.float32, requires_grad=True)
y = F.conv2d(x, weights, bias, stride, pad)
out_N, out_C, out_H, out_W = y.shape
out_total = out_N*out_C*out_H*out_W

dy = torch.arange(out_total, dtype=torch.float32, requires_grad=True).reshape(out_N, out_C, out_H, out_W)
dx, dW, db = torch.autograd.grad(outputs=y, inputs=(x, weights, bias), grad_outputs=dy)

y = list(y.detach().numpy().flatten())
dx = list(dx.detach().numpy().flatten())
dW = list(dW.detach().numpy().flatten())
db = list(db.detach().numpy().flatten())

with open("conv_y.txt", "w") as f:
    f.write(" ".join(map(str, y)))

with open("conv_dx.txt", "w") as f:
    f.write(" ".join(map(str, dx)))

with open("conv_dW.txt", "w") as f:
    f.write(" ".join(map(str, dW)))

with open("conv_db.txt", "w") as f:
    f.write(" ".join(map(str, db)))