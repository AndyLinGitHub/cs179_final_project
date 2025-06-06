import torch
import torch.nn.functional as F

N = 1024
C = 4
H = 1
W = 1
total = N*C*H*W

x = torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(N, C, H, W)
y = F.softplus(x) + 1

dy = torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(N, C, H, W)
dx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=dy)

y = list(y.detach().numpy().flatten())
dx = list(dx[0].detach().numpy().flatten())

with open("softplus_add1_y.txt", "w") as f:
    f.write(" ".join(map(str, y)))

with open("softplus_add1_dx.txt", "w") as f:
    f.write(" ".join(map(str, dx)))