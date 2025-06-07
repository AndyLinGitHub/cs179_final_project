import torch
import torch.nn.functional as F

N = 2
in_features = 8
out_features = 16

x = torch.arange(N * in_features, dtype=torch.float32, requires_grad=True).reshape(N, in_features)
weights = torch.arange(out_features * in_features, dtype=torch.float32, requires_grad=True).reshape(out_features, in_features)
bias = torch.arange(out_features, dtype=torch.float32, requires_grad=True)
y = F.linear(x, weights, bias)

dy = torch.arange(N * out_features, dtype=torch.float32, requires_grad=True).reshape(N, out_features)
dx, dW, db = torch.autograd.grad(outputs=y, inputs=(x, weights, bias), grad_outputs=dy)

y = list(y.detach().numpy().flatten())
dx = list(dx.detach().numpy().flatten())
dW = list(dW.detach().numpy().flatten())
db = list(db.detach().numpy().flatten())

with open("fc_y.txt", "w") as f:
    f.write(" ".join(map(str, y)))

with open("fc_dx.txt", "w") as f:
    f.write(" ".join(map(str, dx)))

with open("fc_dW.txt", "w") as f:
    f.write(" ".join(map(str, dW)))

with open("fc_db.txt", "w") as f:
    f.write(" ".join(map(str, db)))
