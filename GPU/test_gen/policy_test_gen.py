import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

class ConvBetaActorCritic(nn.Module):
    def __init__(self, K, act_dim=4):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(act_dim, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, act_dim, K, K)
            conv_out_dim = self.cnn(dummy).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(conv_out_dim, 256), nn.ReLU(),
        )

        self.alpha_head = nn.Linear(256, act_dim)
        self.beta_head  = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs):  # obs: (B, 4, K, K)
        x = self.shared(self.cnn(obs))
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta  = F.softplus(self.beta_head(x))  + 1.0
        value = self.value_head(x).squeeze(-1)
        return alpha, beta, value

N = 1024
C = 4
K = 7
model = ConvBetaActorCritic(K, C)

# Set all weights and biases to 0.01
with torch.no_grad():
    for param in model.parameters():
        param.fill_(0.001)


x = torch.arange(N*C*K*K, dtype=torch.float32, requires_grad=False).reshape(N, C, K, K) / (N*C*K*K)
with torch.no_grad():
    after_shared = model.shared(model.cnn(x))
    alpha_pre = model.alpha_head(after_shared)
    beta_pre = model.beta_head(after_shared)
    alpha = F.softplus(alpha_pre) + 1.0
    beta  = F.softplus(beta_pre)  + 1.0
    value = model.value_head(after_shared).squeeze(-1)


value = list(value.detach().numpy().flatten())
with open("policy_value.txt", "w") as f:
    f.write(" ".join(map(str, value)))

alpha = list(alpha.detach().numpy().flatten())
with open("policy_alpha.txt", "w") as f:
    f.write(" ".join(map(str, alpha)))

beta = list(beta.detach().numpy().flatten())
with open("policy_beta.txt", "w") as f:
    f.write(" ".join(map(str, beta)))

x = torch.arange(N*C*K*K, dtype=torch.float32, requires_grad=True).reshape(N, C, K, K) / (N*C*K*K)
# alpha, beta, value = model(x)
after_shared = model.shared(model.cnn(x))
alpha_pre = model.alpha_head(after_shared)
beta_pre = model.beta_head(after_shared)
alpha = F.softplus(alpha_pre) + 1.0
beta  = F.softplus(beta_pre)  + 1.0
value = model.value_head(after_shared).squeeze(-1)

dist = Beta(alpha, beta)
with torch.no_grad():
    action = dist.rsample()
logp = dist.log_prob(action).sum(-1)
entropy = dist.entropy().sum(-1)

dlogp = torch.ones(N, dtype=torch.float32, requires_grad=True)
dh = torch.ones(N, dtype=torch.float32, requires_grad=True)
dv = torch.ones(N, dtype=torch.float32, requires_grad=True)

grad_x = torch.autograd.grad(outputs=(entropy, logp, value), inputs=x, grad_outputs=(dh, dlogp, dv))
dx = list(grad_x[0].detach().numpy().flatten())
with open("policy_dx.txt", "w") as f:
    f.write(" ".join(map(str, dx)))
