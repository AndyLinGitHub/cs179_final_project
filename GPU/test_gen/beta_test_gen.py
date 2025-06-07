import torch
from torch.distributions import Beta

B = 1024
dim = 4
total = B*dim

alpha = torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(B, dim) + 1
beta = torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(B, dim) + 2

dist = Beta(alpha, beta)
with torch.no_grad():
    action = dist.rsample()
    logp = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)

action = list(action.detach().numpy().flatten())
logp = list(logp.detach().numpy().flatten())
entropy = list(entropy.detach().numpy().flatten())
with open("beta_dist_action.txt", "w") as f:
    f.write(" ".join(map(str, action)))

with open("beta_dist_logp.txt", "w") as f:
    f.write(" ".join(map(str, logp)))

with open("beta_dist_entropy.txt", "w") as f:
    f.write(" ".join(map(str, entropy)))

dlogp = torch.arange(B, dtype=torch.float32, requires_grad=True) + 1
dh = torch.arange(B, dtype=torch.float32, requires_grad=True) + 1
action = (torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(B, dim) + 1) / (total + 2)
logp = dist.log_prob(action).sum(-1)
entropy = dist.entropy().sum(-1)

grad_x = torch.autograd.grad(outputs=logp, inputs=(alpha, beta), grad_outputs=dlogp)
da_logp, db_logp = grad_x

grad_x = torch.autograd.grad(outputs=entropy, inputs=(alpha, beta), grad_outputs=dh)
da_h, db_h = grad_x

da_logp = list(da_logp.detach().numpy().flatten())
db_logp = list(db_logp.detach().numpy().flatten())
da_h = list(da_h.detach().numpy().flatten())
db_h = list(db_h.detach().numpy().flatten())
with open("beta_dist_da_logp.txt", "w") as f:
    f.write(" ".join(map(str, da_logp)))

with open("beta_dist_db_logp.txt", "w") as f:
    f.write(" ".join(map(str, db_logp)))

with open("beta_dist_da_h.txt", "w") as f:
    f.write(" ".join(map(str, da_h)))

with open("beta_dist_db_h.txt", "w") as f:
    f.write(" ".join(map(str, db_h)))