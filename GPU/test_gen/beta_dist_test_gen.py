import torch
from torch.distributions import Beta

B = 4096
dim = 4
total = B*dim

alpha = torch.full((B, dim), 1.5, dtype=torch.float32, requires_grad=True)#torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(B, dim) + 1
beta = torch.full((B, dim), 1.5, dtype=torch.float32, requires_grad=True)#(torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(B, dim) + 1) * 2 

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

# de = torch.ones(B, dtype=torch.float32, requires_grad=True) + 1
# with torch.no_grad():
#     action = dist.rsample()
# logp = dist.log_prob(action).sum(-1)
# entropy = dist.entropy().sum(-1)

# grad_x = torch.autograd.grad(outputs=(logp, entropy), inputs=(alpha, beta), grad_outputs=(dlogp, dh))
# da, db = grad_x
# # print(da.mean(), db.mean())

# da = list(da.detach().numpy().flatten())
# db = list(db.detach().numpy().flatten())
# with open("beta_dist_da.txt", "w") as f:
#     f.write(" ".join(map(str, da)))

# with open("beta_dist_db.txt", "w") as f:
#     f.write(" ".join(map(str, db)))