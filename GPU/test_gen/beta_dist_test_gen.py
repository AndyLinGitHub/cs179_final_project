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

dlogp = torch.ones(B, dtype=torch.float32, requires_grad=True) + 1
dh = torch.ones(B, dtype=torch.float32, requires_grad=True) + 1
# action = (torch.arange(total, dtype=torch.float32, requires_grad=True).reshape(B, dim) + 1) / (total + 2)
with torch.no_grad():
    action = dist.rsample()
logp = dist.log_prob(action).sum(-1)
entropy = dist.entropy().sum(-1)

# grad_x = torch.autograd.grad(outputs=entropy, inputs=(alpha, beta), grad_outputs=dh)
# da_h, db_h = grad_x
# grad_x = torch.autograd.grad(outputs=logp, inputs=(alpha, beta), grad_outputs=dlogp)
# da_logp, db_logp = grad_x
# grad_x = torch.autograd.grad(outputs=(logp, entropy), inputs=(alpha, beta), grad_outputs=(dlogp, dh))
grad_x = torch.autograd.grad(outputs=(logp, entropy), inputs=(alpha, beta), grad_outputs=(dlogp, dh))
da, db = grad_x
# print(da.mean(), db.mean())


# da_logp = list(da_logp.detach().numpy().flatten())
# db_logp = list(db_logp.detach().numpy().flatten())
# da_h = list(da_h.detach().numpy().flatten())
# db_h = list(db_h.detach().numpy().flatten())
# with open("beta_dist_da_logp.txt", "w") as f:
#     f.write(" ".join(map(str, da_logp)))

# with open("beta_dist_db_logp.txt", "w") as f:
#     f.write(" ".join(map(str, db_logp)))

# with open("beta_dist_da_h.txt", "w") as f:
#     f.write(" ".join(map(str, da_h)))

# with open("beta_dist_db_h.txt", "w") as f:
#     f.write(" ".join(map(str, db_h)))

da = list(da.detach().numpy().flatten())
db = list(db.detach().numpy().flatten())
with open("beta_dist_da.txt", "w") as f:
    f.write(" ".join(map(str, da)))

with open("beta_dist_db.txt", "w") as f:
    f.write(" ".join(map(str, db)))