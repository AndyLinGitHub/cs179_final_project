import torch, numpy as np
from torch.distributions import Beta

# Create tensors that track gradients
alpha = torch.tensor(
    np.arange(1, 1024 + 1, dtype=float).reshape(256, 4),
    requires_grad=True
)
beta = torch.tensor(
    np.arange(1, 2048 + 1, 2, dtype=float).reshape(256, 4),
    requires_grad=True
)
# Distribution, sample (rsample() keeps the re-parameterisation trick)
dist    = Beta(alpha, beta)
with torch.no_grad():
    action  = dist.rsample()                       # shape (256, 4)

action = torch.ones((256, 4), dtype=torch.float32, requires_grad=False) / 2
#logp = dist.log_prob(action).sum(-1)
entropy = dist.entropy()
print(entropy.shape)
entropy = entropy.sum(-1)
#print(entropy.detach().numpy().mean())

#dlogp = torch.ones(256, dtype=torch.float32, requires_grad=True)
#grad_x = torch.autograd.grad(outputs=logp, inputs=(alpha, beta), grad_outputs=dlogp)
#grad_alpha, grad_beta = grad_x

dlogp = torch.ones(256, dtype=torch.float32, requires_grad=True)
grad_x = torch.autograd.grad(outputs=entropy, inputs=(alpha, beta), grad_outputs=dlogp)
grad_alpha, grad_beta = grad_x

print(grad_alpha.shape)
print(grad_beta.shape)

print(np.sum(list(grad_alpha.detach().numpy().flatten())))
print(np.sum(list(grad_beta.detach().numpy().flatten())))
print(list(grad_alpha.detach().numpy().flatten())[:32])
#print(list(grad_beta.detach().numpy().flatten()))

"""
# Option B – functional style, returns the grads directly
grads = torch.autograd.grad(logp, (alpha, beta), grad_outputs=dy)
grad_alpha, grad_beta = grads        # each (256, 4)

# ---- gradients of log-probability ----
logp.backward(retain_graph=True)   # keep graph because we'll run another .backward()
grad_alpha_logp = alpha.grad.clone()
grad_beta_logp  = beta.grad.clone()

alpha.grad.zero_(); beta.grad.zero_()          # clear before next call

# ---- gradients of entropy ----
entropy.backward()
grad_alpha_entropy = alpha.grad
grad_beta_entropy  = beta.grad
"""
print()

import torch
import torch.special  # For digamma and loggamma functions
def beta_entropy(alpha, beta):
    # Ensure alpha and beta are float tensors with gradients enabled
    logB = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    term1 = (alpha - 1) * torch.special.digamma(alpha)
    term2 = (beta - 1) * torch.special.digamma(beta)
    term3 = (alpha + beta - 2) * torch.special.digamma(alpha + beta)
    entropy = logB - term1 - term2 + term3
    return entropy

# Example parameters (requires_grad=True to track gradients)
alpha = torch.tensor(2.0, requires_grad=True)
beta = torch.tensor(3.0, requires_grad=True)

# Compute entropy
H = beta_entropy(alpha, beta)

# Compute gradients
H.backward()

# Get gradients
dH_dalpha = alpha.grad
dH_dbeta = beta.grad

print(f'dH/dalpha: {dH_dalpha.item()}')
print(f'dH/dbeta: {dH_dbeta.item()}')
