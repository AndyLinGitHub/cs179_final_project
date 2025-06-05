import numpy as np

X = np.arange(0, 16).reshape(2, 8).astype(float)
W = np.arange(0, 128).reshape(16, 8).astype(float)
b = np.arange(0, 16).reshape(16).astype(float)

#print(W)

# Forward pass
def forward_fc(X, W, b):
    return X @ W.T + b

# Backward pass
def backward_fc(X, W, b, d_out):
    dx = d_out @ W      # (N, D_in)
    dW = d_out.T @ X    # (D_out, D_in)
    db = d_out.sum(axis=0)  # (D_out,)
    
    return dx, dW, db

# Example usage
out = forward_fc(X, W, b)
#print(out.shape)
print(list(out.flatten()))

# Simulate gradient from next layer
d_out = np.ones((2, 16))
dX, dW, db = backward_fc(X, W, b, d_out)
print(list(dX.flatten()))
print(list(dW.flatten()))
print(list(db.flatten()))

