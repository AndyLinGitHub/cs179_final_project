import numpy as np

X = np.arange(0, 16).reshape(2, 8).astype(float)
W = np.arange(0, 128).reshape(16, 8).astype(float)
b = np.arange(0, 16).reshape(16).astype(float)

print(W)

# Forward pass
def forward_fc(X, W, b):
    return X @ W.T + b

# Backward pass
def backward_fc(X, W, b, d_out):
    dX = d_out @ W.T
    dW = X.T @ d_out / X.shape[0]
    db = np.sum(d_out, axis=0, keepdims=True) / X.shape[0]
    return dX, dW, db

# Example usage
out = forward_fc(X, W, b)
print(out.shape)
print(list(out.flatten()))

# Simulate gradient from next layer
#d_out = np.random.randn(batch_size, output_size)
#dX, dW, db = backward_fc(X, W, b, d_out)

