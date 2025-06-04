import numpy as np

x = np.arange(-50, 50).reshape(2, 2, 5, 5).astype(float)
y = np.maximum(x, 0)

print("Output")
print(list(y.flatten()))

dy = np.ones_like(x).astype(float)
dx = dy * (x > 0).astype(np.float32)
print("dx")
print(list(dx.flatten()))