import numpy as np

import numpy as np

def conv2d_forward(x, W, b, stride=1, padding=0):
    N, C_in, H, W_in = x.shape
    C_out, _, K_h, K_w = W.shape

    # Padding
    x_padded = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    H_padded, W_padded = x_padded.shape[2:]

    H_out = (H_padded - K_h) // stride + 1
    W_out = (W_padded - K_w) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    for ic in range(C_in):
                        h_start = i * stride
                        w_start = j * stride
                        patch = x_padded[n, ic, h_start:h_start+K_h, w_start:w_start+K_w]
                        out[n, oc, i, j] += np.sum(patch * W[oc, ic])
            out[n, oc] += b[oc]

    return out, x_padded  # Save padded input for backward

def conv2d_backward(dout, x_padded, W, stride=1, padding=0):
    N, C_in, H_padded, W_padded = x_padded.shape
    C_out, _, K_h, K_w = W.shape
    _, _, H_out, W_out = dout.shape

    dx_padded = np.zeros_like(x_padded)
    dW = np.zeros_like(W)
    db = np.zeros(C_out)

    for n in range(N):
        for oc in range(C_out):
            db[oc] += np.sum(dout[n, oc])
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    for ic in range(C_in):
                        patch = x_padded[n, ic, h_start:h_start+K_h, w_start:w_start+K_w]
                        dW[oc, ic] += patch * dout[n, oc, i, j]
                        dx_padded[n, ic, h_start:h_start+K_h, w_start:w_start+K_w] += W[oc, ic] * dout[n, oc, i, j]

    # Remove padding from dx
    if padding > 0:
        dx = dx_padded[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_padded

    return dW, db, dx


inputs = np.arange(0, 100).reshape(2, 2, 5, 5).astype(float)
kernels = np.arange(0, 72).reshape(4, 2, 3, 3).astype(float)
bias = np.arange(0, 4).astype(float)

output, x_padded = conv2d_forward(inputs, kernels, bias, stride=1, padding=0)

dout = np.ones_like(output).astype(float)
dW, db, dx = conv2d_backward(dout, x_padded, kernels, stride=1, padding=0)

print("Output")
print(output.shape)
print(list(output.flatten()))

print("dW")
print(list(dW.flatten()))

print("db")
print(list(db.flatten()))

print("dx")
print(list(dx.flatten()))