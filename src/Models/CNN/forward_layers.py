import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from numba import njit

# ---=== Helper Functions ===---

@njit
def calc_output_size(size, filter_size, padding, stride):
    return ((size - filter_size + 2 * padding) // stride) + 1

@njit
def pad_x(x, pad):
    H_x, W_x, C_x = x.shape
    
    padded = np.zeros((H_x + 2 * pad, W_x + 2 * pad, C_x), dtype = np.float32)
    padded[pad: pad + H_x, pad: pad + W_x, :] = x
    return padded

@njit
def relu(x):
    return np.maximum(x, 0)

@njit
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# ---=== Forward Propagation Per Layer ===---

@njit
def conv2d(x, W, b, stride = 1, padding = 0):
    H_x, W_x, C_in = x.shape
    H_k, W_k, _, C_out = W.shape

    # Add padding
    x_padded = x
    if padding > 0:
        x_padded = pad_x(x, padding)

    # Calculate output shape
    H_out = calc_output_size(H_x, H_k, stride, padding)
    W_out = calc_output_size(W_x, W_k, stride, padding)

    # Calculate feature map
    out = np.zeros((H_out, W_out, C_out))
    for row in range(H_out):
        for col in range(W_out):
            for channel in range(C_out):
                val = 0.0
                for i in range(H_k):
                    for j in range(W_k):
                        for k in range(C_in):
                            val += x_padded[row * stride + i, col * stride + j, k] * W[i, j, k, channel]
                val += b[channel]

                # ReLU
                out[row, col, channel] = max(val, 0) 
    return out

@njit
def pooling(x, pool_size = 2, stride = 2, type = "max"):
    H_x, W_x, C_in = x.shape

    # Calculate output shape
    H_out = calc_output_size(H_x, pool_size, 0, stride)
    W_out = calc_output_size(W_x, pool_size, 0, stride)

    out = np.zeros((H_out, W_out, C_in), dtype = np.float32)
    for row in range(H_out):
        for col in range(W_out):
            row_start = row * stride
            row_end = row_start + pool_size

            col_start = col * stride
            col_end = col_start + pool_size

            for channel in range(C_in):
                patch = x[row_start: row_end, col_start: col_end, channel]

                if type == "max":
                    out[row, col, channel] = np.max(patch)
                else:
                    out[row, col, channel] = np.mean(patch)
    return out

@njit
def flatten(x):
    return x.ravel()

@njit
def dense(x, W, b, activation):
    # Calculate output
    out = np.dot(x, W) + b
    if activation == "relu":
        return relu(out)
    elif activation == "softmax":
        return softmax(out)
    else:
        raise ValueError("Invalid dense layer activation function.")