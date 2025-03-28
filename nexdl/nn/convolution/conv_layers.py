from nexdl import tensor as nx

def im2col(input, kernel_size, strides, padding):
    """Convert input into column matrix for efficient convolution."""
    N, C, H, W = input.shape
    KH, KW = kernel_size
    stride_h, stride_w = strides
    pad_h, pad_w = padding

    out_h = (H + 2 * pad_h - KH) // stride_h + 1
    out_w = (W + 2 * pad_w - KW) // stride_w + 1

    # Pad input
    input_padded = nx.backend.pad(input, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    col_matrix = nx.backend.zeros((N, C, KH, KW, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start, w_start = i * stride_h, j * stride_w
            col_matrix[:, :, :, :, i, j] = input_padded[:, :, h_start:h_start+KH, w_start:w_start+KW]

    return col_matrix.reshape((N, C * KH * KW, out_h * out_w))

def convolution_forward(input, weights, bias, strides, padding):
    """Optimized 2D convolution forward and backward pass."""
    input = nx.backend.asarray(input)
    weights = nx.backend.asarray(weights)
    bias = nx.backend.asarray(bias)

    N, C, H, W = input.shape
    K, _, KH, KW = weights.shape  # K: number of filters

    # Transform input into columns
    col_matrix = im2col(input, (KH, KW), strides, padding)  # (N, C*KH*KW, out_h*out_w)

    # Reshape weights for matrix multiplication
    weights_reshaped = weights.reshape((K, -1))  # (K, C*KH*KW)

    # Perform matrix multiplication
    output = nx.backend.einsum('nci,kc->nki', col_matrix, weights_reshaped)  # (N, K, out_h*out_w)

    # Reshape and add bias
    out_h = (H + 2 * padding[0] - KH) // strides[0] + 1
    out_w = (W + 2 * padding[1] - KW) // strides[1] + 1
    output = output.reshape((N, K, out_h, out_w)) + bias.reshape((1, K, 1, 1))

    return output

def convolution_backward(input, weights, doutput, strides, padding, kernel_size):
    """Optimized backward pass for input and weight gradients using im2col."""
    input = nx.backend.asarray(input)
    weights = nx.backend.asarray(weights)
    doutput = nx.backend.asarray(doutput)

    N, C, H, W = input.shape
    K, _, KH, KW = weights.shape

    # Transform doutput into matrix form
    out_h, out_w = doutput.shape[2], doutput.shape[3]
    doutput_col = doutput.reshape((N, K, -1))  # (N, K, out_h*out_w)

    # Reshape weights
    weights_reshaped = weights.reshape((K, -1))  # (K, C*KH*KW)

    # Compute input gradient
    dinput_col = nx.backend.einsum('nki,kc->nci', doutput_col, weights_reshaped)  # (N, C*KH*KW, out_h*out_w)

    # Reverse im2col operation
    dinput = nx.backend.zeros((N, C, H + 2 * padding[0], W + 2 * padding[1]))

    for i in range(out_h):
        for j in range(out_w):
            h_start, w_start = i * strides[0], j * strides[1]
            dinput[:, :, h_start:h_start+KH, w_start:w_start+KW] += dinput_col[:, :, i * out_w + j].reshape(N, C, KH, KW)

    # Remove padding and return
    dinput = dinput[:, :, padding[0]:H+padding[0], padding[1]:W+padding[1]]

    # Compute weight gradient
    col_matrix = im2col(input, kernel_size, strides, padding)  # (N, C*KH*KW, out_h*out_w)
    dweights = nx.backend.einsum('nki,nci->kc', doutput_col, col_matrix)  # (K, C*KH*KW)
    dweights = dweights.reshape(K, C, KH, KW)  # Reshape back

    return dinput, dweights
