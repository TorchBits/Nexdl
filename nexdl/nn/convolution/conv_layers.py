# from nexdl import tensor as nx
# import numpy as np
# from scipy.ndimage import convolve

# def im2col_scipy(input, kernel_size, stride, padding):
#     """Efficient im2col using scipy.ndimage."""
#     N, C, H, W = input.shape
#     KH, KW = kernel_size
#     stride_h, stride_w = stride

#     # Pad input
#     if isinstance(padding, int):
#         pad_h = pad_w = padding
#     else:
#         pad_h, pad_w = padding

#     input_padded = np.pad(
#         input, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
#     )

#     out_h = (H + 2 * pad_h - KH) // stride_h + 1
#     out_w = (W + 2 * pad_w - KW) // stride_w + 1


#     # Extract patches using stride tricks
#     patches = np.lib.stride_tricks.sliding_window_view(
#         input_padded, window_shape=(C, KH, KW), axis=(1, 2, 3)
#     )[:, :, ::stride_h, ::stride_w]

#     # Reshape into im2col format: (N, C*KH*KW, out_h*out_w)
#     return patches.reshape(N, C * KH * KW, out_h * out_w)


# def convolution_forward(input, weights, bias, stride, padding):
#     """Optimized 2D convolution using im2col with scipy."""
#     input = np.asarray(input)
#     weights = np.asarray(weights)
#     bias = np.asarray(bias)

#     N, C, H, W = input.shape
#     K, _, KH, KW = weights.shape  # K: number of filters

#     # Use scipy-based im2col
#     col_matrix = im2col_scipy(input, (KH, KW), stride, padding)  # (N, C*KH*KW, out_h*out_w)

#     # Reshape weights for matrix multiplication
#     # weights_reshaped = weights.reshape((K, -1))  # (K, C*KH*KW)
#     weights_reshaped = weights.reshape(K, C * KH * KW)  # Ensure proper shape

#     output = np.einsum('nci,kc->nki', col_matrix, weights_reshaped)  # (N, K, out_h*out_w)
#     if isinstance(padding, int):
#         pad_h = pad_w = padding
#     else:
#         pad_h, pad_w = padding

#     # Compute output dimensions
#     out_h = (H + 2 * pad_h - KH) // stride[0] + 1
#     out_w = (W + 2 * pad_w - KW) // stride[1] + 1

#     # Reshape and add bias
#     output = output.reshape((N, K, out_h, out_w)) + bias.reshape((1, K, 1, 1))

#     return output


# def convolution_backward(input, weights, doutput, strides, padding, kernel_size):
#     """Optimized backward pass for input and weight gradients using im2col."""
#     input = np.asarray(input)
#     weights = np.asarray(weights)
#     doutput = np.asarray(doutput)

#     N, C, H, W = input.shape
#     K, _, KH, KW = weights.shape

#     stride_h, stride_w = strides

#     # Step 1: Use im2col to extract patches from the input
#     col_matrix = im2col_scipy(input, (KH, KW), stride=(stride_h, stride_w), padding=padding)  # (N, C*KH*KW, out_h*out_w)

#     # Step 2: Compute gradient with respect to weights
#     doutput_col = doutput.reshape(N, K, -1)  # (N, K, out_h*out_w)

#     # Compute the gradient w.r.t. weights
#     # We use einsum to compute the outer product of patches and gradients of output
#     dweights = np.einsum('nci,nko->kic', col_matrix, doutput_col)  # (K, C*KH*KW, out_h*out_w) -> (K, C*KH*KW)
#     weights_reshaped = weights.reshape(K, C * KH * KW)  # (K, C*KH*KW)
    
#     # Compute the gradient of the input
#     dinput_col = np.einsum('nik,kc->nic', doutput_col, weights_reshaped)  # (N, out_h*out_w, C*KH*KW) -> (N, C, H, W)

#     # Step 4: Reshape dinput to match input shape
#     dinput = col2im(dinput_col, (N, C, H, W), (KH, KW), padding, stride=(stride_h, stride_w))

#     return dinput, dweights


# def col2im(col, input_shape, kernel_size, padding, stride):
#     """Reverse the im2col operation to reconstruct the input gradient."""
#     N, C, H, W = input_shape
#     KH, KW = kernel_size
#     stride_h, stride_w = stride

#     # Compute the output height and width
#     pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
#     out_h = (H + 2 * pad_h - KH) // stride_h + 1
#     out_w = (W + 2 * pad_w - KW) // stride_w + 1

#     # Initialize the gradient for the input with zeros
#     dinput = np.zeros((N, C, H, W), dtype=col.dtype)

#     # Reverse im2col to accumulate the gradients in the correct locations
#     col_reshaped = col.reshape(N, C, KH, KW, out_h, out_w)
#     for n in range(N):
#         for h in range(out_h):
#             for w in range(out_w):
#                 dinput[n, :, h * stride_h:h * stride_h + KH, w * stride_w:w * stride_w + KW] += col_reshaped[n, :, :, :, h, w]

#     return dinput

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def im2col(input, kernel_size, stride, padding):
    """Efficient im2col implementation for NCHW inputs."""
    N, C, H, W = input.shape
    KH, KW = kernel_size
    stride_h, stride_w = stride
    
    # Handle padding
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Pad the input
    input_padded = np.pad(input, 
                         ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                         mode='constant')
    
    # Calculate output dimensions
    out_h = (H + 2 * pad_h - KH) // stride_h + 1
    out_w = (W + 2 * pad_w - KW) // stride_w + 1
    
    # Extract patches using stride tricks
    patches = sliding_window_view(input_padded, (N, C, KH, KW), axis=(0, 1, 2, 3))
    patches = patches[0, 0, ::stride_h, ::stride_w]  # Get correct windows
    
    # Reshape to (N, C*KH*KW, out_h*out_w)
    return patches.transpose(0, 3, 1, 2, 4, 5).reshape(N, C * KH * KW, out_h * out_w)

def convolution_forward(input, weights, bias, stride, padding):
    """Forward pass with proper dimension handling."""
    N, C, H, W = input.shape
    K, _, KH, KW = weights.shape
    
    # Get column matrix
    col_matrix = im2col(input, (KH, KW), stride, padding)
    
    # Reshape weights and perform multiplication
    weights_reshaped = weights.reshape(K, -1)
    output = np.einsum('nci,kc->nki', col_matrix, weights_reshaped)
    
    # Calculate output dimensions
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    out_h = (H + 2 * pad_h - KH) // stride[0] + 1
    out_w = (W + 2 * pad_w - KW) // stride[1] + 1
    
    # Reshape and add bias
    return output.reshape(N, K, out_h, out_w) + bias.reshape(1, K, 1, 1)

def convolution_backward(input, weights, doutput, stride, padding):
    """Corrected backward pass implementation."""
    N, C, H, W = input.shape
    K, _, KH, KW = weights.shape
    
    # Get column matrix
    col_matrix = im2col(input, (KH, KW), stride, padding)
    
    # Reshape doutput
    doutput_col = doutput.reshape(N, K, -1)
    
    # Compute weight gradients
    dweights = np.einsum('nci,nkj->kcij', col_matrix, doutput_col)
    dweights = dweights.sum(axis=(2, 3)).reshape(K, C, KH, KW)
    
    # Compute input gradients
    weights_reshaped = weights.reshape(K, -1)
    dinput_col = np.einsum('nki,kc->nci', doutput_col, weights_reshaped)
    
    # Reconstruct input gradient
    dinput = col2im(dinput_col, input.shape, (KH, KW), padding, stride)
    
    return dinput, dweights

def col2im(col, input_shape, kernel_size, padding, stride):
    """Optimized col2im implementation."""
    N, C, H, W = input_shape
    KH, KW = kernel_size
    stride_h, stride_w = stride
    
    # Handle padding
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Calculate output dimensions
    out_h = (H + 2 * pad_h - KH) // stride_h + 1
    out_w = (W + 2 * pad_w - KW) // stride_w + 1
    
    # Initialize gradient
    dinput_padded = np.zeros((N, C, H + 2*pad_h, W + 2*pad_w), dtype=col.dtype)
    
    # Reshape column matrix
    col_reshaped = col.reshape(N, C, KH, KW, out_h, out_w)
    
    # Vectorized accumulation
    for h in range(out_h):
        for w in range(out_w):
            h_start = h * stride_h
            w_start = w * stride_w
            dinput_padded[:, :, h_start:h_start+KH, w_start:w_start+KW] += \
                col_reshaped[:, :, :, :, h, w]
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        return dinput_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    return dinput_padded