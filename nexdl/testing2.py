from nexdl import nx as np, nn
from nexdl.tensory import Tensor
from nexdl.module import ModuleClass as Module
import torch

class MaxPool2DClass(Module):
    def __init__(self, kernel_size=(2, 2), stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.register_buffer("_mask", None)  # For storing flat indices of max values

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=False)

        batch_size, channels, in_height, in_width = x.data.shape
        k_height, k_width = self.kernel_size
        stride_height, stride_width = self.stride

        # Calculate output dimensions
        out_height = (in_height - k_height) // stride_height + 1
        out_width = (in_width - k_width) // stride_width + 1

        # Apply sliding window view with correct shape
        x_reshaped = x.data.reshape(batch_size * channels, 1, in_height, in_width)
        x_cols = np.lib.stride_tricks.sliding_window_view(
            x_reshaped, (1, k_height, k_width), axis=(1, 2, 3))
        x_cols = x_cols[:, :, ::stride_height, ::stride_width, :, :]
        x_cols = x_cols.reshape(batch_size * channels, out_height, out_width, k_height * k_width)

        # Compute max pooling and store flat indices
        max_values = np.max(x_cols, axis=-1)
        max_indices = np.argmax(x_cols, axis=-1)

        # Reshape output
        output = max_values.reshape(batch_size, channels, out_height, out_width)

        # Store flat indices for backward pass
        self._mask = max_indices.reshape(batch_size, channels, out_height, out_width)

        # Wrap in Tensor
        output_tensor = Tensor(output, requires_grad=x.requires_grad)
        return output_tensor

# Create a MaxPool2D layer
maxpool = MaxPool2DClass(kernel_size=(2, 2), stride=(2, 2))
torchpool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
# Set to training mode
maxpool.train()

# Small input tensor
input_tensor = Tensor(np.array([[[[1, 2], [3, 4]]]]), requires_grad=True)  # Shape: (1, 1, 2, 2)
torch_tensor = torch.tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), requires_grad=True)
# Forward pass
output = maxpool(input_tensor)
torchput = torchpool(torch_tensor)
print("Output:", output.data)
print("torchput:", torchput.data)
# Check _mask
print("_mask:", maxpool._mask)
# print("t_mask:", torchpool.mask)

# Simulate a loss and backward pass
loss = output.sum()
tloss = torchput.sum()

print("Loss:", loss.data)
print("tLoss:", tloss.data)
loss.backward()
tloss.backward()
# Check gradients
print("Input gradient:", input_tensor.grad)
print("torch Tensor", torch_tensor.grad)
