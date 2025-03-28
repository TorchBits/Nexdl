from collections import OrderedDict
from nexdl.nn.module import Module

class Sequential(Module):
    """A sequential container that applies layers in order, like PyTorch's nn.Sequential."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = OrderedDict()
        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)  # Register as submodule

    def forward(self, x):
        """Pass input sequentially through layers."""
        for layer in self.children():  # Iterate over registered submodules
            x = layer(x)
        return x

    def parameters(self):
        """Return all parameters from contained layers."""
        params = []
        for module in self.children():
            params.extend(module.parameters())  # Collect parameters recursively
        return params

    def zero_grad(self):
        """Reset gradients for all parameters."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad[...] = 0  # Zero out gradients

    def __repr__(self):
        layers_str = "\n".join(f"  ({name}): {layer}" for name, layer in self.layers.items())
        return f"{self.__class__.__name__}(\n{layers_str}\n)"
