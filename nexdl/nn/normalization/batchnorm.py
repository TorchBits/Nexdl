from nexdl import tensor as nx
from nexdl.nn.module import Module
from nexdl.nn.parameter import Parameter


class BatchNorm2d(Module):
    def __init__(self, num_features: int, momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()

        # Learnable parameters (gamma: scale, bias: shift)
        self.gamma = Parameter(nx.ones((1, num_features, 1, 1), dtype=nx.backend.float32))
        self.bias = Parameter(nx.zeros((1, num_features, 1, 1), dtype=nx.backend.float32))

        # Running statistics (buffers, not learnable)
        self.register_buffer("running_mean_x", nx.zeros((1, num_features, 1, 1), dtype=nx.backend.float32))
        self.register_buffer("running_var_x", nx.ones((1, num_features, 1, 1), dtype=nx.backend.float32)) 

        # Hyperparameters
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(self, x: nx.Tensor) -> nx.Tensor:
        """Forward pass for BatchNorm2d."""
        if self.training:
            # Compute batch mean and variance
            mean_x = x.mean(axis=(0, 2, 3), keepdims=True)
            var_x = x.var(axis=(0, 2, 3), keepdims=True)

            # Update running statistics using exponential moving average
            self.running_mean_x = (self.momentum * self.running_mean_x + (1 - self.momentum) * mean_x).copy()
            self.running_var_x = (self.momentum * self.running_var_x + (1 - self.momentum) * var_x).copy()
        else:
            # Use running statistics in evaluation mode
            mean_x = self.running_mean_x
            var_x = self.running_var_x

        # Normalize input
        stddev_x = nx.sqrt(var_x + self.epsilon)
        standard_x = (x - mean_x) / stddev_x

        # Scale and shift
        return self.gamma * standard_x + self.bias

    def apply_gradients(self, learning_rate: float) -> None:
        """Update parameters using gradient descent."""
        self.gamma.data -= learning_rate * self.gamma.grad
        self.bias.data -= learning_rate * self.bias.grad

        # Reset gradients after applying updates
        self.gamma.grad = None
        self.bias.grad = None