from nexdl import tensor as nx
from nexdl.nn.module import Module
from nexdl.nn.parameter import Parameter

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(nx.tensor(nx.backend.random.randn(in_features, out_features) * 0.01, requires_grad=True))
        # self.bias = Parameter(nx.tensor(nx.backend.zeros(out_features), requires_grad=True))
        self.bias = Parameter(nx.tensor(nx.backend.zeros(out_features), requires_grad=True))

        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)


    def forward(self, x):
        return x.matmul(self.weight) + self.bias

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Softmax(Module):
    def forward(self, x, dim=None):
        return x.softmax(dim)

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = nx.tensor(nx.backend.random.rand(*x.shape) > self.p, dtype=x.dtype)
            return x * mask / (1 - self.p)
        return x

class MSELoss(Module):
    def forward(self, input, target):
        return ((input - target) ** 2).mean()

class CrossEntropyLoss(Module):
    def forward(self, input, target):
        exp = nx.exp(input - nx.max(input, axis=-1, keepdims=True))
        softmax = exp / nx.sum(exp, axis=-1, keepdims=True)
        log_softmax = nx.log(softmax)
        return -nx.sum(target * log_softmax.T) / input.shape[0]

# class BCELoss(Module):
#     def forward(self, input, target):
#         """Binary Cross-Entropy Loss: -[y log(p) + (1-y) log(1-p)]"""
#         input = nx.clip(input, 1e-7, 1 - 1e-7)  # Prevent log(0) instability
#         return -nx.mean(target * nx.log(input) + (1 - target) * nx.log(1 - input))

class BCELoss(Module):
    def forward(self, input, target):
        # Clip input to prevent log(0) and ensure numerical stability
        input = nx.clip(input, 1e-7, 1 - 1e-7)
        loss = -nx.mean(target * nx.log(input) + (1 - target) * nx.log(1 - input))
        return loss


# class BCELoss(Module):
#     def forward(self, input, target):
#         # Clip input to prevent log(0) and ensure numerical stability
#         input = nx.where(input < 1e-7, nx.tensor(1e-7, dtype=input.dtype), input)
#         input = nx.where(input > 1 - 1e-7, nx.tensor(1 - 1e-7, dtype=input.dtype), input)

#         # Compute BCE Loss
#         loss = -nx.mean(target * nx.log(input) + (1 - target) * nx.log(1 - input))
#         return loss

class MAELoss(Module):
    def forward(self, input, target):
        """Mean Absolute Error (L1 Loss)"""
        return nx.mean(nx.abs(input - target))

class Tanh(Module):
    def forward(self, input, target):
        """Tanh-based loss (||tanh(x) - tanh(y)||^2)"""
        return ((nx.tanh(input) - nx.tanh(target)) ** 2).mean()

class HuberLoss(Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        """Huber Loss: Quadratic for small errors, Linear for large errors"""
        error = input - target
        mask = nx.abs(error) < self.delta
        return nx.mean(mask * 0.5 * error ** 2 + (1 - mask) * self.delta * (nx.abs(error) - 0.5 * self.delta))

class LogCoshLoss(Module):
    def forward(self, input, target):
        """Log-Cosh Loss: Smooth variant of MAE"""
        error = input - target
        return nx.mean(nx.log(nx.cosh(error)))

class KLDivLoss(Module):
    def forward(self, input, target):
        """Kullback-Leibler Divergence Loss"""
        input = nx.log_softmax(input, dim=-1)
        return nx.sum(target * (nx.log(target) - input)) / input.shape[0]

class Sigmoid(Module):
    def forward(self, x):
        return 1 / (1 + (-x).exp())
