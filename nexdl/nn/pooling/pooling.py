from nexdl import tensor as nx 
from nexdl.tensor import MaxPool2DFunction, AvgPool2DFunction
from nexdl.nn.module import Module

class MaxPool2d(Module):
    def __init__(self, kernel_size=(2, 2), stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: nx.Tensor) -> nx.Tensor:
        return MaxPool2DFunction.apply(x, self.kernel_size, self.stride)

class AvgPool2d(Module):
    def __init__(self, kernel_size=(2, 2), stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: nx.Tensor) -> nx.Tensor:
        return AvgPool2DFunction.apply(x, self.kernel_size, self.stride)



