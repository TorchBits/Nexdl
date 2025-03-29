from nexdl import tensor as nx
from nexdl.nn.module import Module
from nexdl.nn.parameter import Parameter
from nexdl.nn.convolution.conv_layers import *
from nexdl.tensor import Function

class Conv2DFunction(Function):
    @staticmethod
    def apply(x, weight, bias, stride, padding, kernel_size):
        # Compute convolution
        output = convolution_forward(x.data, weight.data, bias.data, stride, padding)

        # Wrap output in Tensor and store context for backward pass
        result = nx.tensor(output, requires_grad=x.requires_grad or weight.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = Conv2DFunction()
            result._grad_fn.inputs = (x, weight, bias)
            result._grad_fn.saved_tensors = (stride, padding, kernel_size)
        
        result.is_leaf = False
        return result

    def backward(self, grad_output):
        x, weight, bias = self.inputs
        stride, padding, kernel_size = self.saved_tensors

        # Compute gradients
        dx, dweight = convolution_backward(x.data, weight.data, grad_output.data, stride, padding, kernel_size)
        dbias = nx.backend.sum(grad_output.data, axis=(0, 2, 3))  # Sum across batch and spatial dims

        return nx.tensor(dx), nx.tensor(dweight), nx.tensor(dbias)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        2D Convolution Layer with Autograd Support.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        # Initialize weights and biases as Tensors with requires_grad=True
        self.weights = Parameter(nx.randn(out_channels, in_channels, *self.kernel_size).astype(nx.backend.float32), requires_grad=True)
    
        self.bias = Parameter(nx.zeros((out_channels,), dtype=nx.backend.float32), requires_grad=True)
        self.register_parameter("weights",self.weights)
        self.register_parameter("bias",self.bias)
        

    def forward(self, x: nx.Tensor) -> nx.Tensor:
        return Conv2DFunction.apply(x, self.weights, self.bias, self.stride, self.padding, self.kernel_size)

    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}')

