# from .tensor import *
# from . import nn
# from . import optim
# from . import utils

# # PyTorch-like shorthand
# tensor = Tensor

# __all__ = ['Tensor', 'tensor', 'nn', 'optim', 'utils',
# 			"sum","mean","var","tensordot","dot","matmul","cat","concatenate",
# 			"add", "mul", "sub", "div", "pow", "neg", 
#  			"cos", "sin", "tan", "acos", "atan", "asin", "sinh", "cosh", "tanh", "sigmoid", 
# 			"max", "min", "relu", "softmax", "mean", "pad", "var", "log", "sqrt", "exp", 
# 			"clip", "ones", "zeros", "zeros_like", "ones_like", "rand", "randn", "stack", 
# 			"reshape", "transpose", "squeeze", "cat", "concatenate", "abs"]

from .tensor import Tensor, sum, mean, var, tensordot, dot, matmul, cat, concatenate, \
                    add, mul, sub, div, pow, neg, cos, sin, tan, acos, atan, asin, sinh, cosh, tanh, sigmoid, \
                    max, min, relu, softmax, pad, log, sqrt, exp, clip, ones, zeros, \
                    zeros_like, ones_like, rand, randn, stack, reshape, transpose, squeeze, abs

from . import nn
from . import optim
from . import utils
from .backend import backend
# PyTorch-like shorthand
tensor = Tensor

__all__ = [
    'Tensor', 'tensor', 'nn', 'optim', 'utils',
    "sum", "mean", "var", "tensordot", "dot", "matmul", "cat", "concatenate",
    "add", "mul", "sub", "div", "pow", "neg",
    "cos", "sin", "tan", "acos", "atan", "asin", "sinh", "cosh", "tanh", "sigmoid",
    "max", "min", "relu", "softmax", "pad", "log", "sqrt", "exp",
    "clip", "ones", "zeros", "zeros_like", "ones_like", "rand", "randn", "stack",
    "reshape", "transpose", "squeeze", "abs", "backend"
]
