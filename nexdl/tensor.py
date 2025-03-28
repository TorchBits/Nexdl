from .backend import backend
from typing import List, Optional, Tuple


#_____________________________The AutoGrad function class_________________________________

class Function:
    _function_count = 0  # Unique function ID counter

    def __init__(self):
        self.name = f"{self.__class__.__name__}{Function._function_count}"
        Function._function_count += 1
        self.saved_tensors = None  # Placeholder for saved tensors

    @classmethod
    def apply(cls, *inputs, **kwargs):
        ctx = cls()  # Use an instance of Function as the context
        requires_grad = any(inp.requires_grad for inp in inputs if isinstance(inp, Tensor))

        outputs = cls.forward(ctx, *inputs, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Convert outputs to Tensor objects and attach autograd metadata
        tensors = tuple(Tensor(out, requires_grad=requires_grad) for out in outputs)

        if requires_grad:
            for tensor in tensors:
                tensor._grad_fn = ctx  # Store the function instance as grad_fn
                ctx.inputs = inputs  # Track inputs for backward()
                ctx.kwargs = kwargs
                ctx.ctx = ctx

        return tensors if len(tensors) > 1 else tensors[0]

    def save_for_backward(self, *tensors):
        """Store tensors for use in backward pass."""
        self.saved_tensors = tensors

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError



#_____________________________________FUnction implementations___________________________________________

def sum(tensor, axis=None, keepdims=False):
    """Module-level sum function that works with Tensors"""
    if not isinstance(tensor, Tensor):
        raise TypeError("Expected a Tensor")
    return SumFunction.apply(tensor, axis, keepdims)
 

def add(tensor, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return AddFunction.apply(tensor, other)

def mul(tensor, other):
    other = other if isinstance(other, Tensor) else Tensor(backend.array(other, dtype=backend.float32))
    return MulFunction.apply(tensor, other)

def sub(tensor, other):
    other = other if isinstance(other, Tensor) else Tensor(backend.array(other, dtype=backend.float32))
    return SubFunction.apply(tensor, other)

def div(tensor,other):
    other = other if isinstance(other, Tensor) else Tensor(backend.array(other, dtype=backend.float32))
    return DivFunction.apply(tensor,other)



def pow(tensor, other):
    other = other if isinstance(other, Tensor) else Tensor(backend.array(other, dtype=backend.float32))
    return PowFunction.apply(tensor, other)

def neg(tensor):
    return NegFunction.apply(tensor)

def matmul(tensor, other):
    return MatMulFunction.apply(tensor, other)


def tensor(data, requires_grad=False, backend=backend, dtype=None, device="cpu"):
    if dtype is None:
        dtype = backend.float32  # Default to float32 if not specified
    
    # Ensure data is in the correct backend format
    backend_data = backend.asarray(data, dtype=dtype)
    
    # Create full Tensor instance
    t = Tensor(data=backend_data, 
              requires_grad=requires_grad,
              backend=backend,
              dtype=dtype,
              device=device)
    
    # Mark as leaf node (creation operation)
    t.is_leaf = True
    t._grad_fn = None  # No gradient function for creation ops
    
    return t


def dot(tensor,other):
    return DotFunction.apply(tensor,other)

def tensordot(tensor,other,axes=None):
    return TensorDotFunction.apply(tensor,other,axes)

def cos(tensor):
    return CosFunction.apply(tensor)

def sin(tensor):
    return SinFunction.apply(tensor)

def tan(tensor):
    return TanFunction.apply(tensor)

def acos(tensor):
    return AcosFunction.apply(tensor)

def atan(tensor):
    return AtanFunction.apply(tensor)

def asin(tensor):
    return AsinFunction.apply(tensor)

def sinh(tensor):
    return SinhFunction.apply(tensor)

def cosh(tensor):
    return CoshFunction.apply(tensor)

def tanh(tensor):
    return TanhFunction.apply(tensor)

def sigmoid(tensor):
    return SigmoidFunction.apply(tensor)

def max(tensor,axis=None,keepdims=False):
        return MaxFunction.apply(tensor,axis,keepdims)

def min(tensor,axis=None,keepdims=False):
    return MinFunction.apply(tensor,axis,keepdims)

def relu(tensor):
    return ReLUFunction.apply(tensor)

def softmax(tensor, dim=None):
    return SoftmaxFunction.apply(tensor, dim)

def mean(tensor, axis=None,keepdims=False):
    return MeanFunction.apply(tensor, axis,keepdims)

def pad(tensor, pad_width,mode="constant",constant_values=0):
    return PadFunction.apply(tensor,pad_width,mode,constant_values)

def var(tensor,axis=None,keepdims=False,unbiased=True):
    return VarFunction.apply(tensor,axis,keepdims)

def log(tensor):
    return LogFunction.apply(tensor)

def sqrt(tensor):
    return SqrtFunction.apply(tensor)

def exp(tensor):
    return ExpFunction.apply(tensor)

def clip(tensor, min, max):
    return ClipFunction.apply(tensor,min,max)

@staticmethod
def ones(shape, dtype=backend.float32):
    """Mimic np.ones(), returns a Tensor of ones."""
    return Tensor(backend.ones(shape, dtype=dtype))

@staticmethod
def zeros(shape, dtype=backend.float32):
    """Mimic np.ones(), returns a Tensor of ones."""
    return Tensor(backend.zeros(shape, dtype=dtype))

def zeros_like(self,**kwargs):
    return ZerosLikeFunction.apply(self,**kwargs)

def ones_like(self,**kwargs):
    return OnesLikeFunction.apply(self, **kwargs)

def rand(*args, **kwargs):
    return RandFunction.apply(*args, **kwargs)

def randn(*args, **kwargs):
    return RandnFunction.apply(*args, **kwargs)

# def stack(tensors, axis=0):
#     if not isinstance(tensors, (list, tuple)):
#         raise TypeError("Tensor.stack expects a list or tuple of tensors")
#     return StackFunction.apply(*tensors, axis)

def stack(tensors, axis=0, *args, **kwargs):
    if isinstance(tensors, Tensor):
        tensors = [tensors] + list(args)
    elif isinstance(tensors, (list, tuple)):
        if args:
            raise ValueError("Cannot mix list and varargs syntax")
        tensors = list(tensors)
    else:
        raise TypeError(f"Expected Tensor or sequence of Tensors, got {type(tensors)}")
    
    # Validate all inputs
    if not tensors:
        raise ValueError("Need at least one tensor to stack")
    if not all(isinstance(t, Tensor) for t in tensors):
        raise TypeError("All inputs must be Tensors")
    
    # Check shapes
    first_shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != first_shape:
            raise ValueError(f"Shape mismatch: expected {first_shape}, got {t.shape}")
    
    # Convert axis to integer if it's a tensor
    if isinstance(axis, Tensor):
        axis = int(axis.item())
    
    # Unpack tensors list as individual arguments
    return StackFunction.apply(*tensors, axis=axis, **kwargs)

    
    

def reshape(tensor,*shape):
    return ReshapeFunction.apply(tensor, shape)

def transpose(tensor,*axes):
    return TransposeFunction.apply(tensor, axes)

def squeeze(tensor, axis):
    return SqueezeFunction.apply(tensor,axis)


def cat(tensors, axis=0):
    return CatFunction.apply(*tensors, axis=axis)

def concatenate(tensors,axis=0):
    return CatFunction.apply(*tensors,axis)


def abs(tensor):
    return AbsFunction.apply(tensor)

def linspace(start,stop,sequence_num):
    return LinspaceFunction(start,stop,sequence_num)


def no_grad():
    return NoGradFunction.apply()
 #____________________________________________The tensor class _______________________________________________________
class Tensor:
    def __init__(self, data, requires_grad=False, backend=backend, dtype=backend.float32,device="cpu"):
            self.data = backend.asarray(data, dtype=backend.float32)
            self.backend = backend
            self.requires_grad = requires_grad
            self.grad = None if not requires_grad else backend.zeros_like(self.data)
            self._grad_fn = None  # Function that produced this tensor
            self.is_leaf = True
            self.retain_grad = False
            self.dtype = dtype
            self.device = device

    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.data.dtype})"

    def __getitem__(self, index):
        return Tensor(self.data[index], requires_grad=self.requires_grad)


    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Called backward on a tensor that does not require gradients.")

        if grad is None:
            grad = backend.ones_like(self.data)

        # self.grad = grad.astype(backend.float32) if backend.is_array(grad) else backend.array(grad, dtype=backend.float32)
        self.grad = backend.asarray(grad, dtype=backend.float32)
        queue = [(self, grad)]

        while queue:
            tensor, grad = queue.pop()
            if tensor._grad_fn:
                ctx = tensor._grad_fn.ctx
                grads = tensor._grad_fn.backward(ctx, grad)

                if not isinstance(grads, tuple):
                    grads = (grads,)

                for parent, g in zip(tensor._grad_fn.inputs, grads):
                    if parent.requires_grad:
                        g_conv = backend.asarray(g, dtype=backend.float32) 
                        if parent.grad is None:
                            parent.grad = backend.zeros_like(parent.data)
                        # parent.grad += g
                        parent.grad = parent.grad.astype(backend.float32) + g_conv.astype(backend.float32)

                        queue.append((parent, g_conv))

    def retain_grad(self):
        """Enable gradient retention for non-leaf tensors."""
        if not self.is_leaf:
            self.retain_grad = True

    def item(self):
        # Ensure the tensor has only one element
        if self.data.size != 1:
            raise ValueError("item() can only be called on a tensor with a single element")
        # Return the single element as a Python scalar
        return self.data.item()

    def astype(self,value):
        return self.data.astype(value)

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def flatten(self):
        return Tensor(self.data.flatten(), requires_grad=self.requires_grad)

    @property
    def shape(self):
        return self.data.shape

    def size(self, axis=None):
        if axis is None:
            return self.data.shape  # Return full shape like PyTorch
        return self.data.shape[axis]  # Return the specific dimension size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return Tensor(self.data.transpose())

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def cat(tensor, axis):
        return cat(tensor, axis)

    def concatenate(tensor, axis):
        return concatenate(tensor,axis)


    def transpose(self, *axes):
        return transpose(self, axes)

    def __add__(self, other):
        return add(self,other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return mul(self,other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return sub(self,other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self,other):
        return div(self, other)

    def __rtruediv__(self,other):
        return self.__truediv__(other)

    def __pow__(self, other):
        return pow(self,other)

    def __neg__(self):
        return neg(self)

    def matmul(self,other):
        return matmul(self,other)

    def __matmul__(self,other):
        return matmul(self,other)

    def sum(self, axis=None, keepdims=False):
        return sum(self,axis,keepdims)

    def dot(self,other):
        return dot(self,other)

    def tensordot(self,other,axes=None):
        return tensordot(self,other,axes)

    def abs(self):
        return abs(self)

    def cos(self):
        return cos(self)

    def sin(self):
        return sin(self)

    def tan(self):
        return tan(self)

    def acos(self):
        return acos(self)

    def atan(self):
        return atan(self)

    def asin(self):
        return asin(self)

    def sinh(self):
        return sinh(self)

    def cosh(self):
        return cosh(self)

    def tanh(self):
        return tanh(self)

    def sigmoid(self):
        return sigmoid(self)

    def max(self,axis=None,keepdims=False):
        return max(self,axis,keepdims)

    def min(self,axis,keepdims):
        return min(self,axis,keepdims)

    def copy(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)


    def relu(self):
        return relu(self)

    def softmax(self, dim=None):
        return softmax(self, dim)

    def mean(self, axis=None,keepdims=False):
        return mean(self, axis,keepdims)

    def pad(self, pad_width,mode="constant",constant_values=0):
        return pad(self,pad_width,mode,constant_values)

    def var(self,axis=None,keepdims=False,unbiased=True):
        return var(self,axis,keepdims)

    def log(self):
        return log(self)

    def sqrt(self):
        return sqrt(self)

    def exp(self):
        return exp(self)

    def clip(self, min, max):
        return clip(self,min,max)


    @staticmethod
    def einsum(equation, *args):
        return EinsumFunction.apply(equation, *args)


    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)

    def stack(self, axis=0):
        return stack(self, axis)

    def reshape(self,*shape):
        return reshape(self,shape)

    def squeeze(self,axis):
        return sequeeze(self,axis)

    def to(self, device):
        if self.device == device:
            return self  # Avoid redundant conversions
        new_data = backend.to_gpu(self.data) if device == 'gpu' else backend.to_cpu(self.data)
        return Tensor(new_data, requires_grad=self.requires_grad, device=device)


    def __array__(self, dtype=None, copy=False):
        """Support for NumPy's __array__ interface."""
        if copy:
            return backend.array(self.data, dtype=dtype, copy=True)
        return backend.array(self.data, dtype=dtype)



class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a.data + b.data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class MulFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        # ctx['a'] = a 
        # ctx['b'] = b
        ctx.save_for_backward(a, b)
        return a.data * b.data

    @staticmethod
    def backward(ctx, grad_output):
        # a, b = ctx['a'],ctx['b']
        a, b = ctx.saved_tensors
        return grad_output * b.data, grad_output * a.data

class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        ctx.original_shape = tensor.shape
        ctx.new_shape = shape
        ctx.backend = tensor.backend  # ✅ Store backend in ctx

        reshaped = tensor.backend.reshape(tensor.data, shape)
        return Tensor(reshaped, requires_grad=tensor.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.backend.reshape(grad_output.data, ctx.original_shape)
        if grad_output.requires_grad:
            return Tensor(grad), None  # ✅ Return None for shape
        return grad, None


# class ReshapeFunction(Function):
#     @staticmethod
#     def forward(ctx, tensor, shape):
#         # Store original shape as an attribute, not dictionary item
#         ctx.original_shape = tensor.shape
#         ctx.new_shape = shape
        
#         # Perform reshape using backend
#         reshaped = tensor.backend.reshape(tensor.data, shape)
#         return Tensor(reshaped, requires_grad=tensor.requires_grad)

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Restore original shape
#         return Tensor(ctx.backend.reshape(grad_output.data, ctx.original_shape), None)

class SqueezeFunction(Function):
    @staticmethod
    def forward(ctx, tensor, axis=None):
        # Save original shape and axis for backward pass
        ctx.save_for_backward(tensor)
        ctx.original_shape = tensor.data.shape
        ctx.axis = axis
        
        # Perform squeeze operation
        backend = tensor.backend
        squeezed_data = backend.squeeze(tensor.data, axis=axis)
        
        # Return new tensor with proper gradient tracking
        return Tensor(squeezed_data, requires_grad=tensor.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        if not isinstance(grad_output, Tensor):
            return None, None
            
        # Retrieve saved information
        tensor = ctx.saved_tensors[0]
        original_shape = ctx.original_shape
        axis = ctx.axis
        
        # Compute gradient by unsqueezing back to original shape
        backend = grad_output.backend
        grad_input = backend.reshape(grad_output.data, original_shape)
        
        # Only return gradient for tensor (not for axis)
        return Tensor(grad_input, requires_grad=False), None


class TransposeFunction(Function):
    @staticmethod
    def forward(ctx, tensor, axes):
        ctx["axes"] = axes
        return tensor.data.transpose(axes)

    @staticmethod
    def backward(ctx, grad_output):
        axes = ctx["axes"]
        return grad_output.transpose(backend.argsort(axes)),



class DivFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data / b.data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output / b.data
        grad_b = -grad_output * a.data / (b.data ** 2)
        return grad_a, grad_b


class AbsFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(backend.abs(a.data))  # Element-wise absolute value

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = grad_output * nx.backend.sign(a.data)  # Gradient of abs(x) is sign(x)
        return grad_a


class SubFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a.data - b.data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output


class PowFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data ** b.data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        eps = 1e-8
        grad_a = grad_output * b.data * (a.data ** (b.data - 1))
        grad_b = grad_output * (a.data ** b.data) * backend.log(backend.clip(a.data, eps, None))
        return grad_a, grad_b


class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data @ b.data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.data.T, a.data.T @ grad_output


class NegFunction(Function):
    @staticmethod
    def forward(ctx, a):
        return -a.data

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output,


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.maximum(0, a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output * (a.data > 0),


class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        exp = backend.exp(a.data - backend.max(a.data, axis=dim, keepdims=True))
        softmax = exp / backend.sum(exp, axis=dim, keepdims=True)
        ctx.save_for_backward(softmax)
        return softmax

    @staticmethod
    def backward(ctx, grad_output):
        softmax = ctx.saved_tensors[0]
        return grad_output * softmax * (1 - softmax),


class SumFunction(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a, axis)
        return a.backend.sum(a.data, axis=axis, keepdims=keepdims)  # Use `a.backend`

    @staticmethod
    def backward(ctx, grad_output):
        a, axis = ctx.saved_tensors
        if axis is not None:
            grad_output = a.backend.expand_dims(grad_output, axis=axis)
        return a.backend.broadcast_to(grad_output, a.shape),  # Ensure shape matches


class StackFunction(Function):
    @staticmethod
    def forward(ctx, *tensors, axis=0, **kwargs):
        # Ensure axis is integer
        axis = int(axis) if not isinstance(axis, int) else axis
        
        # Validate we have at least one tensor
        if not tensors:
            raise ValueError("Need at least one tensor to stack")
        
        ctx.save_for_backward(*tensors)
        ctx.axis = axis
        ctx.num_tensors = len(tensors)
        
        # Get backend from first tensor
        backend = tensors[0].backend
        stacked_data = backend.stack([t.data for t in tensors], axis=axis)
        
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(stacked_data, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        if not isinstance(grad_output, Tensor):
            return (None,) * ctx.num_tensors
        
        # Split gradients
        grads = grad_output.backend.split(
            grad_output.data,
            ctx.num_tensors,
            axis=ctx.axis
        )
        
        # Return gradients only for tensors that require grad
        return tuple(
            Tensor(g, requires_grad=False) if ctx.saved_tensors[i].requires_grad else None
            for i, g in enumerate(grads)
            )


class MeanFunction(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a, axis)
        return backend.mean(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        a, axis = ctx.saved_tensors
        if axis is not None:
            grad_output = backend.expand_dims(grad_output, axis=axis)
        return backend.broadcast_to(grad_output / backend.prod(a.data.shape), a.data.shape),


class DotFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return backend.dot(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return backend.dot(grad_output, b.data.T), backend.dot(a.data.T, grad_output)


class TensorDotFunction(Function):
    @staticmethod
    def forward(ctx, a, b, axes):
        ctx.save_for_backward(a, b, axes)
        return backend.tensordot(a.data, b.data, axes)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, axes = ctx.saved_tensors
        if isinstance(axes, int):
            axes_a = list(range(-axes, 0))
            axes_b = list(range(0, axes))
        elif isinstance(axes, (list, tuple)) and len(axes) == 2:
            axes_a, axes_b = axes
        else:
            raise ValueError("Invalid axes argument for tensordot.")

        grad_a = backend.tensordot(grad_output, b.data, axes=(axes_b, axes_b))
        grad_b = backend.tensordot(a.data, grad_output, axes=(axes_a, axes_a))
        return grad_a, grad_b


class CosFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.cos(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return -grad_output * backend.sin(a.data),

class SinFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.sin(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output * backend.cos(a.data),

class TanFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.tan(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output / (backend.cos(a.data) ** 2),

class AcosFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.arccos(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return -grad_output / backend.sqrt(1 - a.data ** 2),

class AsinFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.arcsin(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output / backend.sqrt(1 - a.data ** 2),

class AtanFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.arctan(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output / (1 + a.data ** 2),

class SinhFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.sinh(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output * backend.cosh(a.data),

class CoshFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.cosh(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output * backend.sinh(a.data),

class TanhFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.tanh(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output * (1 - backend.tanh(a.data) ** 2),

class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, a):
        sigmoid = 1 / (1 + backend.exp(-a.data))
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid = ctx.saved_tensors[0]
        return grad_output * sigmoid * (1 - sigmoid),


class CatFunction(Function):
    @staticmethod
    def forward(ctx, *tensors, axis=0):
        # Convert all inputs to Tensors
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        
        if not tensors:
            raise ValueError("Need at least one tensor to concatenate")
            
        backend = tensors[0].backend
        requires_grad = any(t.requires_grad for t in tensors)

        # Explicitly use numpy's built-in max
        max_dims = backend.max([t.ndim for t in tensors])

        # Reshape tensors to match dimensions
        reshaped_tensors = []
        for t in tensors:
            if t.ndim < max_dims:
                # new_shape = t.shape + (1,) * (max_dims - t.ndim)
                new_shape = (1,) * (max_dims - t.ndim) + t.shape  # Pad on the left instead of right

                reshaped_tensors.append(Tensor(backend.reshape(t.data, new_shape)))
            else:
                reshaped_tensors.append(t)


        # Verify shapes
        first_shape = [s for i, s in enumerate(reshaped_tensors[0].shape) if i != axis]
        for t in reshaped_tensors[1:]:
            other_shape = [s for i, s in enumerate(t.shape) if i != axis]
            if other_shape != first_shape:
                raise ValueError("All tensors must have same shape except in concatenation dimension")

        ctx.save_for_backward(*reshaped_tensors)
        ctx.axis = axis

        # Concatenate
        print("Shapes before concatenation:", [t.shape for t in reshaped_tensors])

        concatenated = backend.concatenate([t.data for t in reshaped_tensors], axis=axis)
        print("Concatenated shape:", concatenated.shape)

        return Tensor(concatenated, requires_grad=requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        axis = ctx.axis

        grads = []
        start = 0
        for t in tensors:
            end = start + t.shape[axis]
            slices = [slice(None)] * grad_output.ndim
            slices[axis] = slice(start, end)
            grads.append(Tensor(grad_output.data[tuple(slices)], requires_grad=False))
            start = end

        return tuple(grads)

# class CatFunction(Function):
#     @staticmethod
#     def forward(ctx, *tensors, axis=0):
#         if len(tensors) == 1 and isinstance(tensors[0], list):
#             tensors = tuple(tensors[0])
#         ctx.save_for_backward(tensors, axis)
#         return backend.concatenate([t.data for t in tensors], axis=axis)

#     @staticmethod
#     def backward(ctx, grad_output):
#         tensors, axis = ctx.saved_tensors
#         grads = []
#         start = 0
#         for t in tensors:
#             end = start + t.data.shape[axis]
#             slice_obj = [slice(None)] * grad_output.ndim
#             slice_obj[axis] = slice(start, end)
#             grads.append(grad_output[tuple(slice_obj)])
#             start = end
#         return tuple(grads)



class MaxFunction(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        max_vals = backend.max(a.data, axis=axis, keepdims=keepdims)
        ctx.save_for_backward(a, max_vals, axis, keepdims)
        return max_vals

    @staticmethod
    def backward(ctx, grad_output):
        a, max_vals, axis, keepdims = ctx.saved_tensors
        mask = (a.data == max_vals)
        mask = mask / backend.sum(mask, axis=axis, keepdims=True)
        if not keepdims and axis is not None:
            grad_output = backend.expand_dims(grad_output, axis=axis)
        grad_output = backend.broadcast_to(grad_output, a.data.shape)
        return grad_output * mask,

class MinFunction(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        min_vals = backend.min(a.data, axis=axis, keepdims=keepdims)
        ctx.save_for_backward(a, min_vals, axis, keepdims)
        return min_vals

    @staticmethod
    def backward(ctx, grad_output):
        a, min_vals, axis, keepdims = ctx.saved_tensors
        mask = (a.data == min_vals)
        mask = mask / backend.sum(mask, axis=axis, keepdims=True)
        if not keepdims and axis is not None:
            grad_output = backend.expand_dims(grad_output, axis=axis)
        grad_output = backend.broadcast_to(grad_output, a.data.shape)
        return grad_output * mask,

class PadFunction(Function):
    @staticmethod
    def forward(ctx, a, pad_width, mode="constant", constant_values=0):
        ctx.save_for_backward(a, pad_width)
        return backend.pad(a.data, pad_width, mode=mode, constant_values=constant_values)

    @staticmethod
    def backward(ctx, grad_output):
        a, pad_width = ctx.saved_tensors
        slices = tuple(slice(p[0], grad_output.shape[i] - p[1]) for i, p in enumerate(pad_width))
        return grad_output[slices],


class VarFunction(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False, unbiased=True):
        mean = backend.mean(a.data, axis=axis, keepdims=True)
        n = a.data.shape[axis] if axis is not None else a.data.size

        if unbiased and n > 1:
            correction = n / (n - 1)  # Apply Bessel's correction
        else:
            correction = 1.0

        var = backend.var(a.data, axis=axis, keepdims=keepdims) * correction
        ctx.save_for_backward(a, mean, axis, n, unbiased)
        return var

    @staticmethod
    def backward(ctx, grad_output):
        a, mean, axis, n, unbiased = ctx.saved_tensors
        if unbiased and n > 1:
            correction = n / (n - 1)  # Bessel's correction factor
        else:
            correction = 1.0

        if axis is not None:
            grad_output = backend.expand_dims(grad_output, axis=axis)

        grad_input = (2 * (a.data - mean) / (n - int(unbiased))) * grad_output * correction
        return grad_input, None, None, None  # None for axis, keepdims, and unbiased


class OnesFunction(Function):
    @staticmethod
    def forward(ctx, shape, **kwargs):
        return backend.ones(shape, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        return None


class ZerosFunction(Function):
    @staticmethod
    def forward(ctx, shape, **kwargs):
        return backend.zeros(shape, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        return None


class ZerosLikeFunction(Function):
    @staticmethod
    def forward(ctx, a, **kwargs):
        return backend.zeros_like(a.data, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        return None


class OnesLikeFunction(Function):
    @staticmethod
    def forward(ctx, a, **kwargs):
        return backend.ones_like(a.data, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        return None


# class RandFunction(Function):
#     @staticmethod
#     def forward(ctx, *args, **kwargs):
#         return backend.random.rand(*args, **kwargs)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return None


# class RandnFunction(Function):
#     @staticmethod
#     def forward(ctx, *args, **kwargs):
#         return backend.random.randn(*args, **kwargs)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return None

class RandFunction(Function):
    @staticmethod
    def forward(ctx, *size, requires_grad=False):
        # Generate random data (uniform distribution [0, 1))
        data = backend.random.rand(*size)
        ctx.requires_grad = requires_grad
        return Tensor(data, requires_grad=requires_grad)
    
    # No backward needed - random generation is non-differentiable

class RandnFunction(Function):
    @staticmethod
    def forward(ctx, *size, requires_grad=False):
        # Generate random data
        data = backend.random.randn(*size)
        
        # Create FULL Tensor with all methods
        tensor = Tensor(data, requires_grad=requires_grad)
        
        # Mark as leaf node (since random generation has no gradient)
        tensor.is_leaf = True
        return tensor

class LogFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.log(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output / a.data


class SqrtFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.sqrt(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output / (2 * backend.sqrt(a.data)),


class ExpFunction(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return backend.exp(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return grad_output * backend.exp(a.data),


class ClipFunction(Function):
    @staticmethod
    def forward(ctx, a, min=None, max=None):
        ctx.save_for_backward(a)
        ctx.min = min
        ctx.max = max
        return backend.clip(a.data, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        min, max = ctx.min, ctx.max
        
        mask = backend.ones_like(a.data, dtype=bool)
        if min is not None:
            mask &= (a.data >= min)
        if max is not None:
            mask &= (a.data <= max)

        return grad_output * mask,


class EinsumFunction(Function):
    @staticmethod
    def forward(ctx, equation, *args):
        ctx.save_for_backward(*args)
        ctx.equation = equation
        return backend.einsum(equation, *[arg.data for arg in args])

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.saved_tensors
        equation = ctx.equation

        input_labels, output_label = equation.split("->")
        input_labels = input_labels.split(",")

        grads = []
        for i, arg in enumerate(args):
            grad_labels = input_labels[:]
            grad_labels[i] = output_label
            grad_eq = ",".join(grad_labels) + "->" + input_labels[i]
            grad_inputs = backend.einsum(grad_eq, grad_output, *[arg.data for j, arg in enumerate(args) if j != i])
            grads.append(grad_inputs)

        return tuple(grads)



class MaxPool2DFunction(Function):
    @staticmethod
    def apply(x, kernel_size, stride):
        batch_size, channels, in_height, in_width = x.data.shape
        k_height, k_width = kernel_size
        stride_height, stride_width = stride

        # Compute output dimensions
        out_height = (in_height - k_height) // stride_height + 1
        out_width = (in_width - k_width) // stride_width + 1

        # Sliding window
        x_reshaped = x.data.reshape(batch_size * channels, 1, in_height, in_width)
        x_cols = backend.lib.stride_tricks.sliding_window_view(
            x_reshaped, (k_height, k_width), axis=(-2, -1)
        )
        x_cols = x_cols[:, :, ::stride_height, ::stride_width, :, :]
        x_cols = x_cols.reshape(batch_size * channels, out_height, out_width, k_height * k_width)

        # Max pooling
        # max_values = nx.max(x_cols, axis=-1)
        # max_indices = nx.argmax(x_cols, axis=-1)
        # Compute max pooling
        max_values = backend.max(x_cols, axis=-1)
        max_indices = backend.argmax(x_cols, axis=-1)

        # Ensure max_indices has correct shape (batch_size, channels, out_height, out_width, 1)
        max_indices = max_indices.reshape(batch_size, channels, out_height, out_width, 1)


        # Create output tensor
        result = Tensor(max_values.reshape(batch_size, channels, out_height, out_width),
                        requires_grad=x.requires_grad)

        # Store context for backward pass
        if x.requires_grad:
            result._grad_fn = MaxPool2DFunction()
            result._grad_fn.inputs = (x,)
            result._grad_fn.saved_tensors = (max_indices, x.shape, kernel_size, stride)

        result.is_leaf = False
        return result

    def backward(self, grad_output):
        x, = self.inputs
        max_indices, x_shape, kernel_size, stride = self.saved_tensors
        batch_size, channels, out_height, out_width = grad_output.data.shape
        k_height, k_width = kernel_size
        stride_height, stride_width = stride

        # Initialize gradient w.r.t. input
        grad_input = backend.zeros(x_shape, dtype=x.data.dtype)

        # Propagate gradients
        for i in range(out_height):
            for j in range(out_width):
                mask = max_indices[:, :, i, j,0]  # Indices of max values in flattened window
                grad_input[:, :, i * stride_height + mask // k_width, j * stride_width + mask % k_width] += grad_output[:, :, i, j]

        return Tensor(grad_input)  # Return gradient w.r.t. input


class AvgPool2DFunction(Function):
    @staticmethod
    def apply(x, kernel_size, stride):
        batch_size, channels, in_height, in_width = x.data.shape
        k_height, k_width = kernel_size
        stride_height, stride_width = stride

        # Compute output dimensions
        out_height = (in_height - k_height) // stride_height + 1
        out_width = (in_width - k_width) // stride_width + 1

        # Apply sliding window
        x_reshaped = x.data.reshape(batch_size * channels, 1, in_height, in_width)
        x_cols = backend.lib.stride_tricks.sliding_window_view(
            x_reshaped, (k_height, k_width), axis=(-2, -1)
        )
        x_cols = x_cols[:, :, ::stride_height, ::stride_width, :, :]
        x_cols = x_cols.reshape(batch_size * channels, out_height, out_width, k_height * k_width)

        # Compute average pooling
        avg_values = backend.mean(x_cols, axis=-1)
        
        # Create output tensor
        result = Tensor(avg_values.reshape(batch_size, channels, out_height, out_width),
                        requires_grad=x.requires_grad)

        # Store context for backward pass
        if x.requires_grad:
            result._grad_fn = AvgPool2DFunction()
            result._grad_fn.inputs = (x,)
            result._grad_fn.saved_tensors = (x.shape, kernel_size, stride)

        result.is_leaf = False
        return result

    def backward(self, grad_output):
        x, = self.inputs
        x_shape, kernel_size, stride = self.saved_tensors
        batch_size, channels, out_height, out_width = grad_output.data.shape
        k_height, k_width = kernel_size
        stride_height, stride_width = stride

        # Initialize gradient for input
        grad_input = backend.zeros(x_shape, dtype=x.data.dtype)

        # Propagate gradients evenly across the pooling window
        grad_per_element = grad_output / (k_height * k_width)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + k_height
                w_start = j * stride_width
                w_end = w_start + k_width
                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_per_element[:, :, i, j]

        return Tensor(grad_input)  # Return gradient w.r.t. input

