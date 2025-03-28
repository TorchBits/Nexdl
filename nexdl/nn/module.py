# import numpy as np
# import os
# from typing import Dict, Iterator, Tuple, Optional
# from nexdl import tensor as nx

# class Module:
#     def __init__(self):
#         """Ensure all internal attributes exist before __setattr__ can be triggered."""
#         if not hasattr(self, "_modules"):
#             super().__setattr__('_modules', {})
#         if not hasattr(self, "_parameters"):
#             super().__setattr__('_parameters', {})
#         if not hasattr(self, "_buffers"):
#             super().__setattr__('_buffers', {})

#         self.training = True  # Default to training mode

#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)

#     def forward(self, *args, **kwargs):
#         raise NotImplementedError("Subclasses must implement the forward method.")

#     def parameters(self, recurse: bool = True) -> Iterator[nx.Tensor]:
#         """Returns an iterator over module parameters."""
#         for _, param in self.named_parameters(recurse=recurse):
#             yield param

#     def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nx.Tensor]]:
#         """Returns an iterator over module parameters with names."""
#         for name, param in self._parameters.items():
#             yield prefix + name, param
#         if recurse:
#             for module_name, module in self._modules.items():
#                 yield from module.named_parameters(prefix=prefix + module_name + '.', recurse=recurse)

#     def num_parameters(self, verbose: bool = False) -> int:
#         """Counts total number of parameters in the model."""
#         total_params = sum(param.data.size for _, param in self.named_parameters())
        
#         if verbose:
#             for name, param in self.named_parameters():
#                 print(f"{name}: {param.data.size} parameters")
#             print(f"Total parameters: {total_params}")

#         return total_params

#     def buffers(self, recurse: bool = True) -> Iterator[np.ndarray]:
#         """Returns an iterator over module buffers."""
#         for _, buffer in self.named_buffers(recurse=recurse):
#             yield buffer

#     def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, np.ndarray]]:
#         """Returns an iterator over module buffers with names."""
#         for name, buffer in self._buffers.items():
#             yield prefix + name, buffer
#         if recurse:
#             for module_name, module in self._modules.items():
#                 yield from module.named_buffers(prefix=prefix + module_name + '.', recurse=recurse)

#     def children(self) -> Iterator['Module']:
#         """Returns an iterator over immediate child modules."""
#         return iter(self._modules.values())

#     def all_modules(self) -> Iterator['Module']:
#         """Returns an iterator over all modules in the network."""
#         yield self
#         for module in self._modules.values():
#             yield from module.all_modules()

#     def add_module(self, name: str, module: Optional['Module']) -> None:
#         """Adds a child module to the current module."""
#         if module is None:
#             self._modules.pop(name, None)
#         else:
#             self._modules[name] = module

#     def register_parameter(self, name, param):
#         """Registers a parameter in the module."""
#         self._parameters[name] = param  # Store parameter properly
#         return param  # Return the parameter

#     def register_buffer(self, name: str, tensor: Optional[np.ndarray]) -> None:
#         """Adds a buffer to the module."""
#         if tensor is None:
#             self._buffers.pop(name, None)
#         else:
#             self._buffers[name] = tensor

#     def zero_grad(self) -> None:
#         """Sets gradients of all parameters to zero."""
#         for param in self.parameters():
#             param.zero_grad()

#     def state_dict(self) -> Dict[str, np.ndarray]:
#         """Returns a dictionary containing the state of the module (parameters and buffers)."""
#         state_dict = {}
#         for name, param in self.named_parameters():
#             state_dict[name] = param.data
#         for name, buffer in self.named_buffers():
#             state_dict[name] = buffer
#         return state_dict

#     def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
#         """Loads the state of the module from a dictionary."""
#         for name, param in self.named_parameters():
#             if name in state_dict:
#                 param.data = state_dict[name]
#         for name, buffer in self.named_buffers():
#             if name in state_dict:
#                 self._buffers[name] = state_dict[name]

#     def save(self, path: str) -> None:
#         """Saves the module's state dictionary to a file."""
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         np.savez(path, **self.state_dict())

#     def load(self, path: str) -> None:
#         """Loads the module's state dictionary from a file."""
#         state_dict = np.load(path, allow_pickle=True)
#         self.load_state_dict(state_dict)

#     def train(self, mode: bool = True) -> None:
#         """Sets the module in training mode."""
#         self.training = mode
#         for module in self.children():
#             module.train(mode)

#     def eval(self) -> None:
#         """Sets the module in evaluation mode."""
#         self.train(False)

#     def __setattr__(self, name: str, value: object) -> None:
#         """Overrides attribute assignment to handle parameters and submodules."""
#         if hasattr(self, '_parameters') and isinstance(value, nx.Tensor):
#             self.register_parameter(name, value)
#         elif hasattr(self, '_modules') and isinstance(value, Module):
#             self.add_module(name, value)
#         elif hasattr(self, '_buffers') and isinstance(value, np.ndarray):
#             self.register_buffer(name, value)
#         else:
#             super().__setattr__(name, value)

#     def __getattr__(self, name: str) -> object:
#         """Overrides attribute access to handle parameters and submodules."""
#         if '_parameters' in self.__dict__ and name in self._parameters:
#             return self._parameters[name]
#         if '_buffers' in self.__dict__ and name in self._buffers:
#             return self._buffers[name]
#         if '_modules' in self.__dict__ and name in self._modules:
#             return self._modules[name]
#         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# class ModuleList(Module):
#     """A simple implementation of PyTorch's nn.ModuleList in NumPy."""
#     def __init__(self, modules=None):
#         super().__init__()
#         if modules:
#             for i, module in enumerate(modules):
#                 self.add_module(str(i), module)

#     def append(self, module):
#         """Add a module to the list."""
#         self.add_module(str(len(self._modules)), module)

#     def __getitem__(self, index):
#         """Retrieve a module by index."""
#         return list(self._modules.values())[index]

#     def __len__(self):
#         """Get the number of modules in the list."""
#         return len(self._modules)

#     def parameters(self):
#         """Get all parameters from the modules."""
#         params = []
#         for module in self._modules.values():
#             params.extend(module.parameters())
#         return params

from typing import (Dict, Iterator, Tuple, Optional, List, Union, Any, 
                    TypeVar, Generic, overload, Set, Callable)
import numpy as np
import os
import pickle
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from nexdl import tensor as nx

T = TypeVar('T', bound='Module')

class DeviceType(Enum):
    CPU = auto()
    # GPU = auto()  # Would be added with CUDA support

class Module(ABC):
    def __init__(self) -> None:
        self._modules: Dict[str, 'Module'] = {}
        self._parameters: Dict[str, nx.Tensor] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        self._hooks: Dict[str, List[Callable]] = {'forward': [], 'backward': []}
        self.training: bool = True
        self._device: DeviceType = DeviceType.CPU
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Apply forward hooks
        for hook in self._hooks['forward']:
            args, kwargs = hook(self, args, kwargs)
        
        result = self.forward(*args, **kwargs)
        
        # Apply post-forward hooks
        for hook in self._hooks.get('forward_post', []):
            result = hook(self, args, kwargs, result)
            
        return result

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the forward method.")

    # Device management
    def to(self, device: DeviceType) -> 'Module':
        """Move the module to the specified device."""
        self._device = device
        for param in self.parameters():
            param.device = device
        for module in self.children():
            module.to(device)
        return self
    
    def cpu(self) -> 'Module':
        """Move the module to CPU."""
        return self.to(DeviceType.CPU)

    # Parameter management
    def parameters(self, recurse: bool = True) -> Iterator[nx.Tensor]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nx.Tensor]]:
        for name, param in self._parameters.items():
            yield prefix + name, param
        if recurse:
            for module_name, module in self._modules.items():
                yield from module.named_parameters(prefix=f"{prefix}{module_name}.", recurse=recurse)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count total parameters, optionally only trainable ones."""
        return sum(
            param.data.size 
            for name, param in self.named_parameters() 
            if not only_trainable or param.requires_grad
        )

    # Buffer management
    def buffers(self, recurse: bool = True) -> Iterator[np.ndarray]:
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, np.ndarray]]:
        for name, buffer in self._buffers.items():
            yield prefix + name, buffer
        if recurse:
            for module_name, module in self._modules.items():
                yield from module.named_buffers(prefix=f"{prefix}{module_name}.", recurse=recurse)

    # Module management
    def children(self) -> Iterator['Module']:
        return iter(self._modules.values())

    def modules(self) -> Iterator['Module']:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if not isinstance(name, str):
            raise TypeError(f"module name must be a string, got {type(name)}")
        if module is None:
            self._modules.pop(name, None)
        else:
            if not isinstance(module, Module):
                raise TypeError(f"{name} is not a Module subclass")
            self._modules[name] = module

    def register_parameter(self, name: str, param: Optional[nx.Tensor]) -> None:
        if param is None:
            self._parameters.pop(name, None)
        else:
            if not isinstance(param, nx.Tensor):
                raise TypeError(f"parameter must be Tensor, got {type(param)}")
            self._parameters[name] = param

    def register_buffer(self, name: str, value: Optional[np.ndarray]) -> None:
        if value is None:
            self._buffers.pop(name, None)
        else:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"buffer must be numpy.ndarray, got {type(value)}")
            self._buffers[name] = value

    # Gradient management
    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    # State management
    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        for name, param in self.named_parameters():
            state_dict[name] = {'data': param.data, 'requires_grad': param.requires_grad}
        for name, buffer in self.named_buffers():
            state_dict[name] = buffer
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        missing_keys = []
        unexpected_keys = list(state_dict.keys())
        
        for name, param in self.named_parameters():
            if name in state_dict:
                param_data = state_dict[name]
                if isinstance(param_data, dict):  # Handle parameter dict
                    param.data = param_data['data']
                    param.requires_grad = param_data.get('requires_grad', True)
                else:  # Backward compatibility
                    param.data = param_data
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        for name, buffer in self.named_buffers():
            if name in state_dict:
                self._buffers[name] = state_dict[name]
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        if missing_keys:
            warnings.warn(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            warnings.warn(f"Unexpected keys in state_dict: {unexpected_keys}")

    # Serialization
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path: Union[str, Path]) -> None:
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    # Training mode
    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    # Hooks
    def register_forward_hook(self, hook: Callable) -> None:
        self._hooks['forward'].append(hook)

    def register_backward_hook(self, hook: Callable) -> None:
        self._hooks['backward'].append(hook)

    # Attribute access
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, nx.Tensor):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, np.ndarray):
            self.register_buffer(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        if '_modules' in self.__dict__ and name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # Extra utilities
    def apply(self, fn: Callable[['Module'], None]) -> 'Module':
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        main_str = f'{self.__class__.__name__}('
        extra_str = self.extra_repr()
        if extra_str:
            main_str += f'\n  {extra_str.replace("\n", "\n  ")}\n'
        main_str += ')'
        return main_str

class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        key = list(self._modules.keys())[idx]
        self.add_module(key, module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> 'ModuleList':
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: List[Module]) -> 'ModuleList':
        for module in modules:
            self.append(module)
        return self

    def insert(self, index: int, module: Module) -> None:
        modules = list(self._modules.items())
        modules.insert(index, (str(len(self)), module))
        self._modules.clear()
        for i, (_, m) in enumerate(modules):
            self.add_module(str(i), m)