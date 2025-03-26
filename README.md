# NexDL: A Custom Deep Learning Library

NexDL is a lightweight numpy-based deep learning framework inspired by PyTorch, designed for flexibility and extensibility. It provides an intuitive autograd system, a variety of neural network layers, optimizers, and loss functions.

## Features
- **Autograd System**: Tracks operations for automatic differentiation.
- **Custom Tensor Class**: Subclasses NumPy arrays while supporting gradient tracking.
- **Neural Network Modules**: Linear layers, activation functions, and loss functions.
- **Optimizers**: Includes SGD with gradient updates.
- **Custom Backend**: Supports NumPy amd Cupy (for use with GPUs) operations while ensuring gradient computation.
