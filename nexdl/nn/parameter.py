from nexdl import tensor as nx


class Parameter(nx.Tensor):
    def __init__(self, data, requires_grad=True, device='cpu'):
        super().__init__(data, requires_grad=requires_grad, device=device)

    def __repr__(self):
        return f"Parameter(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad}, device={self.device})"



class init:
    # Initialization methods
    def xavier_uniform(shape, gain=1.0):
        """Xavier/Glorot uniform initialization."""
        limit = gain * nx.sqrt(6 / sum(shape))
        return nx.backend.random.uniform(-limit, limit, shape)

    def xavier_normal(shape, gain=1.0):
        """Xavier/Glorot normal initialization."""
        std = gain * nx.sqrt(2 / sum(shape))
        return nx.backend.random.normal(0, std, shape)

    def he_uniform(shape):
        """He/Kaiming uniform initialization."""
        limit = nx.sqrt(6 / shape[1])
        return nx.backend.random.uniform(-limit, limit, shape)

    def he_normal(shape):
        """He/Kaiming normal initialization."""
        std = nx.sqrt(2 / shape[1])
        return nx.backend.random.normal(0, std, shape)

    def lecun_uniform(shape):
        """LeCun uniform initialization."""
        limit = nx.sqrt(3 / shape[1])
        return nx.backend.random.uniform(-limit, limit, shape)

    def lecun_normal(shape):
        """LeCun normal initialization."""
        std = nx.sqrt(1 / shape[1])
        return nx.backend.random.normal(0, std, shape)

    def uniform_(tensor, a=-0.1, b=0.1):
        """In-place uniform initialization within a given range [a, b]."""
        tensor[:] = nx.backend.random.uniform(a, b, tensor.shape)  # In-place update
        return tensor 



