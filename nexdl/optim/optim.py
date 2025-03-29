from nexdl import tensor as nx 

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)  # Ensure parameters is a list
        self.lr = lr

    def step(self):
        """Performs a single optimization step (parameter update)."""
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data = param.data - self.lr * param.grad  # Ensures shape safety
            else: raise Exception 

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.parameters:
            if param.requires_grad:
                param.grad = None  # Prevents accumulation issues

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [nx.zeros_like(param.data) for param in parameters]
        self.v = [nx.zeros_like(param.data) for param in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.requires_grad:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.lr * m_hat / (nx.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters, using fill_na to handle None."""
        for param in self.parameters:
            if param.requires_grad:
                if param.grad is None:
                    # Handle the case where grad is None by using fill_na
                    param.grad = nx.fill_na(param.grad, nx.zeros_like(param.data))
                else:
                    # Ensure the gradient is set to zero
                    param.grad.fill(0)


class AdamW(Adam):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps)
        self.weight_decay = weight_decay
    
    def step(self):
        self.t += 1
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data -= self.weight_decay * param.data  # Weight decay before Adam update
        super().step()


# class RMSprop(Optimizer):
#     def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
#         super().__init__(parameters)
#         self.lr = lr
#         self.alpha = alpha
#         self.eps = eps
#         self.s = {id(param): nx.zeros_like(param.data) for param in self.parameters}
    
#     def step(self):
#         for param in self.parameters:
#             if param.requires_grad and param.grad is not None:
#                 param_id = id(param)
#                 self.s[param_id] = self.alpha * self.s[param_id] + (1 - self.alpha) * (param.grad ** 2)
#                 param.data -= self.lr * param.grad / (nx.sqrt(self.s[param_id]) + self.eps)


def clip_gradients(parameters, max_norm):
    """Clips gradients to prevent exploding gradients."""
    total_norm = nx.sqrt(nx.sum((param.grad ** 2).sum() for param in parameters if param.grad is not None))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad *= scale  # In-place update
