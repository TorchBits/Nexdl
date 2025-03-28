from nexlib import nexdl as nx
import numpy as np
import torch


a = nx.tensor(np.random.rand(2,3), requires_grad=True)
b = nx.tensor(np.random.rand(2,4), requires_grad=True)

# Concatenate
c = nx.cat((a,b), axis=1)
print(c)
# # Backward pass
c.sum().backward()
# print(a.grad.shape)  # Should be (2,3)
# print(b.grad.shape)  # Should be (2,4)
# print(torch.sum([2,3,4,5]))
