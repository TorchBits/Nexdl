from nexlib import nexdl as nx
# from nexdl.rnn.rnn import *
from nexlib.module import ModuleClass as Module, ModuleList
from typing import Optional
# import torch
# import torch.nn as nn
from nexlib.parameter import ParameterClass as Parameter, initClass as init

class CustomRNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize parameters as Tensors with requires_grad=True
        self.W_ih = self.register_parameter(
            "W_ih", 
            Parameter(nx.tensor(nx.backend.random.randn(hidden_size, input_size), requires_grad=True))
        )
       
        self.W_hh = self.register_parameter(
            "W_hh",
            Parameter(nx.tensor(nx.backend.random.randn(hidden_size, hidden_size), requires_grad=True))
        )
        self.bias_ih = self.register_parameter(
            "bias_ih",
            Parameter(nx.tensor(nx.backend.random.randn(hidden_size), requires_grad=True))
        )
        self.bias_hh = self.register_parameter(
            "bias_hh",
            Parameter(nx.tensor(nx.backend.random.randn(hidden_size), requires_grad=True))
        )

    def forward(self, x: nx.Tensor, hidden: nx.Tensor) -> nx.Tensor:

        ih = x @ self.W_ih.T  # Input to hidden transformation
        hh = hidden @ self.W_hh.T  # Hidden to hidden transformation
        
        # Combine transformations with biases
        combined = ih + self.bias_ih + hh + self.bias_hh
        
        # Apply activation function
        return nx.tanh(combined)

class CustomRNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Initialize RNN cells
        self.layers = ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.layers.append(CustomRNNCell(layer_input_size, hidden_size))

    def forward(self, x: nx.Tensor, hidden_state: Optional[nx.Tensor] = None):
        batch_size = x.size(0)
        
        if hidden_state is None:
            hidden_state = nx.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size)
            )

        outputs = []
        for t in range(x.size(1)):
            output = x[:, t, :]
            hidden_states = []
            
            for layer in range(self.num_layers):
                hidden_l = []
                for direction in range(self.num_directions):
                    idx = layer * self.num_directions + direction
                    hidden_l.append(self.layers[idx](output, hidden_state[idx]))
                
                # Properly handle concatenation using unpacking
                if len(hidden_l) > 1:
                    output = nx.cat(hidden_l, axis=1)
                else:
                    output = hidden_l[0]
                    
                hidden_states.extend(hidden_l)
            
            outputs.append(output)
            hidden_state = nx.stack(hidden_states)

        outputs = nx.stack(outputs, axis=1)
        return outputs, hidden_state

   

if __name__ == "__main__":
    # Parameters
    input_size = 10  # Number of input features
    hidden_size = 20  # Number of hidden units
    num_layers = 2  # Number of RNN layers
    bidirectional = True  # Whether the RNN is bidirectional

    # Initialize the model
    model = CustomRNN(input_size, hidden_size, num_layers, bidirectional)
    model = CustomRNN(input_size, hidden_size, num_layers, bidirectional)
    print("Model parameters:", model.parameters())
    print("Model submodules:", model._modules)


    # Dummy input: (batch_size, sequence_length, input_size)
    batch_size = 5
    sequence_length = 7
    input_data = nx.tensor(nx.backend.random.randn(batch_size, sequence_length, input_size))

    # Forward pass
    output, hidden_state = model(input_data)

    print("Output shape:", output.shape)  # Expected: (batch_size, sequence_length, num_directions * hidden_size)
    print("Hidden state shape:", hidden_state.shape)  # Expected: (num_layers * num_directions, batch_size, hidden_size)

    # Compute a simple loss (mean of output)
    loss = output.mean().backward() # Ensure `Tensor` has `.mean()` implemented
    
    # Should trigger backpropagation
    print(model.num_parameters())

  
