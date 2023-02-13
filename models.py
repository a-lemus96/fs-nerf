from typing import Optional, Tuple, List, Union, Callable
import torch
from torch import nn

class PositionalEncoder(nn.Module):
    '''Positional encoder for position and viewing coordinates.
    '''
    def __init__(self,
               d_input: int, 
               n_freqs: int,
               log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embedding_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freqs = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freqs = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Create embedding functions and append to embedding_fns list
        for freq in freqs:
            self.embedding_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embedding_fns.append(lambda x, freq=freq: torch.sin(x* freq))


    # Forward method for passing inputs to the model
    def forward(self, 
                x) -> torch.Tensor:
        '''Apply positional encoding to the input.
        Args:
            x: [N, dim_input]. N input coordinates.
        Returns:
            y: [N, dim_input, 2 * n_freqs]. Embedded coordinates.
        '''
        return torch.concat([fn(x) for fn in self.embedding_fns], dim=-1)


class NeRF(nn.Module):
    '''Neural radiance field model.
    '''
    def __init__(self, 
                 d_input: int = 3,
                 n_layers: int = 8,
                 d_filter: int = 256,
                 skip: Tuple[int] = (4,),
                 d_viewdirs: Optional[int] = None):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.activation = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList([nn.Linear(self.d_input, d_filter)] + 
                                    [nn.Linear(d_filter + self.d_input, d_filter) if i in skip 
                                     else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)])

        # Create last layers
        if self.d_viewdirs is not None:
            # If the field has view-dependent effects, split sigma and RGB
            self.sigma_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If the vield has no view-dependency, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(self,
                x: torch.Tensor,
                viewdirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Forward pass method with optional view directions argument.
        Args:
            x: [N, 3]. Point coordinates.
            viewdirs: [N, 3]. View directions
        '''
        # Check if view directions are not required but given as input
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Model does not have view-dependent effects but viewing directions were given.')

        # Apply forward pass up to the layer just before the bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)
        
        # Apply bottleneck pass
        if self.d_viewdirs is not None:
            # Split sigma from network output
            sigma = self.sigma_out(x)

            # Obtain RGB value
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.activation(self.branch(x)) 
            x = self.output(x)

            # Concatenate sigma and RGB value
            x = torch.concat([x, sigma], dim=-1)
        else:
            x = self.output(x)

        return x
