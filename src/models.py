# third-party modules
import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """Positional encoder module for mapping spatial coordinates and viewing
    directions.
    ----------------------------------------------------------------------------
    """
    def __init__(self, d_input, n_freqs, log_space=False):
        """Constructor method. Builds a positional encoding
        ------------------------------------------------------------------------
        Args:
            d_input: int. number of input dimensions
            n_freqs: int. number of frequencies to be used in the mapping
            log_space: bool. compute frequencies in linear or log scale"""
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs) # output dim
        # define a list of embedding functions
        self.embedding_fns = [lambda x: x] # initialize with identity fn

        # Define frequencies in either linear or log scale
        if self.log_space:
            freqs = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freqs = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Create embedding functions and append to embedding_fns list
        for freq in freqs:
            self.embedding_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embedding_fns.append(lambda x, freq=freq: torch.cos(x * freq))


    # Forward method for passing inputs to the model
    def forward(self, x):
        """Applies positional encoding to the input coordinates.
       -------------------------------------------------------------------------
        Args:
            x: (N, dim_input)-shape torch.Tensor. N input coordinates
        Returns:
            y: (N, dim_input, 2*n_freqs)-shape torch.Tensor. Embedded coords"""
        return torch.concat([fn(x) for fn in self.embedding_fns], dim=-1)


class NeRF(nn.Module):
    """Neural Radiance Field model definition.
    ----------------------------------------------------------------------------
    """
    def __init__(self, 
                 d_input=3,
                 n_layers=8,
                 d_filter=256,
                 skip=(4,),
                 d_viewdirs=None):
        """Constructor method. Builds a fully connected network as that of
        original NeRF paper using ReLU activation functions.
        ------------------------------------------------------------------------
        Args:
            d_input: int. Dimension of spatial (with viewing dir) coords
            n_layers: int. Number of hidden layers before applying bottleneck
            d_filter: int. Width of hidden layers
            skip: Tuple[int]. Layer positions at where to concatenate input
            d_viewdirs: Optional[int]. Dimension of viewing direction coords""" 
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.activation = nn.functional.relu # use ReLU activation fn
        self.d_viewdirs = d_viewdirs

        # create model layers
        hidden_layers = [nn.Linear(d_filter + self.d_input, d_filter) if i in skip 
                         else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        self.layers = nn.ModuleList([nn.Linear(self.d_input, d_filter)] + hidden_layers)

        # create last layers
        if self.d_viewdirs is not None:
            # field has view-dependent effects, split density and color
            self.sigma_out = nn.Linear(d_filter, 1)
            self.rgb_layer = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # if the vield has no view-dependency, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(self, x, viewdirs=None):
        """Forward pass method with optional view directions argument.
        ------------------------------------------------------------------------
        Args:
            x: (N, 3)-shape torch.Tensor. Space coordinates
            viewdirs: Optional[(N, 3)-shape torch.Tensor]. Viewing directions"""
        # check if view directions are not required but given as input
        if self.d_viewdirs is None and viewdirs is not None:
            err_msg = "Model does not have view-dependent effects but viewing directions were given."
            raise ValueError(err_msg)

        # apply forward pass up to the immediate layer before the bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)
        
        # apply bottleneck pass
        if self.d_viewdirs is not None:
            # get sigma from network output
            sigma = self.sigma_out(x)

            # get RGB value
            x = self.rgb_layer(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.activation(self.branch(x)) 
            x = self.output(x)

            # Concatenate sigma and RGB value
            x = torch.concat([x, sigma], dim=-1)
        else:
            x = self.output(x)

        return x
