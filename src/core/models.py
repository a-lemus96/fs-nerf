# stdlib modules
from typing import Tuple

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
    def __init__(
            self, 
            d_pos: int = 3,
            d_dir: int = 3,
            n_layers: int = 8,
            d_hidden: int = 256,
            skip: Tuple[int] = (4,),
            **kwargs: dict 
    ) -> None:
        """Constructor method. Builds a fully connected network as that of
        original NeRF paper using ReLU activation functions.
        ------------------------------------------------------------------------
        Args:
            d_pos: int. Dimension of spatial coordinates
            d_dir: int. Dimension of viewing directions
            n_layers: int. Number of hidden layers before applying bottleneck
            d_hidden: int. Width of hidden layers
            skip: Tuple[int]. Layer positions at where to concatenate input
            **kwargs: dict. Positional encoding keyword arguments
        ------------------------------------------------------------------------
        """
        super().__init__()
        self.d_pos = d_pos
        self.d_dir = d_dir
        self.skip = skip
        self.activation = nn.functional.relu # use ReLU activation fn

        # encoder for spatial coordinates
        n_freqs = kwargs['pos_fn']['n_freqs']
        log_space = kwargs['pos_fn']['log_space']
        self.__pos_encoder = PositionalEncoder(d_pos, n_freqs, log_space)
        d_pos_encoded = self.__pos_encoder.d_output # encoded output dim
        # encoder for viewing directions
        n_freqs = kwargs['dir_fn']['n_freqs']
        log_space = kwargs['dir_fn']['log_space']
        self.__dir_encoder = PositionalEncoder(d_dir, n_freqs, log_space)
        d_dir_encoded = self.__dir_encoder.d_output # encoded output dim

        # hidden layers
        hidden_layers = [
                nn.Linear(d_hidden + d_pos_encoded, d_hidden) if i in skip 
                else nn.Linear(d_hidden, d_hidden) for i in range(n_layers - 1)
        ]
        self.layers = nn.ModuleList(
                [nn.Linear(d_pos_encoded, d_hidden)] + hidden_layers
        )

        # last layers
        self.sigma = nn.Linear(d_hidden, 1)
        self.connection = nn.Linear(d_hidden, d_hidden)
        self.branch = nn.Linear(d_hidden + d_dir_encoded, d_hidden // 2)
        self.rgb = nn.Linear(d_hidden // 2, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, dirs = None):
        """Forward pass method with optional view directions argument.
        ------------------------------------------------------------------------
        Args:
            x: (N, 3)-shape torch.Tensor. Space coordinates
            dirs: Optional[(N, 3)-shape torch.Tensor]. Viewing directions"""
        # apply forward pass up to the immediate layer before the bottleneck
        x_in = self.__pos_encoder(x)
        x = x_in
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_in], dim=-1)
        
        if dirs is not None:
            # get sigma from network output
            sigma = self.sigma(x)

            # get RGB value
            x = self.connection(x)
            dir_in = self.__dir_encoder(dirs)
            x = torch.concat([x, dir_in], dim=-1)
            x = self.activation(self.branch(x)) 
            x = self.rgb(x)
            x = self.sigmoid(x)

            # Concatenate sigma and RGB value
            x = torch.concat([x, sigma], dim=-1)
        else:
            # get sigma from network output
            x = self.sigma(x)

        return x
