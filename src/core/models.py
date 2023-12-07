# stdlib modules
import math
from typing import List, Optional, Tuple

# third-party modules
import torch
from torch import nn, Tensor


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
        super(PositionalEncoder, self).__init__()
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
        super(NeRF, self).__init__()
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

class Sine(nn.Module):
    """
    Sine activation function module.
    ----------------------------------------------------------------------------
    """

    def __init__(self, w: float = 1.) -> None:
        """
        Constructor method. Builds a sine activation function module.
        ------------------------------------------------------------------------
        Args:
            w: float. Frequency of the sine function
        """
        super(Sine, self).__init__()
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method.
        ------------------------------------------------------------------------
        Args:
            x: (N, d)-shape torch.Tensor. Input tensor
        Returns:
            (N, d)-shape torch.Tensor. Output tensor
        """
        return torch.sin(self.w * x)


class SirenLinear(nn.Module):
    """
    Linear layer for SIREN model. It is based on the original implementation of
    [1].
    ----------------------------------------------------------------------------
    References:
        [1] SiNeRF: Sinusoidal Neural Radiance Fields for Joint Pose Estimation
            and Scene Reconstruction. Yitong Xia, Hao Tang, Radu Timofte, Luc 
            Van Gool. BMCV 2022. https://arxiv.org/abs/2210.04553
    """
    def __init__(
            self,
            in_dim: int = 256,
            out_dim: int = 256,
            use_bias: bool = True,
            w: float = 1.,
            is_first: bool = False
    ) -> None:
        """
        Constructor method. Builds a linear layer for SIREN-based model.
        ------------------------------------------------------------------------
        Args:
            in_dim: int. Input dimension
            out_dim: int. Output dimension
            use_bias: bool. Whether to use bias
            w: float. Frequency of the sine function
            is_first: bool. Whether the layer is the first layer
        """
        super(SirenLinear, self).__init__()
        self.fc_layer = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.use_bias = use_bias
        self.activation = Sine(w)
        self.is_first = is_first
        self.in_dim = in_dim
        self.w = w
        self.c = 6. # constant for initialization
        self.__init_weights()
        
    def __init_weights(self) -> None:
        """
        Initializes the weights of the sine linear layer.
        ------------------------------------------------------------------------
        """
        with torch.no_grad():
            dim = self.in_dim
            sigma = (1 / dim) if self.is_first else (math.sqrt(self.c / dim))
            self.fc_layer.weight.uniform_(-sigma, sigma)
            if self.use_bias and self.fc_layer.bias is not None:
                self.fc_layer.bias.uniform_(-sigma, sigma)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass method.
        ------------------------------------------------------------------------
        Args:
            x: (N, d)-shape torch.Tensor. Input tensor
        Returns:
            out: (N, d)-shape torch.Tensor. Output tensor
        """
        out = self.fc_layer(x)
        out = self.activation(out)
        
        return out


class SiNeRF(nn.Module):
    """
    SIREN MLP model for NeRF. It is based on the original implementation of [1].
    ----------------------------------------------------------------------------
    References:
        [1] SiNeRF: Sinusoidal Neural Radiance Fields for Joint Pose Estimation
            and Scene Reconstruction. Yitong Xia, Hao Tang, Radu Timofte, Luc 
            Van Gool. BMCV 2022. https://arxiv.org/abs/2210.04553
    """
    def __init__(
            self, 
            pos_dim: int = 3,
            dir_dim: int = 3,
            width: int = 256,
            alpha: List[float] = [30., 1., 1., 1., 1., 1., 1., 1.],
    ) -> None:
        """
        Constructor method. Builds a SIREN MLP model for NeRF.
        ------------------------------------------------------------------------
        Args:
            pos_dim: int. Dimension of the position input
            dir_dim: int. Dimension of the direction input
            width: int. Base width of the hidden layers
            alpha: List[float]. List of alpha values for each layer
        """
        super(SiNeRF, self).__init__()
        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        self.alpha = alpha

        hidden = [SirenLinear(width, width, True, a) for a in alpha[1:]]

        self.first_layers = nn.Sequential(
                SirenLinear(pos_dim, width, True, alpha[0], True),
                *hidden
        )
        self.sigma_layers = nn.Sequential(
                SirenLinear(width, width // 2, True),
                nn.Linear(width // 2, 1, True),
                nn.ReLU()
        )
        self.fc_feature = nn.Linear(width, width)
        self.rgb_layers = nn.Sequential(
                SirenLinear(width + dir_dim, width // 2, True),
                nn.Linear(width // 2, 3, True),
                nn.Sigmoid()
        )
    
    def forward(
            self,
            x: torch.Tensor,
            dirs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass method. It computes density and RGB values. If no viewing
        directions are provided, only density is computed.
        ------------------------------------------------------------------------
        Args:
            x: (N, 3)-shape torch.Tensor. Spatial coords
            dirs: (N, 3)-shape torch.Tensor. Viewing directions
        Returns:
            (N, 1)((N, 4))-shape torch.Tensor. Density (and RGB) values
        """
        x = self.first_layers(x)
        if dirs is not None:
            sigma = self.sigma_layers(x)
            x = self.fc_feature(x)
            x = torch.cat([x, dirs], dim=-1)
            x = torch.cat([self.rgb_layers(x), sigma], dim=-1)
        else:
            x = self.sigma_layers(x)

        return x

class SiReNeRF(nn.Module):
    """
    Siren-residual MLP model for NeRF.
    ----------------------------------------------------------------------------
    """
    def __init__(
            self,
            pos_dim: int = 3,
            dir_dim: int = 3,
            width: int = 256,
            alpha: List[float] = [30., 1., 1., 1., 1., 1., 1., 1.]
    ) -> None:
        """
        Constructor method. Builds a residual SIREN MLP model for NeRF.
        ------------------------------------------------------------------------
        Args:
            pos_dim: int. Dimension of the position input
            dir_dim: int. Dimension of the direction input
            width: int. Base width of the hidden layers
            alpha: List[float]. List of alpha values for each layer
        """
        super(SiReNeRF, self).__init__()

        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        self.alpha = alpha

        first = [SirenLinear(pos_dim, width, True, alpha[0], True)]
        in_dims = [pos_dim] + [width] * (len(alpha) - 2)
        hidden = [SirenLinear(width + dim, width, True, a) 
                  for dim, a in zip(in_dims, alpha[1:])]

        self.hidden_layers = nn.ModuleList(first + hidden)

        self.sigma_layers = nn.Sequential(
                SirenLinear(width, width // 2, True),
                nn.Linear(width // 2, 1, True),
                nn.ReLU()
        )
        self.fc_feature = nn.Linear(width, width)
        self.rgb_layers = nn.Sequential(
                SirenLinear(width + dir_dim, width // 2, True),
                nn.Linear(width // 2, 3, True),
                nn.Sigmoid()
        )

    def forward(
            self,
            x: torch.Tensor,
            dirs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass method. It computes density and RGB values. If no viewing
        directions are provided, only density is computed.
        ------------------------------------------------------------------------
        Args:
            x: (N, 3)-shape torch.Tensor. Spatial coords
            dirs: (N, 3)-shape torch.Tensor. Viewing directions
        Returns:
            (N, 1)((N, 4))-shape torch.Tensor. Density (and RGB) values
        """
        for layer in self.hidden_layers[:-1]:
            x = torch.concat([layer(x), x], dim=-1)
        x = self.hidden_layers[-1](x) # last layer

        if dirs is not None:
            sigma = self.sigma_layers(x)
            x = self.fc_feature(x)
            x = torch.cat([x, dirs], dim=-1)
            x = torch.cat([self.rgb_layers(x), sigma], dim=-1)
        else:
            x = self.sigma_layers(x)

        return x


class FourierNN(nn.Module):
    """
    Shallow SIREN with one hidden layer.
    ----------------------------------------------------------------------------
    """
    def __init__(self, pos_dim: int = 3, width: int = 256, w: float = 1.):
        super(FourierNN, self).__init__()
        self.shallow = nn.Sequential(
                SirenLinear(3, width, w=w), 
                Sine(w),
                nn.Linear(width, width, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.shallow(x)


class FourierEnsemble(nn.Module):
    """
    Fourier ensemble of tiny FourierNN models.
    ----------------------------------------------------------------------------
    """
    def __init__(
            self, 
            n: int = 7,
            pos_dim: int = 3,
            width: int = 128,
            fmin: int = 2.,
            fmax: int = 500.,
    ):
        super(FourierEnsemble, self).__init__()
        # exponential spacing of frequencies using base 2
        freqs = torch.logspace(math.log2(fmin), math.log2(fmax), n, base=2)
        self.freqs = freqs
        # freqs = torch.linspace(fmin, fmax, n)
        # initialize ensemble of FourierNN models
        self.ensemble = nn.ModuleList([FourierNN(pos_dim, width, w=freq) 
                                       for freq in freqs])
        self.mask = [True] + [False] * (n - 1)
        self.toggle_ensemble(0) # freeze all but first
        # initialize linear layers
        self.linear = nn.Linear(n*width, 128)
        self.sigma_layer = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
        self.rgb_layer = nn.Sequential(nn.Linear(128 + pos_dim, 3), nn.Sigmoid())


    def forward(self, x: Tensor, d: Optional[Tensor] = None) -> Tensor:
        x = [model(x) for model in self.ensemble] # ensemble forward pass
        # apply binary mask, (y is a dummy variable)
        x = [y * 0. if not self.mask[i] else y for i, y in enumerate(x)]
        x = self.linear(torch.cat(x, dim=-1)) # linear layer

        if d is not None:
            x = [self.rgb_layer(torch.cat([x, d], dim=-1)), self.sigma_layer(x)]
            return torch.cat(x, dim=-1)
        else:
            return self.sigma_layer(x)


    def toggle_ensemble(self, i: int) -> None:
        """
        Freezes/unfreezes consecutive models in the ensemble.
        ------------------------------------------------------------------------
        Args:
            i: int. Index of the model to be unfrozen
        """
        s = len(self.ensemble)
        assert i < s, "Index out of range"
        self.mask[i] = True
        for param in self.ensemble[i].parameters():
            param.requires_grad = True
        if i > 0:
            for param in self.ensemble[i - 1].parameters():
                param.requires_grad = False
