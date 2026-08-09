# stdlib modules
import math
from typing import List

# third-party modules
import torch
from torch import nn


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

class Sinerf(nn.Module):
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
        super(Sinerf, self).__init__()
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
