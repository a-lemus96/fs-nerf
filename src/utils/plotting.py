# stdlib imports
from typing import Tuple, List, Union, Callable

# third-party imports
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import ndarray

def density_animate(
        curves1: np.array,
        curves2: np.array,
        ) -> animation.FuncAnimation:
    r""""""
    fig, ax = plt.subplots()

    curve1, = ax.plot(curves1[0, ..., 0], curves1[0, ..., 1], color='orange',
                     linewidth=3, label='NeRF')
    curve2, = ax.plot(curves2[0, ..., 0], curves2[0, ..., 1], color='blue',
                     linewidth=3, label='DS-NeRF')
    title = ax.set_title(f"Iteration: {0}")

    ax.set(xlim=[1.2, 7.], ylim=[0., 9.],
           xlabel=r"$t$ value", ylabel=r'Density, $\sigma$')
    ax.legend(loc="upper right")

    def update(
            frame: int
            ) -> Tuple[mpl.artist.Artist, ...]:
        r""""""
        # update the curves
        t1, sigma1 = curves1[frame, ..., 0], curves1[frame, ..., 1]
        t2, sigma2 = curves2[frame, ..., 0], curves2[frame, ..., 1] 

        # update line plots
        curve1.set_xdata(t1)
        curve1.set_ydata(sigma1)
        curve2.set_xdata(t2)
        curve2.set_ydata(sigma2)

        # update axes' title
        title.set_text(f"Iteration: {frame}k")

        return (curve1, curve2, title)

    anim = animation.FuncAnimation(fig=fig, func=update,
                                   frames=curves1.shape[0], interval=500)

    return anim

def apply_colormap(
        data: ndarray,
        cmap: str = 'viridis',
        norm: Normalize = None
) -> ndarray:
    """Apply a colormap to the data.
    ----------------------------------------------------------------------------
    Args:
        data (ndarray): The data to apply the colormap to
        cmap (str): The name of the colormap to use
        norm (Normalize): The normalization to use
    Returns:
        ndarray: The data with the colormap applied
    ----------------------------------------------------------------------------
    """
    # get the colormap
    cmap = plt.get_cmap(cmap)
    # get the normalization
    if norm is None:
        norm = Normalize(vmin=data.min(), vmax=data.max())
    # apply the colormap
    return cmap(norm(data))
