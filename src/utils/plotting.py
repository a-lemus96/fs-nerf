# stdlib imports
from typing import Tuple, List, Union, Callable

# third-party imports
import matplotlib as mpl
from matplotlib.axes import Axes
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

def maps(
    rgb: ndarray,
    depth: ndarray,
    rgb_gt: ndarray,
    depth_gt: ndarray
    ) -> Tuple[Figure, Axes]:
    """
    Plot RGB and depth maps for comparison against ground truth maps.
    ----------------------------------------------------------------------------
    Args:
        rgb (ndarray): (H, W, 3). Predicted RGB map
        depth (ndarray): (H, W). Predicted depth map
        rgb_gt (ndarray): (H, W, 3). Ground truth RGB map
        depth_gt (ndarray): (H, W). Ground truth depth map
    Returns:
        fig (Figure): Figure object containing the plots
        axs (Axes): Axes object containing the plots
    ----------------------------------------------------------------------------
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("Predicted RGB")
    axs[0, 1].imshow(depth)
    axs[0, 1].set_title("Predicted Depth")
    axs[1, 0].imshow(rgb_gt)
    axs[1, 0].set_title("Ground Truth RGB")
    axs[1, 1].imshow(depth_gt)
    axs[1, 1].set_title("Ground Truth Depth")
    plt.tight_layout()

    return fig, axs
