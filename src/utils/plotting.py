# stdlib imports
from typing import Tuple, List

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
import wandb
import plotly.graph_objects as go
from torch import Tensor


class CameraPosesPlot:
    """
    Class to  hold the plot of the distribution of camera poses using plotly. It
    allows the user to define a plotly chart for camera poses. Plot can be
    logged to weights and biases.
    ----------------------------------------------------------------------------
    Attributes:

    Methods:
    """

    def __init__(self, title: str = "Camera Distribution"):
        """
        Constructor for the CameraPosePlotter class. Initializes the
        plotly figure object.
        ------------------------------------------------------------------------
        """
        # Initialize the figure object and set its title
        self._fig = go.Figure()
        self._fig.update_layout(title=title)
        self._fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
        )

    def clear(self):
        """
        Clears the plotly figure object. It does so by initializing
        a new figure object.
        ------------------------------------------------------------------------
        """
        self._fig = go.Figure()

    def set_poses(self, poses: Tensor):
        """
        Method to set the camera poses in the plotly figure object as 3D scatter
        plot traces. Sets the default trace mode to 'markers'.
        ------------------------------------------------------------------------
        """
        scatter_trace = go.Scatter3d(
            name="poses",
            x=poses[:, 0, 3],
            y=poses[:, 1, 3],
            z=poses[:, 2, 3],
            mode="markers",
        )
        self._fig.add_trace(scatter_trace)

    def configure_pose_markers(
        self, size: int = 7, opacity: float = 0.8, color: str = "red"
    ):
        marker_config = dict(size=size, opacity=opacity, color=color)
        self._fig.update_traces(
            marker=marker_config, selector=dict(name="poses")
        )

    def set_axes_titles(self, x_title, y_title, z_title):
        """
        Sets the axes' titles of the plotly figure object.
        ------------------------------------------------------------------------
        """
        self._fig.update_layout(
            scene=dict(
                xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title
            )
        )

    def set_axes_margins(self, left: int, right: int, top: int, bottom: int):
        """
        Sets the axes' margins.
        ------------------------------------------------------------------------
        """
        self._fig.update_layout(
            margin={"l": left, "r": right, "t": top, "b": bottom}
        )

    def set_axes_ranges(self, xrange: List, yrange: List, zrange: List):
        """
        Sets the axes' value ranges.
        ------------------------------------------------------------------------
        """
        self._fig.update_layout(
            scene={
                "xaxis": {"range": xrange},
                "yaxis": {"range": yrange},
                "zaxis": {"range": zrange},
            }
        )

    def upload_plot(self):
        """
        Logs the 3D plotly figure object to the weights and biases platform.
        ------------------------------------------------------------------------
        """
        wandb.log(
            {
                "Camera Poses": self._fig,
            }
        )


def density_animate(
    curves1: np.array,
    curves2: np.array,
) -> animation.FuncAnimation:
    r""""""
    fig, ax = plt.subplots()

    (curve1,) = ax.plot(
        curves1[0, ..., 0],
        curves1[0, ..., 1],
        color="orange",
        linewidth=3,
        label="NeRF",
    )
    (curve2,) = ax.plot(
        curves2[0, ..., 0],
        curves2[0, ..., 1],
        color="blue",
        linewidth=3,
        label="DS-NeRF",
    )
    title = ax.set_title(f"Iteration: {0}")

    ax.set(
        xlim=[1.2, 7.0],
        ylim=[0.0, 9.0],
        xlabel=r"$t$ value",
        ylabel=r"Density, $\sigma$",
    )
    ax.legend(loc="upper right")

    def update(frame: int) -> Tuple[mpl.artist.Artist, ...]:
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

    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=curves1.shape[0], interval=500
    )

    return anim


def apply_colormap(
    data: ndarray, cmap: str = "plasma", norm: Normalize = None
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
        norm = Normalize(vmin=0.0, vmax=6.0)
    # apply the colormap
    return cmap(norm(data))
