# stdlib imports
from typing import Tuple, List

# third-party imports
import wandb
import plotly.graph_objects as go
from torch import Tensor


class Camera3DPlotter:
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

    def set_poses(self, poses: Tensor, name: str = ""):
        """
        Method to set the camera poses in the plotly figure object as 3D scatter
        plot traces. Sets the default trace mode to 'markers'.
        ------------------------------------------------------------------------
        """
        scatter_trace = go.Scatter3d(
            name=name,
            x=poses[:, 0, 3],
            y=poses[:, 1, 3],
            z=poses[:, 2, 3],
            mode="markers",
        )
        self._fig.add_trace(scatter_trace)

    def configure_pose_markers(
        self, name: str = "", size: int = 7, opacity: float = 0.8, color: str = "red"
    ):
        marker_config = dict(size=size, opacity=opacity, color=color)
        self._fig.update_traces(marker=marker_config, selector=dict(name=name))

    def set_axes_titles(self, x_title, y_title, z_title):
        """
        Sets the axes' titles of the plotly figure object.
        ------------------------------------------------------------------------
        """
        self._fig.update_layout(
            scene=dict(xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title)
        )

    def set_axes_margins(self, left: int, right: int, top: int, bottom: int):
        """
        Sets the axes' margins.
        ------------------------------------------------------------------------
        """
        self._fig.update_layout(margin={"l": left, "r": right, "t": top, "b": bottom})

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
