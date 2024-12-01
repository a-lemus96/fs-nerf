import wandb
import plotly.graph_objects as go

class PosePlotter:
    '''
    Class to plot the distribution of camera poses using plotly. It allows the
    user to create a plotly chart.
    ----------------------------------------------------------------------------
    Attributes:

    Methods:
    '''
    def __init__(self, poses, title):
        '''
        Constructor for the PosePlotter class. Initializes the plotly figure
        object.
        ------------------------------------------------------------------------
        '''
        self._poses = poses
        # Initialize the figure object with the title
        self._fig = go.Figure()
        self._fig.update_layout(title=title)

    def add_poses(self, poses):
        '''
        Method to set the camera poses in the plotly figure object.
        ------------------------------------------------------------------------
        '''
        scatter_trace = go.Scatter3d(
                x=poses[:, 0, 3],
                y=poses[:, 1, 3],
                z=poses[:, 2, 3],
        )
        self._fig.add_trace(scatter_trace)

    def clear(self):
        '''
        Method to clear the plotly figure object. It does so by initializing
        a new figure object.
        ------------------------------------------------------------------------
        '''
        self._fig = go.Figure()

    def set_axes_titles(self, x_title='X', y_title='Y', z_title='Z'):
        '''
        Method to set the axes titles of the plotly figure object.
        ------------------------------------------------------------------------
        '''
        self._fig.update_layout(
                scene=dict(
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    zaxis_title=z_title
                )
        )
