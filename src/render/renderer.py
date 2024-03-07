from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator

from typing import Dict, Tuple
import torch
from torch import nn
from torch import Tensor

import utils.utilities as utils

class Renderer:
    def __init__(
            self, 
            near: float, 
            far: float,
            chunksize: int,
            white_bkgd: bool = True,
            train: bool = False,
            **kwargs
    ):
        """
        Initialize a renderer and associate it with an occupancy grid estimator.
        ------------------------------------------------------------------------
        Args:
            - near: float. Near bound.
            - far: float. Far bound.
            - chunksize: int. Chunk size for rendering.
            - white_bkgd: bool. Whether to use a white background.
            - train: bool. Whether to use the renderer in training mode.
            - **kwargs: Dict. Additional arguments for the estimator.
        ------------------------------------------------------------------------
        """
        # unpack kwargs
        self.tn = near
        self.tf = far
        self.chunksize = chunksize
        # compute background color
        light = torch.ones(3, dtype=torch.float32)
        dark = torch.zeros(3, dtype=torch.float32)
        self.bkgd = torch.where(white_bkgd, light, dark)
        # set operational mode
        self.train = train
        # unpack kwargs
        self.render_step_size = kwargs['render_step_size']
        aabb = kwargs['aabb']
        resolution = kwargs['resolution']
        grid_nlevels = kwargs['grid_nlevels']
        # init estimator
        self.estimator = OccGridEstimator(aabb, resolution, grid_nlevels)

        self.k = 0 # step counter

    def render_rays(
            self,
            rays_o: Tensor,
            rays_d: Tensor,
            model: nn.Module,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Render a batch of rays using a radiance field.
        ------------------------------------------------------------------------
        Args:
            - rays_o: (..., 3). Ray origins.
            - rays_d: (..., 3). Ray directions.
            - model: Radiance field.
        Returns:
            - rgb: (..., 3). Rendered colors.
            - opacity: (..., 1). Opacity.
            - depth: (..., 1). Depth along rays.
            - extras: Dict[str, Tensor]. Extra intermediate results.
        ------------------------------------------------------------------------
        """
        # query local density
        def _sigma_fn(t_starts, t_ends, ray_idxs):
            to = rays_o[ray_idxs]
            td = rays_d[ray_idxs]
            x = to + td * (t_starts + t_ends)[:, None] / 2.0
            sigmas = model(x)
            return sigmas.squeeze(-1)

        # perform grid sampling
        ray_idxs, t_starts, t_ends = self.estimator.sampling(
                rays_o, rays_d,
                sigma_fn=_sigma_fn,
                render_step_size=self.render_step_size,
                stratified=self.train,
                near_plane=self.near,
                far_plane=self.far
        )

        # query local rgb and density
        def _rgb_sigma_fn(t_starts, t_ends, ray_idxs):
                to = rays_o[ray_idxs]
                td = rays_d[ray_idxs]
                x = to + td * (t_starts + t_ends)[:, None] / 2.0
                out = model(x, td)
                rgbs = out[..., :3]
                sigmas = out[..., -1]

                return rgbs, sigmas.squeeze(-1)

        self.bkgd = self.bkgd.to(device)

        # perform volume rendering
        try:
            data = rendering(t_starts, t_ends, ray_idxs, n_rays=len(rays),
                    rgb_sigma_fn=_rgb_sigma_fn, render_bkgd=self.bkgd)
        except AssertionError as assert_err:
            print(assert_err)
            data = (self.bkgd.expand(rays_o.shape), torch.zeros_like(rays_o),
                    torch.zeros_like(rays_o[:-1]).unsqueeze(-1)), {})
        
        return data
        
    def render_poses(
            self,
            intrinsics: Tuple[int, int, float],
            poses: Tensor,
            model: nn.Module,
            ndc: bool = False,
            device: torch.device = torch.device('cpu')
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Render a batch of poses using a radiance field.
        ------------------------------------------------------------------------
        Args:
            - intrinsics: (W, H, focal).
            - poses: (..., 4, 4). Poses.
            - model: Radiance field.
            - ndc: Whether to use normalized device coordinates.
            - device: Device to run on.
        Returns:
            - rgb_maps: (..., H, W, 3). Rendered images.
            - depth_maps: (..., H, W). Depth maps.
        ------------------------------------------------------------------------
        """
        H, W, focal = intrinsics
        rays_o, rays_d = utils.get_rays(H, W, focal, poses, device)
        if ndc:
            # convert rays to normalized device coordinates
            rays_o, rays_d = utils.to_ndc(rays_o, rays_d, 1., [H, W, focal])
        # flatten rays
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # chunkify rays
        chunks_o = utils.get_chunks(rays_o, chunksize=self.chunksize)
        chunks_d = utils.get_chunks(rays_d, chunksize=self.chunksize)

        # render chunks
        rgb_maps = []
        depth_maps = []

        for rays_o, rays_d in zip(chunks_o, chunks_d):
            (rgb, _, depth, _), _ = self.render_rays(rays_o, rays_d, model)
            rgb_maps.append(rgb)
            depth_maps.append(depth)

        # unchunkify results
        rgb_maps = torch.cat(rgb_maps, dim=0)
        depth_maps = torch.cat(depth_maps, dim=0)

        # reshape results
        rgb_maps = rgb_maps.reshape(*poses.shape[:-2], H, W, 3)
        depth_maps = depth_maps.reshape(*poses.shape[:-2], H, W)

        return rgb_maps, depth_maps

    def step(self, model):
        self.k += 1 # increment step counter
        # define occupancy evaluation function
        def occ_eval_fn(x):
            density = model(x)
            return density * render_step_size
        # update estimator
        self.estimator.update_every_n_steps(step=self.k, 
                occ_eval_fn=_occ_eval_fn, occ_thre=self.occ_thre)

    def eval(self):
        self.train = False
        self.estimator.eval()

    def train(self):
        self.train = True
        self.estimator.train()

    def set_chunksize(self, value):
        self.chunksize = value
