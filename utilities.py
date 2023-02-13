import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import imageio
from typing import Optional, Tuple, List, Union, Callable
from tqdm import tqdm

def save_origins_and_dirs(poses):
    '''Plot and save optical axis positions and orientations for each camera pose.
    Args:
        poses: [num_poses, 4, 4]. Camera poses.
    Returns:
        None
    '''
    # Compute optical axis orientations and positions
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses]) 
    origins = poses[:, :3, -1]
    # Plot 3D arrows representing position and orientation of the cameras
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(),
                  dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[..., 2].flatten(),
                  length=0.5,
                  normalize=True)
    plt.savefig('out/verify/poses.png')
    plt.close()

# RAY HELPERS

def get_rays(height: int,
             width: int,
             focal_length: float,
             camera_pose: torch.Tensor=None,
             local_only: bool=False
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Find origin and direction of rays through every pixel and camera origin.
    Args:
        height: Image height.
        width: Image width.
        focal_length: Focal length of the camera.
        camera_pose: [4, 4]. Camera pose matrix.
        local_only: bool. If set, return ray dirs in camera frame.
    Returns:
        origins_world: [height, width, 3]. Coordinates of rays using world coordinates.
        directions_world: [height, width, 3]. Orientations of rays in world coordinates.
    '''
    if camera_pose is None and not local_only:
        raise ValueError("Non local coordinates of camera rays require camera_pose parameter.")


    # Create grid of coordinates
    i, j = torch.meshgrid(torch.arange(width, dtype=torch.float32).to(camera_pose), 
                          torch.arange(height, dtype=torch.float32).to(camera_pose), 
                          indexing='ij')
    i, j = torch.transpose(i, -1, -2), torch.transpose(j, -1, -2)

    # Use pinhole camera model to represent grid in terms of camera coordinate frame
    focal = focal_length.item()
    directions = torch.stack([(i - width * 0.5) / focal, 
                               -(j - height * 0.5) / focal,
                               -torch.ones_like(i)], dim=-1)

    # Normalize directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    if local_only:
        return directions

    # Apply camera rotation to ray directions
    directions_world = torch.sum(directions[..., None, :] * camera_pose[:3, :3], axis=-1) 
    # Apply camera translation to ray origin
    origins_world = camera_pose[:3, -1].expand(directions_world.shape)

    return origins_world, directions_world 

def sample_stratified(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: Optional[bool] = True,
    inverse_depth: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Sample along rays using stratified sampling approach.
    Args:
        rays_origins: [height, width, 3]. Ray origins.
        rays_directions: [height, width, 3]. Ray orientations.
        near: Near bound for sampling.
        far: Far bound for sampling.
        n_samples: Number of samples.
        perturb: Use random sampling from within each bin. If disabled, use bin delimiters as sample points.
        inverse_depth:
    Returns:
        points: [height, width, n_samples, 3]. World coordinates for all samples along every ray.  
        z_vals: [height, width, n_samples]. Samples expressed as distances along every ray.
    '''
    # Grab samples for parameter t
    t_vals = torch.linspace(0., 1., n_samples, device=rays_origins.device)
    if not inverse_depth:
        # Sample linearly between 'near' and 'far' bounds.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw samples from rays according to perturb parameter
    if perturb:
        mid_values = 0.5 * (z_vals[1:] + z_vals[:-1]) # middle values between adjacent z points
        upper = torch.concat([mid_values, z_vals[-1:]], axis=-1) # append upper z point to mid values
        lower = torch.concat([z_vals[:1], mid_values], axis=-1) # prepend lower z point to mid values
        t_rand = torch.rand([n_samples], device=z_vals.device) # sample N uniform distributed points 
        z_vals  = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_origins.shape[:-1]) + [n_samples])

    # Compute world coordinates for ray samples
    points = rays_origins[..., None, :] + rays_directions[..., None, :] * z_vals[..., :, None]

    return points, z_vals

def sample_normal(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_strat: torch.Tensor,
    means: torch.Tensor,
    stdevs: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    inverse_depth: bool = False
    ) -> Tuple:
    '''Sample along rays using Gaussian sampling appoach. For each ray, two vals
    expected: mean and stdev defining a Gaussian distribution to take n_samples
    from.
    Args:
        rays_o: [..., 3]. Ray origins (world coordinates).
        rays_d: [..., 3]. Ray directions (world coordinates).
        z_strat: [..., n_samples_stratified]. Stratified samples
        means: [...]. Mean values for each ray.
        stdevs: [...]. Std. deviation values for each ray.
        near: Near bound for sampling.
        far: Far bound for sampling.
        n_samples: Number of samples.
        inverse_depth: If set. Use inverse depth instead of depth for sampling. 
    Returns:
        points: [..., n_samples, 3]. World coords for samples along every ray.  
        z_vals: [..., n_samples]. Samples as distances along every ray.
    '''
    # Grab samples for parameter t
    z_vals = torch.normal(0., 1., size=list(rays_d.shape[:-1]) + [n_samples],
                          device=means.device)
    z_vals = (z_vals * stdevs[..., None]) + means[..., None] 

    # Clip values according to near and far bounds
    z_vals = torch.clamp(z_vals, min=near)

    # Concatenate and sort stratified and normal samples
    z_vals, _ = torch.sort(torch.cat([z_vals, z_strat], -1), -1)

    # Compute world coordinates for ray samples
    points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
    
    return points, z_vals

# DEPTH UTILITIES

def dmap_from_layers(
    masks: torch.Tensor,
    ivals: torch.Tensor,
    ) -> torch.Tensor:
    '''Convert depth masks and depth intervals to depth map(s) with interval
       midpoints as the set of discrete depth values.
    Args:
        masks: [..., M, H, W]. Set(s) of M boolean  masks.
        ivals: [M, 2]. Set of M depth intervals.
    Returns:
        d_maps: [..., H, W]. Discretized depth map(s).'''

    H, W = masks.shape[-2:] # image dimensions

    # Compute pixel depth values according to masks
    d_maps = masks[..., None] * ivals[None, :, None, None, :]
    
    return d_maps


# VOLUME RENDERING UTILITIES

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    '''Mimick functionality of tf.math.cumprod(..., exclusive=True)
    Args:
        tensor: Tensor whose cumprod (cumulative product) along dim=-1 is to be
                computed.
    Returns:
        cumprod: Cumprod of tensor along dim=-1, mimiciking the functionality
                 of tf.math.cumprod(..., exclusive=True).
    '''
    # Compute exclusive cumulative product
    cumprod = torch.cumprod(tensor, -1)
    # Roll elements along last dimension by 1
    cumprod = torch.roll(tensor, 1, -1)
    # Replace first element with 1 
    cumprod[..., 0] = 1.

    return cumprod

def raw2outputs(raw: torch.Tensor,
                z_vals: torch.Tensor,
                rays_dirs: torch.Tensor,
                raw_noise_std: float = 0.0,
                white_background: bool = False) -> Tuple[torch.Tensor,
                                                         torch.Tensor,
                                                         torch.Tensor,
                                                         torch.Tensor]:
    '''Convert NeRF raw output to RGB images and other useful representations.
    Args:
       raw: [N, n_samples, 4]. Raw outputs from NeRF model. First 3 elements 
            along axis=-1 represent RGB color. 
       z_vals: [N, n_samples]: N sets of n_samples along different camera rays.
       rays_dirs: [N, 3]. N camera ray directions.
    Returns:
        rgb_map: [N, 3]. RGB rendered values for each ray.
        depth_map: [N]. Estimation for depth map.
        acc_map: [N]. Accumulation map.
        weights: [N, n_samples]. RGB weigth values. 
    '''
    # Compute difference between adjacent elements of z_vals. [N, n_samples]
    diffs = z_vals[..., 1:] - z_vals[..., :-1]
    diffs = torch.cat([diffs, 1e10 * torch.ones_like(diffs[..., :1])], dim=-1)
    
    # For non-unit directions. Convert differences to real-world distances,
    # that is, rescaling them by its direction ray norm
    diffs = diffs * torch.norm(rays_dirs[..., None, :], dim=-1)
    
    # Add noise to model's predictions for density. 
    # It can be used for regularization during training.
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict alpha compositing coefficients for each sample. [N, n_samples] 
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * diffs)
   
    # Compute weights for each sample. [N, n_samples]
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map
    rgb = torch.sigmoid(raw[..., :3])
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) 
    
    # Depth map estimation computed as predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is computed as the inverse depth
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                              depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray.
    acc_map = torch.sum(weights, dim=-1)

    # Use accumulated alpha map to composite onto a white background
    if white_background:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights

def sample_pdf(bins: torch.Tensor,
               weights: torch.Tensor,
               n_samples: int,
               perturb: bool = False) -> torch.Tensor:
    '''Apply inverse transform sampling to a weighted set of points.
    Args:
        bins:
        weights:
        n_samples:
        perturb:
    Returns:
        
    '''
    # Normalize weights to get a probability density function
    weights  = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdims=True)

    # Convert density function into cumulative function
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Take sample positions to grab from CDF. Linear when perturb is False.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1] + [n_samples])) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device = cdf.device)

    # Find indices along CDF whose values in u would be placed
    u = u.contiguous() # Return contiguous tensor
    inds = torch.searchsorted(cdf, u, right=True)

    # Clamp out of bounds indices
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from CDF and corresponding bin centers
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, 
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)

    # Convert samples to ray length
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

def sample_hierarchical(
    rays_o: torch.Tensor, 
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                          perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]

    return pts, z_vals_combined, new_z_samples


# FULL FORWARD PASS

def get_chunks(inputs: torch.Tensor,
               chunksize: int = 2**15) -> List[torch.Tensor]:
    '''Divide an input into chunks. This is done due to potential memory issues.
    The forward pass is computed in chunks, which are then aggregated across a
    single batch. The gradient propagation is done until all batch has been pro-
    cessed.
    Args:
        inputs:
        chunksize:
    Returns:
        '''
    inds = range(0, inputs.shape[0], chunksize)

    return [inputs[i:i + chunksize] for i in inds]

def prepare_chunks(points: torch.Tensor,
                   encoding_function: Callable[[torch.Tensor], torch.Tensor],
                   chunksize: int = 2**15) -> List[torch.Tensor]:
    '''Encode and chunkify points to prepare for NeRF model.
    Args:
        points: [N, n_samples, 3]. World coordinates for points.
        encoding_function: Positional encoder callable.
        chunksize: int. Chunk size.
    Returns:
        points: [M, chunksize, 3]. M chunks of embedded points.'''
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)

    return points

def prepare_viewdirs_chunks(
    points: torch.Tensor,
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15) -> List[torch.Tensor]:
    '''Encode and chunkify viewing directions to prepare for NeRF model.
    Args:
        points: [N, n_samples, 3]. World coordinates for points.
        rays_d: [N, 3]. Ray directions expressed as cartesian vectors.
        encoding_function: Positional encoder callable.
        chunksize: int. Chunk size.
    Returns:
        viewdirs: [M, chunksize, 3]. M chunks of embedded dirs.'''
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    
    return viewdirs

def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    mids: torch.Tensor = None,
    stdevs: torch.Tensor = None,
    kwargs_sample_stratified: dict = None, 
    kwargs_sample_normal: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model = None,
    viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize = 2**15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    r"""
    Compute forward pass through NeRF model(s).
    Args:
    Returns:
    """

    # Sample query points along each ray
    if kwargs_sample_stratified is not None:
        query_points, z_vals = sample_stratified(rays_o, rays_d, near, far,
                                             **kwargs_sample_stratified)

    if kwargs_sample_normal is not None: 
        if mids is not None and stdevs is not None:
            query_points, z_vals = sample_normal(rays_o, rays_d, z_vals, mids,
                                                 stdevs, near, far,
                                                 **kwargs_sample_normal)
    
    outputs = {'z_vals_stratified': z_vals}

    # Prepare batches
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                   viewdirs_encoding_fn,
                                                   chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass
    # Split encoded points into chunks, run model on all chunks, and concatenate
    # results (avoids OOM issues)
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    
    if kwargs_sample_hierarchical is not None:
        # Fine model pass
        if n_samples_hierarchical > 0:
            # Save previous outputs to return
            rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

            # Apply hierarchical sampling for fine query points
            query_points, z_vals_combined, z_hierarch = sample_hierarchical(
                    rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
                    **kwargs_sample_hierarchical)

            # Prepare inputs
            batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
            if viewdirs_encoding_fn is not None:
                batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                           viewdirs_encoding_fn,
                                                           chunksize=chunksize)
            else:
                batches_viewdirs = [None] * len(batches)

            # Forward pass new samples through fine model
            fine_model = fine_model if fine_model is not None else coarse_model
            predictions = []
            for batch, batch_viewdirs in zip(batches, batches_viewdirs):
                predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
            raw = torch.cat(predictions, dim=0)
            raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

            # Perform differentiable volume rendering on fine predictions
            rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined,
                                                               rays_d)

            # Store outputs.
            outputs['z_vals_hierarchical'] = z_hierarch
            outputs['z_vals_combined'] = z_vals_combined
            outputs['rgb_map_0'] = rgb_map_0
            outputs['depth_map_0'] = depth_map_0
            outputs['acc_map_0'] = acc_map_0 

    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights

    return outputs

# VIDEO RENDERING UTILITIES

def render_path(
    render_poses: torch.Tensor,
    near: float,
    far: float,
    hwf: torch.Tensor,
    encode: Callable[[torch.Tensor], torch.Tensor], 
    model: nn.Module,
    kwargs_sample_stratified: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model: nn.Module = None,
    encode_viewdirs: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**12
    ) -> torch.Tensor:
    r"""Render frames for the incoming camera poses.
    ----------------------------------------------------------------------------
    Args:
        render_poses: [N, 4, 4]. Camera poses to render from.
        chunksize: int. Size of smaller minibatches to avoid OOM.
        hwf: [3]. Height, width and focal length.
    Returns:
        frames: [N, H, W, 3]. Rendered RGB frames."""

    H, W, focal = hwf
    model.eval()

    frames = []
    print("Rendering frames...")
    for i, pose in enumerate(tqdm(render_poses)):
        with torch.no_grad():
            # Get rays
            rays_o, rays_d = get_rays(H, W, focal, pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])

            # Compute NeRF forward pass
            outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=n_samples_hierarchical,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=chunksize)

            rgb_predicted = outputs['rgb_map']
            rgb_predicted = rgb_predicted.reshape([H, W, 3]).detach().cpu().numpy()

        frames.append(rgb_predicted)
    frames = np.stack(frames, 0)

    return frames

to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

def render_video(
    basedir: str,
    frames: torch.Tensor):
    r"""Video rendering functionality. It takes a series of frames and joins
    them in a .mp4 file.
    ----------------------------------------------------------------------------
    Args:
        basedir: str. Base directory where to store .mp4 file.
        frames: [N, H, W, 3]. N video frames."""

    imageio.mimwrite(basedir + 'rgb.mp4', to8b(frames), fps=30, quality=8)


# TRAINING UTILITIES
class CustomScheduler:

  def __init__(
    self,
    optimizer,
    n_iters,
    etaN = 5e-6,
    lambW = 0.01,
    n_warmup = 2500,
  ):
    self.optimizer = optimizer
    self.eta0 = optimizer.param_groups[0]['lr']
    self.etaN = etaN
    self.lambW = lambW
    self.n_warmup = n_warmup
    self.iters = 0
    self.n_iters = n_iters

  def step(self):
    lr = (self.lambW + (1 - self.lambW) * np.sin(np.pi/2 * np.clip(self.iters/self.n_warmup,0,1)))
    lr *= np.exp((1 - self.iters/self.n_iters)*np.log(self.eta0) + (self.iters/self.n_iters)*np.log(self.etaN))

    self.iters += 1

    self.optimizer.param_groups[0]['lr'] = lr
