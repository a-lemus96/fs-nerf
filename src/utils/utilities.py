# stdlib modules
import os
from typing import Optional, Tuple, List, Union, Callable

# third-party modules
import numpy as np
import torch
from torch import nn
from torch import jit
from torch import Tensor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

def get_rays(
        height: int,
        width: int,
        focal_length: float,
        camera_pose: torch.Tensor=None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Find origin and direction of rays through every pixel and camera origin.
    Args:
        height: Image height.
        width: Image width.
        focal_length: Focal length of the camera.
        camera_pose: [4, 4]. Camera pose matrix.
    Returns:
        origins_world: [height, width, 3]. Coordinates of rays using world coordinates.
        directions_world: [height, width, 3]. Orientations of rays in world coordinates.
    '''
    # create grid of coordinates
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

    if camera_pose is None:
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
        t_vals: [height, width, n_samples]. Samples expressed as distances along every ray.
    '''
    # Grab samples for parameter t
    t_vals = torch.linspace(0., 1., n_samples, device=rays_origins.device)
    if not inverse_depth:
        # Sample linearly between 'near' and 'far' bounds.
        t_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw samples from rays according to perturb parameter
    if perturb:
        mids = 0.5 * (t_vals[1:] + t_vals[:-1]) # middle values between adjacent t points
        up = torch.concat([mids, t_vals[-1:]], axis=-1) # append upper t point to mid values
        low = torch.concat([t_vals[:1], mids], axis=-1) # prepend lower t point to mid values
        t_rand = torch.rand([n_samples], device=t_vals.device) # sample N uniform distributed points 
        t_vals  = low + (up - low) * t_rand
    t_vals = t_vals.expand(list(rays_origins.shape[:-1]) + [n_samples])

    # Compute world coordinates for ray samples
    points = rays_origins[..., None, :] + rays_directions[..., None, :] * t_vals[..., :, None]

    return points, t_vals

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
                white_bkgd: bool = False) -> Tuple[torch.Tensor,
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
    sigma = nn.functional.relu(raw[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma * diffs)
   
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
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, weights, sigma

@jit.script
def sample_pdf(bins: torch.Tensor,
               weights: torch.Tensor,
               n_samples: int,
               perturb: bool = False) -> torch.Tensor:
    """Apply inverse transform sampling to a weighted set of points. Function
    inspired on 'NeRF' repository from yliess86 github user.
    ----------------------------------------------------------------------------
    Args:
        bins:
        weights:
        n_samples:
        perturb:
    Returns:"""
    EPS = 1e-5 # epsilon value
    B, N = weights.size() # retrieve batch size and number of samples

    # normalize weights to get a pdf
    weights  = weights + EPS
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    # compute cumulative function
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # take sample positions to grab from cdf. Linear when perturb is False.
    if perturb:
        u = torch.rand((B, n_samples), device = cdf.device)
    else:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(B, n_samples) # [n_rays, n_samples]

    u = u.contiguous() # return contiguous tensor
    # find indices along CDF whose values in u would be placed
    idxs = torch.searchsorted(cdf, u, right=True)

    # clamp out of bounds indices
    idxs_below = torch.clamp_min(idxs - 1, 0)
    idxs_above = torch.clamp_max(idxs, N)
    # stack idxs to form a tensor of size [n_rays, 2 * n_samples]
    idxs = torch.stack([idxs_below, idxs_above], dim=-1).view(B, 2 * n_samples)

    # sample from CDF and corresponding bin centers
    cdf = torch.gather(cdf, dim=1, index=idxs).view(B, n_samples, 2)
    bins = torch.gather(bins, dim=1, index=idxs).view(B, n_samples, 2)

    denom = cdf[:, :, 1] - cdf[:, :, 0] # denominator
    denom[denom < EPS] = 1. # avoid dividing by small numbers

    # convert samples to ray length
    t = (u - cdf[..., 0]) / denom
    t = bins[..., 0] + t * (bins[..., 1] - bins[..., 0])

    return t # [n_rays, n_samples]

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
    z_vals_combined, inds = torch.sort(torch.cat([z_vals, new_z_samples], 
                                                 dim=-1), dim=-1)
    # [N_rays, N_samples + n_samples, 3]
    pts = rays_d[..., None, :] * z_vals_combined[..., :, None]  
    pts += rays_o[..., None, :]
    return pts, z_vals_combined, inds, new_z_samples


# FULL FORWARD PASS
def get_chunks(inputs: torch.Tensor,
               chunksize: int) -> List[torch.Tensor]:
    """
    Split inputs into chunks of size chunksize.
    ----------------------------------------------------------------------------
    Args:
        inputs: tensor to be chunkified
        chunksize: size of each chunk
    Returns:
        list of tensors of size chunksize
    """
    inds = range(0, inputs.shape[0], chunksize)

    return [inputs[i:i + chunksize] for i in inds]

def render_frame(
        H: int,
        W: int,
        focal: float,
        pose: torch.Tensor, 
        chunksize: int,
        near: float,
        far: float,
        pos_fn: Callable[[torch.Tensor], torch.Tensor],
        model: nn.Module,
        kwargs_sample_stratified: dict = None, 
        n_samples_hierarchical: int = 0,
        kwargs_sample_hierarchical: dict = None,
        fine_model = None,
        dir_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        white_bkgd = False) -> torch.Tensor:
    """Render an image from a given pose. Camera rays are chunkified to avoid
    memory issues.
    ----------------------------------------------------------------------------
    Args:
        H: height of the image
        W: width of the image
        focal: focal length of the camera
        pose: [4, 4] tensor with the camera pose
        chunksize: size of the chunks to be processed
        near: near clipping plane
        far: far clipping plane
        pos_fn: positional encoding function
        model: coarse model
        kwargs_sample_stratified: keyword arguments for stratified sampling
        n_samples_hierarchical: number of samples for hierarchical sampling
        kwargs_sample_hierarchical: keyword arguments for hierarchical sampling
        fine_model: fine model
        dir_fn: directional encoding function
        white_bkgd: whether to use a white background
    Returns:
        img: [H, W, 3] tensor with the rendered image
        depth: [H, W] tensor with the depth map
    """
    rays_o, rays_d = get_rays(H, W, focal, pose) # compute rays
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3) # flatten rays

    # chunkify rays to avoid memory issues
    chunked_rays_o = get_chunks(rays_o, chunksize=chunksize)
    chunked_rays_d = get_chunks(rays_d, chunksize=chunksize)

    # compute image and depth in chunks
    img = []
    depth = []
    for chunk_o, chunk_d in zip(chunked_rays_o, chunked_rays_d):
        output = nerf_forward(chunk_o, chunk_d, near, far, pos_fn,
                              model, kwargs_sample_stratified,
                              n_samples_hierarchical, kwargs_sample_hierarchical,
                              fine_model, dir_fn, white_bkgd)
        img.append(output['rgb_map'])
        depth.append(output['depth_map'])

    # aggregate chunks
    img = torch.cat(img, dim=0)
    depth = torch.cat(depth, dim=0)

    return img, depth

def nerf_forward(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    pos_fn: Callable[[torch.Tensor], torch.Tensor],
    coarse_model: nn.Module,
    kwargs_sample_stratified: dict = None, 
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model = None,
    dir_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    white_bkgd = False,
    ) -> dict:
    """
    Perform a forward pass through the NeRF model.
    ----------------------------------------------------------------------------
    Args:
        rays_o: origin of the rays
        rays_d: direction of the rays
        near: near bound
        far: far bound
        pos_fn: positional encoding function for points
        coarse_model: coarse model
        kwargs_sample_stratified: keyword arguments for stratified sampling
        n_samples_hierarchical: number of hierarchical samples
        kwargs_sample_hierarchical: keyword arguments for hierarchical sampling
        fine_model: fine model
        dir_fn: positional encoding function for viewdirs
        white_bkgd: whether to use white background
    Returns:
        dict: dictionary of outputs
    """
    # sample query points along each ray
    points, z_vals = sample_stratified(rays_o, rays_d, near, far,
                                       **kwargs_sample_stratified)
    points_shape = points.shape[:2]

    outputs = {'z_vals_stratified': z_vals}

    # prepare viewdirs
    if dir_fn is not None:
        dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        dirs = dirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        dirs = dir_fn(dirs) # positional encoding

    else:
        dirs = [None] * len(rays_d)

    points = pos_fn(points.reshape((-1, 3))) # positional encoding

    # coarse model pass
    raw = coarse_model(points, viewdirs=dirs)
    raw = raw.reshape(list(points_shape) + [raw.shape[-1]])
    # perform differentiable volume rendering
    data = raw2outputs(raw, z_vals, rays_d, white_bkgd=white_bkgd)
    rgb_map, depth_map, weights, sigma = data

    if kwargs_sample_hierarchical is not None:
        # fine model pass
        if n_samples_hierarchical > 0:
            # save previous outputs to return
            rgb_map_0, depth_map_0, sigma_0 = rgb_map, depth_map, sigma

            # apply hierarchical sampling for fine query points
            hierarch_data = sample_hierarchical(rays_o, rays_d, z_vals, weights, 
                                                n_samples_hierarchical,
                                                **kwargs_sample_hierarchical)
            points, z_vals_combined, inds, z_hierarch = hierarch_data
            points_shape = points.shape[:2]

            if dir_fn is not None:
                dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                dirs = dirs[:, None, ...].expand(points.shape).reshape((-1, 3))
                dirs = dir_fn(dirs) # positional encoding
            else:
                dirs = [None] * len(rays_d)

            points = pos_fn(points.reshape((-1, 3))) # positional encoding

            # forward pass new samples through fine model
            fine_model = fine_model if fine_model is not None else coarse_model
            #predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
            raw = fine_model(points, viewdirs=dirs)
            raw = raw.reshape(list(points_shape) + [raw.shape[-1]])
            # perform differentiable volume rendering on fine predictions
            rgb_map, depth_map, weights, sigma = raw2outputs(raw,
                                                             z_vals_combined,
                                                             rays_d,
                                                             white_bkgd=white_bkgd)

            # store outputs
            outputs['z_vals_hierarchical'] = z_hierarch
            outputs['z_vals_combined'] = z_vals_combined
            outputs['rgb_map_0'] = rgb_map_0
            outputs['depth_map_0'] = depth_map_0
            outputs['sigma_0'] = sigma_0

    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['sigma'] = sigma
    outputs['weights'] = weights

    return outputs

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
