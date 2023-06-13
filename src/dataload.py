# stdlib imports
import glob
import json
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from typing import Tuple, List, Union, Callable

# third-party imports
from matplotlib import pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tv
import tqdm
import imageio
import numpy as np

# custom modules
from utilities import *


# DATA READER CLASS DEFINITION

class DataReader:
    """Class definition for reading several NeRF dataset formats.
    ----------------------------------------------------------------------------
    """ 
    def __read_custom(basedir: str, factor: int, **kwargs) -> Tuple:
        """Reads custom blender dataset.
        ------------------------------------------------------------------------
        Args:
            basedir: basepath
            factor: for reducing image resolution
            **kwargs: keyword args
        Returns:
            imgs: [N, H, W, 3]. N HxW RGB images
            poses: [N, 4, 4]. N 4x4 camera poses
            hwf: [3, ]. Array containing height, width and focal values
            d_maps: (N, H, W)-shape. Contains depth maps
            d_backs: (N, W, W)-shape. Contains boolean masks for background""" 
        basedir = os.path.join(basedir, kwargs['dataset'], kwargs['scene'])

        # Load JSON file
        with open(os.path.join(basedir, 'transforms_train.json'), 'r') as fp:
            meta = json.load(fp)

        # Load frames and poses
        imgs = []
        poses = []

        for frame in meta['frames']:
            fname = os.path.join(basedir, frame['image_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        # Convert to numpy arrays
        imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)
        poses = np.stack(poses, axis=0).astype(np.float32)
        
        # Compute image height, width and camera's focal length
        H, W = imgs.shape[1:3]
        fov_x = meta['camera_angle_x'] # Field of view along camera x-axis
        focal = 0.5 * W / np.tan(0.5 * fov_x)
        hwf = np.array([H, W, np.array(focal)])
      
        imgs = torch.Tensor(imgs[..., :-1]) # discard alpha channel
        poses = torch.Tensor(poses)
        hwf = torch.Tensor(hwf)

        # Load depth information
        fname_d = os.path.join(basedir, meta['depth_path'] + '.npz') 
        d_data = np.load(fname_d)
        d_maps = d_data['depths']
        d_masks = d_data['masks']
        d_backs = torch.Tensor(d_masks[:, -1, ...]) # backgrounds
        d_maps = torch.Tensor(d_maps)

        return imgs, poses, hwf, d_maps, d_backs

    def __read_rtmv(basedir: str, factor: int=None, **kwargs) -> Tuple:
        """Reads RTMV dataset.
        ------------------------------------------------------------------------
        Args:
            basedir: base path
            factor: for reducing image resolution
            **kwargs: keyword args
        Returns:
            imgs: (N, H, W, 3)-shape. N HxW RGB images
            poses: (N, 4, 4)-shape. N 4x4 camera poses
            hwf: (3,)-shape. Array containing height, width and focal values
            d_maps: (N, H, W)-shape. Contains depth maps
            d_backs: (N, W, W)-shape. Contains boolean masks for background""" 

        # build basedir
        basedir = os.path.join(basedir, kwargs['dataset'],
                               kwargs['subset'], kwargs['scene'])
        # retrieve filenames
        files = os.listdir(basedir)
        fnames = [f.split('.')[0] for f in files if f.endswith('.json')]

        imgs, poses, d_maps, d_masks = [], [], [], []
        # iterate through all filenames
        print("Reading files...")
        for fname in tqdm.tqdm(fnames):
            # load JSON file
            with open(os.path.join(basedir, fname + '.json'), 'r') as fp:
                meta = json.load(fp)['camera_data']
                pose = meta['cam2world']
                poses.append(pose)
                focal = meta['intrinsics']['fx']

            #load RGB file
            img = cv2.imread(os.path.join(basedir, fname + '.exr'),
                             cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # load depth map
            d_map = cv2.imread(os.path.join(basedir, fname + '.depth.exr'),
                             cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # apply downsampling
            if factor is not None:
                new_size = (d_map.shape[1] // factor, d_map.shape[0] // factor)
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                d_map = cv2.resize(d_map, new_size, interpolation=cv2.INTER_AREA)

            imgs.append(img[...,::-1]) # change BGR to RGB
            d_map = d_map[..., 0]
            d_maps.append(d_map)
        
        poses = torch.Tensor(np.array(poses))
        imgs = torch.Tensor(np.array(imgs))
        d_maps = torch.Tensor(np.array(d_maps))
        eps = torch.min(d_maps)
        d_backs = torch.eq(d_maps, eps)
        H, W = imgs.shape[1:3]
        hwf = torch.Tensor(np.array([H, W, np.array(focal)]))

        # apply downsampling if applicable
        if factor is not None:
            # apply factor to camera intrinsics
            new_W, new_H = new_size
            new_focal = hwf[2] / float(factor)
            hwf = torch.Tensor((new_H, new_W, new_focal))
        
        return imgs, poses, hwf, d_maps, d_backs

    def __downsample(
            imgs: torch.Tensor, 
            hwf: torch.Tensor, 
            factor: int
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample images and apply resize factor to camera intrinsics. It
        may be used for upsampling by using 0 < 'factor' < 1.
        ------------------------------------------------------------------------
        Args:
            imgs: (N, H, W, 3)-shape. Images
            hwf: (3,)-shape. Camera intrinsics
            factor: Downsample factor
        Returns:
            new_imgs: (N, H, W, 3)-shape. Downsampled images
            new_hwf: (3,)-shape. Updated camera intrinsics"""

        return new_imgs, new_hwf


    # class variable holding possible set of callable objects 
    __read_fns = {'custom': __read_custom,
                  'rtmv': __read_rtmv}

    def __init__(self, dataset: str, subset: str, scene: str, factor: int):
        """Constructor method.
        ------------------------------------------------------------------------
        Args:
            dataset: string type indicating the type of dataset
            subset: dataset subcategory
            scene: name of the scene
            factor: for reducing image resolution
        Returns:
            DataReader object"""
        self.dataset = dataset
        self.subset = subset
        self.scene = scene
        self.factor = factor

    # Public methods

    def get_data(self) -> Tuple:
        # select reading method
        read_fn = self.__class__.__read_fns[self.dataset]
        basedir = os.path.join('..', 'data')

        return read_fn(basedir, self.factor, dataset=self.dataset, 
                       subset=self.subset, scene=self.scene)
