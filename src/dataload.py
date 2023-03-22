# standard library modules
import json
import os
from typing import Tuple, List, Union, Callable

# third-party modules
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np

# custom modules
from utilities import *


# DATA READER CLASS DEFINITION

class DataReader:
    """Class definition for reading several NeRF dataset formats.
    ----------------------------------------------------------------------------
    """ 
    def __read_custom(basedir: str) -> Tuple:
        """Reads custom blender dataset.
        ------------------------------------------------------------------------
        Args:
            basedir: str. Scene path that contains training files.
        Returns:
            imgs: [N, H, W, 3]. N HxW RGB images
            poses: [N, 4, 4]. N 4x4 camera poses
            hwf: [3, ]. Array containing height, width and focal values
            d_maps: (N, H, W)-shape. Contains depth maps
            d_backs: (N, W, W)-shape. Contains boolean masks for background""" 

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

    # class variable holding possible set of callable objects 
    __read_fns = {'custom': __read_custom}


    def __init__(self, dataset: str):
        """Constructor method.
        ------------------------------------------------------------------------
        Args:
            dataset: string type indicating the type of dataset
        Returns:
            DataReader object"""
        self.dataset = dataset

    # Public methods

    def get_data(self, scene: str) -> Tuple:
        # select reading method
        read_fn = self.__class__.__read_fns[self.dataset]
        basedir = os.path.join('..', 'data', self.dataset, scene)

        return read_fn(basedir)
