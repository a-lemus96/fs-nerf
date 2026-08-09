import numpy as np

from ..datasets import llff

LLFF_HOLD = 8  # every LLFF_HOLD-th view is held out for testing


class Splitter:
    """
    Create train/test splits for a NeRF dataset following the FreeNeRF protocol.

    The split is deterministic and reproduces the one used by FreeNeRF/RegNeRF,
    so metrics are directly comparable to the numbers published for those methods:

    - Test split: every LLFF_HOLD-th view (indices 0, 8, 16, ...). It is evaluated
      in full and is currently reused as the validation split by the caller.
    - Train pool: the remaining views.
    - Few-shot train split: `n_views` evenly spaced picks over the train pool. LLFF
      captures sweep the scene sequentially, so evenly spaced indices translate to
      evenly spaced camera poses. Passing a negative `n_views` keeps the whole pool.

    Only the LLFF dataset layout is supported. Usage::

        splitter = Splitter("fern")
        splitter.split(n_views=3)
        train_dataset, test_dataset = splitter.get_datasets(ndc=True)
    """

    def __init__(self, scene: str):
        """
        Loads the scene and prepares its poses, bounds and intrinsics.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene folder name under ../datasets/llff/
        """
        self.scene = scene

        (
            self.img_paths,
            self.poses,
            self.hwf,
            self.min_bound,
            self.max_bound,
            self.path_poses,
        ) = llff.LLFFDataset.load_scene(scene)

    def split(self, n_views: int = -1):
        """
        Computes the train and test view indices for the scene.
        ------------------------------------------------------------------------
        Args:
            n_views (int): number of training views to keep. A negative value
                           keeps every view left in the train pool
        Returns:
            None. Sets self.train_ids and self.test_ids
        """
        all_indices = np.arange(len(self.poses))
        self.test_ids = all_indices[all_indices % LLFF_HOLD == 0]
        pool = all_indices[all_indices % LLFF_HOLD != 0]

        if n_views < 0:
            self.train_ids = pool
        else:
            assert 0 < n_views <= len(pool), (
                f"ValueError, the number of training views must be in [1, {len(pool)}], "
                f"got {n_views}."
            )
            idx_sub = [round(i) for i in np.linspace(0, len(pool) - 1, n_views)]
            self.train_ids = pool[idx_sub]

    def get_datasets(self, train_img_mode: bool = False, **kwargs):
        """
        Builds the train and test datasets from a prior call to split().
        ------------------------------------------------------------------------
        Args:
            train_img_mode (bool): if True, the training dataset yields whole
                                   images instead of rays. The test dataset is
                                   always built in image mode
            **kwargs: forwarded dataset options. 'ndc' (bool, default False)
                      maps rays to normalized device coordinates
        Returns:
            train_dataset (LLFFDataset): dataset over the training views
            test_dataset (LLFFDataset): dataset over the held-out views
        """
        assert (
            self.train_ids is not None
        ), "Split the source data before building the datasets."
        # Get image files
        test_poses = self.poses[self.test_ids]
        test_img_paths = self.img_paths[self.test_ids]
        test_imgs = llff.LLFFDataset.load_img_files(test_img_paths)

        train_poses = self.poses[self.train_ids]
        train_img_paths = self.img_paths[self.train_ids]
        train_imgs = llff.LLFFDataset.load_img_files(train_img_paths)

        # Instantiate datasets
        ndc = kwargs.get("ndc", False)
        test_dataset = llff.LLFFDataset(
            test_imgs,
            test_poses,
            self.min_bound,
            self.max_bound,
            self.hwf,
            True,
            ndc,
        )

        train_dataset = llff.LLFFDataset(
            train_imgs,
            train_poses,
            self.min_bound,
            self.max_bound,
            self.hwf,
            train_img_mode,
            ndc,
        )

        return train_dataset, test_dataset
