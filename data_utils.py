import os
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from skimage.transform import resize
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

DATA_DIR = './data/brain_age'


def get_image_dataloaders(
    img_size: int,
    batch_size: int,
    num_workers: int = 0
):
    print('Loading data. This might take a while...')
    train_ds = BrainAgeImageDataset('train', img_size)
    val_ds = BrainAgeImageDataset('val', img_size)
    test_ds = BrainAgeImageDataset('test', img_size)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return {'train': train_dl, 'val': val_dl, 'test': test_dl}


class BrainAgeImageDataset(Dataset):
    def __init__(self, mode: str, img_size: int):
        assert mode in ['train', 'val', 'test']
        print(f'Loading {mode} data...')

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train.csv'))
        elif mode == 'val':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_segmentation_train.csv'))
        if mode == 'test':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_test.csv'))

        self.images = prefetch_samples(self.df['subject_id'].tolist(), img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a brain MR-image and it's corresponding age

        :param idx: Sample index of the dataset
        :return img: Loaded brain MRI. Shape (H, W, D)
        """
        img = self.images[idx]
        age = self.df.iloc[[idx]]['age'].item()
        return img, age


def normalize(img: np.ndarray, mask: np.ndarray):
    """Normalize a brain MR-image to have zero mean and unit variance.
    For getting the normalization parameters, only the brain pixels should be
    used. All background pixels should be 0 in the end.

    :param img: Brain MR-image. Shape (H, W, D)
    :param mask: Mask to indicate which voxels correspond to the brain (1s) and
                 which are background (0s). Shape (H, W, D)
    :return normalized_img: Brain MR-image normalized to zero mean and unit
                            variance. Shape (H, W, D)
    """
    # ------------------------- ADD YOUR CODE HERE ----------------------------
    #print(img.shape, mask.shape)
    mask_bool = np.array(mask, dtype=bool)
    normalized_img = img - np.mean(img, where=mask_bool) / np.std(img, where=mask_bool)
    normalized_img = normalized_img*mask
    #print(normalized_img.shape)
    # --------------------------------- END -----------------------------------
    return normalized_img


def load_nii(path: str, dtype: str = 'float32') -> np.ndarray:
    """Load an MRI scan from disk and convert it to a given datatype

    :param path: Path to file
    :param dtype: Target dtype
    :return img: Loaded image. Shape (H, W, D)
    """
    return nib.load(path).get_fdata().astype(np.dtype(dtype))


def preprocess(img: np.ndarray, mask: np.ndarray, img_size: int) -> np.ndarray:
    """Preprocess an MRI.

    :param img: MR-image. Shape (H, W, D)
    :param mask: Brain mask. Shape (H, W, D)
    :param img_size: Target size, a scalar.
    :return preprocessed_img: The preprocessed image.
                              Shape (img_size, img_size, img_size)
    """
    # Feel free to add more pre-processing here.
    # ------------------------- ADD YOUR CODE HERE ----------------------------
    preprocessed_img = normalize(img, mask)
    preprocessed_img = resize(preprocessed_img, [img_size] * 3)
    # --------------------------------- END -----------------------------------
    return preprocessed_img


def load_and_preprocess(ID: str, img_size: int) -> np.ndarray:
    """Load an MRI from disk and preprocess it"""
    img = load_nii(os.path.join(DATA_DIR, f"images/sub-{ID}_T1w_unbiased.nii.gz"))
    mask = load_nii(
        os.path.join(DATA_DIR, f"masks/sub-{ID}_T1w_brain_mask.nii.gz"),
        dtype='int'
    )
    return preprocess(img, mask, img_size)


def prefetch_samples(
    IDs: List[str],
    img_size: int,
    num_processes: int = 4
) -> np.ndarray:
    load_fn = partial(load_and_preprocess, img_size=img_size)
    # res = [load_fn(ID) for ID in IDs]
    # To speed up loading, comment the line above and uncomment the two below.
    with Pool(num_processes) as p:
      res = p.map(load_fn, IDs)
    return np.array(res)[:, None]


def load_segmentations(paths: str):
    """Load all segmentations and associated subject_ids"""
    filenames, segmentations = [], []
    for im in tqdm(paths):
        id = im.split('_brain_')[0].split('/')[-1].split('-')[1].split('_')[0]
        segmentations.append(load_nii(im))
        filenames.append(id)
    return filenames, np.array(segmentations)