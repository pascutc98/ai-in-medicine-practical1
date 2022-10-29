import numpy as np
import matplotlib.pyplot as plt


def plot_segmentations(im: np.ndarray, seg: np.ndarray, i: int = 65):
    _, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(np.rot90(im[..., i], k=3), cmap='gray')
    ax[0].imshow(np.rot90(seg[..., i], k=3),
                 alpha=0.5 * (np.rot90(seg[..., i] > 0, k=3)),
                 interpolation=None, cmap='jet')

    k = 1
    ax[1].imshow(np.rot90(im[i, ...], k=k), cmap='gray')
    ax[1].imshow(np.rot90(seg[i, ...], k=k),
                 alpha=0.5 * (np.rot90(seg[i, ...] > 0, k=k)),
                 interpolation=None, cmap='jet')

    ax[2].imshow(np.rot90(im[:, i, :], k=k), cmap='gray')
    ax[2].imshow(np.rot90(seg[:, i, :], k=k),
                 alpha=0.5 * (np.rot90(seg[:, i, :] > 0, k=k)),
                 interpolation=None, cmap='jet')

    plt.show()
