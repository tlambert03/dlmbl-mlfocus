from pathlib import Path
from turtle import forward

from tifffile import imread
from torch.utils.data import Dataset
from torch import nn
from scipy.signal import windows
import numpy as np


from mlfocus.util import deskew, z_offset_from_name, window3d

angle = 31
dz = 0.2
dx = 0.105
deskewFactor = np.cos(np.deg2rad(angle)) * dz / dx
MAT = np.eye(4)
MAT[2, 0] = -deskewFactor




class MyLoader(Dataset):
    def __init__(self, root: str, patch_size: int = 128, transform=None):
        self.root = Path(root)
        self.files = sorted(self.root.rglob("*.tif"))
        self.half_patch = patch_size // 2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        label = z_offset_from_name(self.files[index].name)
        image = imread(self.files[index])
        windowed = image * window3d(image.shape, windows.cosine)
        deskewed = deskew(windowed)

        coords = self.random_coords(image.shape, 10)
        # shift coords to deskewed space
        coords = (np.linalg.inv(MAT[:3, :3]) @ coords).T.astype(int)
        for c in coords:
            patch = tuple(slice(x - self.half_patch, x + self.half_patch) for x in c)
            print(deskewed[patch].min())

        return (1, Z, Y, X), label

    def random_coords(self, shape, n_points: int = 1):
        nz, ny, nx = shape
        pad = 48
        a = self.half_patch + pad
        zs = np.random.randint(a, nz - a, size=n_points)
        ys = np.random.randint(a, ny - a, size=n_points)
        zs = np.random.randint(a, nx - a, size=n_points)
        return np.stack([zs, ys, zs])
