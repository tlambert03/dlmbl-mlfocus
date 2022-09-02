from pathlib import Path
from turtle import forward
from scipy.fft import ifftshift
from torch.fft import ifftshift, fftshift, fftn
from tifffile import imread
from torch.utils.data import Dataset
from torch import nn
from scipy.signal import windows
import numpy as np
import torch


from mlfocus.util import deskew, z_offset_from_name, window3d, radial_profile_2d

angle = 31
dz = 0.2
dx = 0.105
deskewFactor = np.cos(np.deg2rad(angle)) * dz / dx
MAT = np.eye(4)
MAT[2, 0] = -deskewFactor



class MyData(Dataset):
    def __init__(self, root: str, patch_size: int = 128, transform=None):
        self.root = Path(root)
        self.files = sorted(self.root.rglob("*patches/*.npz"))
        self.half_patch = patch_size // 2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        label = z_offset_from_name(file.name)
        image = torch.from_numpy(np.load(file)["arr_0"])
        image = self._prep_fft(image)

        # (1, 2, 128, 128), float
        return image, np.float32(label)

    def _prep_fft(self, img):
        # take windowed fft
        img = img * window3d(img.shape, windows.cosine)
        img = fftshift(fftn(ifftshift(img)))

        # take ZR radial average
        qpatch = self.half_patch // 2
        slc = slice(self.half_patch - qpatch, self.half_patch + qpatch)
        comp_abs = radial_profile_2d(np.abs(img))[slc, :qpatch]
        comp_angle = radial_profile_2d(np.angle(img))[slc, :qpatch]
        
        comp_abs = comp_abs / np.percentile(comp_abs, 99)
        
        # return two channel real/imag stack
        return torch.from_numpy(np.stack([comp_abs, comp_angle]).astype(np.float32))


