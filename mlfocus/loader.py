from pathlib import Path

import numpy as np
from tifffile.tifffile import re
import torch
from scipy.fft import ifftshift
from scipy.signal import windows
from tifffile import imread
from torch.fft import fftn, fftshift, ifftshift
from torch.utils.data import Dataset, RandomSampler
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from mlfocus.util import deskew, radial_profile_2d, window3d, z_offset_from_name


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


angle = 31
dz = 0.2
dx = 0.105
deskewFactor = np.cos(np.deg2rad(angle)) * dz / dx
MAT = np.eye(4)
MAT[2, 0] = -deskewFactor


# class MyData(Dataset):
#     def __init__(self, root: str, patch_size: int = 128, transform=None):
#         self.root = Path(root)
#         self.files = sorted(self.root.rglob("*patches/*.npz"))
#         self.half_patch = patch_size // 2

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         file = self.files[index]
#         # print("get file", file.name)
#         label = z_offset_from_name(file.name)
#         image = torch.from_numpy(np.load(file)["arr_0"])
#         image = self._prep_fft(image)

#         # (2, 64, 32), float32
#         return image, torch.Tensor([label])

#     def _prep_fft(self, img):
#         # take windowed fft
#         img = img * window3d(img.shape, windows.cosine)
#         img = fftshift(fftn(ifftshift(img)))

#         # take ZR radial average
#         qpatch = self.half_patch // 2
#         slc = slice(self.half_patch - qpatch, self.half_patch + qpatch)
#         comp_abs = radial_profile_2d(np.abs(img))[slc, :qpatch]
#         comp_angle = radial_profile_2d(np.angle(img))[slc, :qpatch]

#         comp_abs = comp_abs / np.percentile(comp_abs, 99)

#         # return two channel real/imag stack
#         return torch.from_numpy(np.stack([comp_abs, comp_angle]).astype(np.float32))


def _get_files(root, glob, exclude=(), include=()):
    root = Path(root)
    files = list(root.rglob(glob))
    if exclude:
        print("excluding", exclude)
        files = [f for f in files if all(e not in str(f) for e in exclude)]
    if include:
        files = [f for f in files if all(e in str(f) for e in include)]
    return sorted(files)


class MyFFTData(Dataset):
    # include or exclude can be "bead", "diffuse", "mt"
    def __init__(self, root: str, exclude=(), include=()):
        self.root = Path(root)
        self.files = _get_files(root, "*fft/*.npy", exclude, include)
        # load them all and normalize here
        self.images = [np.load(file)[:, 16:48, :32] for file in tqdm(self.files, "loading fft")]

        self.transforms = Compose(
            [
                ToTensor(),
                AddGaussianNoise(0, 0.05),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        # print("get file", file.name)
        label = z_offset_from_name(file.name)
        data = self.images[index]
        # data = self.transforms(data)
        return data, torch.Tensor([label])

    def get_class(self, index) -> str:
        file = self.files[index]
        if "diffuse" in file.name:
            return "diffuse"
        elif "mt00" in file.name:
            return "mt"
        return "bead"


from tqdm import tqdm


class MIPData(Dataset):
    def __init__(self, root: str, patch_size: int = 128, exclude=(), include=()):
        self.root = Path(root)

        self.files = _get_files(root, "*mips/*.tif", exclude, include)
        # load them all and normalize here
        self.images = [
            normalize(imread(file)).astype(np.float32)
            for file in tqdm(self.files, "loading mips")
        ]
        self.transforms = Compose(
            [
                ToTensor(),
                RandomCrop(patch_size),
                AddGaussianNoise(0, 0.05),
                RandomHorizontalFlip(),
                # intensity scale and shift
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = z_offset_from_name(str(self.files[index]))
        data = self.images[index]
        image = self.transforms(data)
        return image, torch.Tensor([label])

    def get_class(self, index) -> str:
        file = self.files[index]
        if "diffuse" in str(file):
            return "diffuse"
        elif "mt00" in str(file):
            return "mt"
        return "bead"


def normalize(img: np.ndarray) -> np.ndarray:
    mi, ma = np.percentile(img, [1, 99])
    return (img - mi) / ((ma - mi) + np.finfo(np.float32).eps)


def get_loaders(
    root,
    test_split: float = 0.2,
    batch_size: int = 16,
    patches_per_image=8,
    loader_class=MyFFTData,
):
    """Loads the data"""
    from torch.utils.data import DataLoader, random_split

    focus_dataset = loader_class(root)

    # Create indices for the split
    dataset_size = len(focus_dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = random_split(focus_dataset, [train_size, test_size])

    sampler = None
    if loader_class != MyFFTData:
        # MOAR PATCHEZ
        sampler = RandomSampler(
            train_dataset, replacement=True, num_samples=train_size * patches_per_image
        )

    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=loader_class == MyFFTData,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
