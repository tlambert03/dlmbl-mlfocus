from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from napari_ndtiffs.reader import get_deskew_func
from scipy import signal
from scipy.fft import fftn, fftshift, ifftshift
from tifffile import imread, imwrite
import seaborn as sns

if TYPE_CHECKING:
    from torch.utils.data import Dataset


mag = 61.90476
DX = 6.5 / mag  # pixel size
DZ = 0.2  # I checked... and every file in this dataset used this stage step
ANGLE = 31
OFFSET = 100

# Transformation matrix that can be used to deskew images/coordinates
# into proper cartesian space
MAT = np.eye(4)
MAT[2, 0] = -np.cos(np.deg2rad(ANGLE)) * DZ / DX


def z_offset_from_name(name: str, z_off_step=0.1, center=25) -> float:
    stack = int(name.split("stack")[1][:4])
    offset = (stack - center) * z_off_step
    return round(offset, 2)


def deskew(
    data: np.ndarray,
    dx: float = DX,
    dz: float = DZ,
    angle: float = ANGLE,
    offset: int = 0,
) -> np.ndarray:
    """Deskew a numpy array."""
    data = np.clip(data, offset, None) - offset
    deskew_func, _, _ = get_deskew_func(data.shape, dx=dx, dz=dz, angle=angle)
    return deskew_func(data)


def deskew_tiff(path: Union[str, Path], **kwargs) -> np.ndarray:
    """Deskew a tiff file."""
    return deskew(imread(path), **kwargs)


def _deskew_and_save_tiff(pth: Path, overwrite=True, **kwargs):
    if "_deskewed" in str(pth):
        return
    dest = str(pth).replace(".tif", "_deskewed.tif")
    if Path(dest).exists() and not overwrite:
        return
    out = deskew_tiff(pth, **kwargs)
    imwrite(dest, out.astype(np.uint16))


def read_meta(pth: str) -> dict:
    text = Path(pth).read_text()

    ZG = "Z Galvo Offset, Interval (um), # of Pixels for Excitation (0) :"
    zoff = float(text.split(ZG)[1].split("\n")[0].split()[0])

    SPZT = "S PZT Offset, Interval (um), # of Pixels for Excitation (0) :"
    zstep = float(text.split(SPZT)[1].split("\n")[0].split()[1])

    return {"z_step": zstep, "z_offset": zoff}


def deskew_folder(pth: str):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_deskew_and_save_tiff, Path(pth).glob("*.tif"))


def window3d(shape, fwin=signal.windows.kaiser, **winkwargs) -> np.ndarray:
    """Return 3-dimensional window."""
    D, H, W = shape
    d = fwin(D, **winkwargs)
    h = fwin(H, **winkwargs)
    w = fwin(W, **winkwargs)

    m1 = np.outer(np.ravel(h), np.ravel(w))
    win1 = np.tile(m1, np.hstack([D, 1, 1]))

    m2 = np.outer(np.ravel(d), np.ones([1, H]))
    win2 = np.tile(m2, np.hstack([W, 1, 1]))
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    return np.multiply(win1, win2)


def center_crop(data, size=128, cz=None, cy=None) -> np.ndarray:
    _cz, _cy, cx = np.array(data.shape) // 2
    cz = cz if cz is not None else _cz
    cy = cy if cy is not None else _cy
    half = size // 2
    return data[
        cz - half : cz + half + 1,
        cy - half : cy + half + 1,
        cx - half : cx + half + 1,
    ]


def fft3(data: np.ndarray) -> np.ndarray:
    """Return 3-dimensional fft of data."""
    data = data * window3d(data.shape[-3:], beta=6)
    return fftshift(fftn(ifftshift(data)))


def radial_profile(data: np.ndarray, around: Optional[Tuple[int, int]] = None):
    """Create radial profile of 2 array, centered around `around`.

    If not provided, `around` is the center of the image.
    """
    cy, cx = np.array(data.shape) // 2 if around is None else around
    y, x = np.indices(data.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr


def radial_profile_2d(data, around=None):
    """Embarassing way to get a radial profile of a 3d array on the ZR plane"""
    return np.stack([radial_profile(plane, around=around) for plane in data])


def prep_data(data, size=128, cz=None, cy=None) -> np.ndarray:
    if size:
        data = center_crop(data, size=size, cz=cz, cy=cy)
    windowed = data * window3d(data.shape, signal.windows.cosine)
    deskewed = deskew(windowed)
    return center_crop(deskewed, size=size) if size else deskewed


def main(data, nz=48, cz=None, cy=None):
    if isinstance(data, str):
        data = imread(data)
    prepped = prep_data(data, cz=cz, cy=cy)
    fprepped = fft3(prepped * window3d(prepped.shape, beta=6))
    rplane = radial_profile_2d(np.abs(fprepped))
    # cropped
    return rplane[64 - nz // 2 : 64 + nz // 2, :58]


def explore(path, viewer=None):
    if viewer is None:
        import napari.viewer

        viewer = napari.viewer.current_viewer() or napari.viewer.Viewer()

    nz = 48
    angles = []
    abss = []
    for image in sorted(Path(path).glob("*.tif")):
        offset = z_offset_from_name(image.name)
        if 10 * offset % 2 != 0:
            continue

        data = imread(image)
        prepped = prep_data(data)
        fprepped = fft3(prepped * window3d(prepped.shape, beta=6))
        rplane = radial_profile_2d(np.abs(fprepped))
        abss.append(rplane[64 - nz // 2 : 64 + nz // 2, :58])
        rplane = radial_profile_2d(np.angle(fprepped))
        angles.append(rplane[64 - nz // 2 : 64 + nz // 2, :58])

    viewer.add_image(np.stack(abss))
    viewer.add_image(np.stack(angles))
    # viewer.add_image(np.stack(preppedp))


def full_fft(path: str, offset=100):
    data = np.clip(imread(path), offset, None) - offset
    windowed = data * window3d(data.shape, signal.windows.cosine)
    deskewed = deskew(windowed)
    fdeskewed = fft3(deskewed)
    mag, phase = np.abs(fdeskewed), np.angle(fdeskewed)
    magrot = radial_profile_2d(mag)
    phaserot = radial_profile_2d(phase)
    return np.stack([magrot, phaserot])


def random_patch_slices(shape, n: int = 1, patch_size: int = 128):
    # shift coords to deskewed space
    a = patch_size // 2
    coords = np.stack([np.random.randint(a, x - a, size=n) for x in shape])
    coords = (np.linalg.inv(MAT[:3, :3]) @ coords).T.astype(int)
    for c in coords:
        yield tuple(slice(x - a, x + a) for x in c)


def file_to_patches(
    path: str, n_patches: int = 6, patch_size: int = 128, offset: int = 100
) -> np.ndarray:
    data = np.clip(imread(path), offset, None) - offset
    windowed = data * window3d(data.shape, signal.windows.cosine)
    deskewed = deskew(windowed)
    patch_slices = random_patch_slices(data.shape, n_patches, patch_size)
    return np.stack([deskewed[p] for p in patch_slices])


def folder_to_patches(path, tries=0):
    path = Path(path)
    patch_dir = path / "patches"
    patch_dir.mkdir(exist_ok=True, parents=True)
    for image in sorted(path.glob("*.tif")):
        stack = int(image.name.split("stack")[1][:4])
        prefix = f"{path.name}_stack{stack:04}_"
        if any(patch_dir.glob(f"{prefix}*")):
            print("skipping", image)
            continue
        print("processing", image)
        try:
            for i, patch in enumerate(file_to_patches(image)):
                np.savez(patch_dir / f"{prefix}{i:03}.npz", patch)
        except Exception as e:
            if tries >= 5:
                raise e
            print("retrying", image.name)
            folder_to_patches(path, tries=tries + 1)


def preview_data(dataset: "Dataset", n=3, figsize=(3, 10)):
    import matplotlib.pyplot as plt

    data, label = next(iter(dataset))
    print("data shape", data.shape, data.dtype)
    print("label shape", label.shape)
    print("dataset length", len(dataset))

    nC = data.shape[-3]
    _, axes = plt.subplots(n, nC, figsize=figsize)
    for row, idx in enumerate(np.random.randint(0, len(dataset), size=n)):
        img, label = dataset[idx]
        for c in range(nC):
            cimg = img[c]
            cimg = cimg[0] if cimg.ndim == 3 else cimg
            if c == 0:
                cimg = cimg ** 0.01
            idx = (row, c) if nC > 1 else row
            axes[idx].imshow(cimg)
            axes[idx].set_title(f"{label.item():0.2f}")
            axes[idx].axis("off")


def characterize(model, dataset, device=None, criterion=None):
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    records = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)

            prediction = model(x[None])
            R = {
                "file": str(dataset.files[i]),
                "class": dataset.get_class(i),
                "gt": y.item(),
                "pred": prediction.item(),
            }
            if criterion is not None:
                R["err"] = criterion(prediction, y[None]).item()
            records.append(R)

    return pd.DataFrame(records)


def plot_results(model, dataset, device):
    df = characterize(model, dataset, device)
    