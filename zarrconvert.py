from pathlib import Path

import numpy as np
import zarr
from tifffile import TiffFile


data = Path("../mlfocus_data")
zarr_out = data / "zarr"
zarr_out.mkdir(exist_ok=True, parents=True)

shapes = []
nfiles = 0


def convert(subdir):
    if subdir == zarr_out:
        return

    dest = zarr_out / f"{subdir.name}.zarr"
    if dest.exists():
        print("Skipping", subdir.name)
        return

    stacks = []
    for fname in sorted(subdir.glob("*.tif")):
        with TiffFile(fname) as tif:
            print(fname.name)
            stacks.append(tif.asarray())
    stack = np.stack(stacks)

    store = zarr.DirectoryStore(dest)
    root = zarr.group(store=store, overwrite=True)
    z1 = root.zeros("raw", shape=stack.shape, chunks=(1, 64, 64, 64), dtype="uint16")
    z1.attrs["resolution"] = (1, 1, 1, 1)
    z1[:] = stack
    print("saved to", zarr_out / f"{subdir.name}.zarr", stack.shape)


def main():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(convert, sorted(data.iterdir()))

if __name__ == "__main__":
    main()