from pathlib import Path

import gunpowder as gp
import matplotlib.pyplot as plt

zarr_dir = Path("../mlfocus_data") / "zarr"
raw = gp.ArrayKey("RAW")


# source = gp.ZarrSource(
#     str(zarr_dir / "beads1.zarr"),
#     {raw: "raw"},
#     {raw: gp.ArraySpec(interpolatable=True)},
# )
source = (
    tuple(
        gp.ZarrSource(
            str(zarr_dir / "beads1.zarr"),
            {raw: "raw"},
            {raw: gp.ArraySpec(interpolatable=True)},
        )
        for _ in zarr_dir.iterdir()
    )
    + gp.MergeProvider()  # type: ignore
)

# read on various providers
# use random provider instead.


pipeline = source
# formulate a request for "raw"
request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0, 0, 0), (0, 0, 64, 64))

# build the pipeline...
with gp.build(pipeline):

    # ...and request a batch
    batch = pipeline.request_batch(request)

# show the content of the batch
print(f"batch returned: {batch}")
plt.imshow(batch[raw].data)
plt.show()
# %%
