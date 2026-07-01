# Spatial training

`orcann/spatial/training.py` (the training loop) and `orcann/spatial/annotations.py`
(annotation intake). Fits the segmenter on manually annotated recordings.

## Annotation intake (`annotations.py`)

Annotations are ImageJ/FIJI ROI sets and nothing else: a `RoiSet.zip` (or a
single `.roi`) of vector polygons, exported from the ROI Manager via More > Save.
`_rasterize_imagej_rois` reads the set with `roifile` and rasterises each polygon
into one integer label in an `(H, W)` image (0 = background, `k` = the k-th ROI).

ImageJ stores coordinates as (x, y) = (column, row) while arrays are (row, col);
the x→col, y→row mapping is explicit because getting it backwards is a silent
transpose that mirrors every cell. Run `scripts/check_annotation.py` on the first
real recording to eyeball the loaded ROIs on the max projection before training.
Any other file type is rejected with a message pointing back to the ImageJ
export step.

## Data and patches (`training.py`)

A `SegRecording` pairs a `(T, H, W)` movie with an `(H, W)` instance label image.
`load_seg_recording` reads the movie and either a `.npy` label or an ImageJ ROI
set, with an optional `min_area` filter that drops tiny ROIs and relabels.

`_seg_patches` samples training patches with a fixed fraction centred on a random
cell, so sparse foreground is actually seen rather than swamped by background
patches.

## Loss, metrics, loop

The loss is focal + soft Dice (see the detector doc): Dice is invariant to the
foreground fraction and the focal term down-weights easy background, so the sparse
foreground (0.6–19% of a frame) still learns. IoU is reported at the best
threshold over a sweep (`best_iou`), because 0.5 is arbitrary for an imbalanced
soft map and for 3–8 px cells the cut point moves IoU a lot.

`train_segmenter` records the training frame size (so inference can auto-rescale
new recordings), uses a cosine LR schedule, and checkpoints every epoch so a
queue-killed job is not a total loss. `synthetic_sources` makes non-round
synthetic cells for the self-test, so segmentation has shape to learn.
