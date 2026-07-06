"""Activity stage: the bridge from spatial segmentation to functional readouts.

`segment` gives one fluorescence trace per ROI (a label image plus a weighted
mean). This subpackage turns those traces into the functional products the
gallery and the group analysis consume: baseline-corrected dF/F0 (`baseline`),
OASIS spike inference (`deconvolution`), and the interactive per-recording HTML
gallery (`gallery`, `movie_gallery`). `roi_adapter` presents an OrCaNN label
image to the gallery in the shape it expects (a seeds view + a projection set),
so the calcium-pipeline gallery renders unchanged over segmenter outputs.

The stage runner (`orcann.pipeline.run_activity`) writes the calcium-format
per-recording folder that `orcann.analysis` reads; nothing here reaches back
into the spatial stage beyond the on-disk contract in
`orcann.pipeline.inference`.
"""
