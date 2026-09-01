"""Activity stage: the bridge from spatial segmentation to functional readouts.

`segment` gives one fluorescence trace per ROI (a label image plus a weighted
mean). This subpackage turns those traces into the functional products the
gallery and the group analysis consume: baseline-corrected dF/F0 (`baseline`),
OASIS spike inference (`deconvolution`), and the interactive per-recording HTML
gallery (`gallery`, with its page in `gallery.html`). `roi_adapter` turns a label
image into the per-ROI geometry and projections the gallery draws.

The stage runner (`orcann.pipeline.run_activity`) writes the calcium-format
per-recording folder that `orcann.analysis` reads; nothing here reaches back
into the spatial stage beyond the on-disk contract in
`orcann.pipeline.inference`.
"""
