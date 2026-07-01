# Spatial detector

Detects where the cells are. Three modules: `laplacian.py` (the ∇²G operator),
`scattering.py` (the per-frame energy front-end), and `segmenter.py` (the trained
head that turns the energy into per-pixel soma probability and then instances).

## The operator (`ParametricLoG2d`, `laplacian.py`)

The classical pipeline runs a bank of Laplacian-of-Gaussian filters at hand-picked
scales, then hand-tuned intensity/contrast/local-max rejection. This keeps the
form (a multi-scale ∇²G filterbank) but learns the two things that were guessed:
the scale bank fits the real neuron-size distribution, and a learned head replaces
the fixed gates.

```
LoG(x, y; σ) = σ² · ((x² + y² − 2σ²) / σ⁴) · G_σ(x, y)
```

The leading σ² is the conventional scale-normalisation (Lindeberg 1998). The
kernel is sign-flipped so a bright blob gives a positive response and demeaned so
flat background contributes nothing (DC rejection, as in the temporal stage).
Scales are stored as log-σ and exponentiated, so σ > 0 and the bank is
differentiable in the scales. A blob of radius r is matched by σ = r/√2. The
per-scale peak is not equalised across scales, so a single channel is not a clean
size selector; size is recovered by the head combining all channels and by the
spatial extent of each response.

## The energy front-end (`SpatialScatterDetector.energy`, `scattering.py`)

A projection detector would smooth each frame, reduce over time, then run a LoG
bank: two spatial scales straddling a nonlinearity. Applying ∇²G *per frame*
first collapses that, because the kernel's own Gaussian is the smoothing. The
result is a first-order scattering coefficient:

```
S(x, σ) = var_t | ∇²G_σ *_x Y(·, t) |
```

`energy()` returns a stack of temporal moments of the same ∇²G-filtered movie, in
fixed order:

- **structural** (mean): ∇²G of the mean image, responds to all visible somata
  (the primary detector, since annotations mark every visible cell);
- **max**: ∇²G on a robust max projection, the substrate ROIs were drawn on;
  surfaces sparse low-baseline cells the mean/variance channels miss;
- **variance**: active-blob energy (zero-mean ∇²G means subtracting the temporal
  mean removes static structure and keeps active blobs);
- **coherence**: noise-robust coherent activity from off-diagonal moments.

It runs in two passes over frame chunks, so the full per-frame response is never
materialised. The max projection uses top-k (k-th largest ≈ the (1−q) percentile),
a partial sort ~20× faster than `quantile` that still rejects single-frame
hotspots.

## The segmenter (`SpatialSegmenter`, `segmenter.py`)

Consumes `energy()` as its feature front-end and adds a small U-Net that maps the
moment stack to per-pixel soma-probability logits. Trained with a focal + soft
Dice loss, which is invariant to the foreground fraction (the focal term
down-weights the easy background so sparse frames still learn, where plain BCE
would not). `predict_prob` runs the model to a probability map; `segment_instances`
thresholds it and runs a centroid-seeded watershed so touching cells split into
one basin per annotated soma rather than merging under connected components.
`extract_instances` (in `laplacian.py`) is the lighter peak-local-max read-out of
centroids from a cellness map.

## Choices and limits

- **Max projection may be dropped.** It is the most sensitive channel for
  rarely-firing cells but is an order statistic that does not fold into the moment
  algebra, so the integrable form is tried first; if recall on sparse cells
  suffers, max returns as an auxiliary channel.
- **Neuropil correction is disabled for organoids.** Radial-profile neuropil
  correction assumes a geometry and optics the organoid recordings do not match,
  so it is off rather than applied blindly.
