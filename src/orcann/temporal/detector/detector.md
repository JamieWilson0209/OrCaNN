# Temporal detector

`orcann/temporal/detector.py`. Detects when each cell fires: a shape-parameterised
wavelet bank whose multi-scale response feeds a learned per-bin rate, plus a
geometric read-out of event duration.

## The wavelet bank (`ParametricDoGWavelet1d`)

The classical method correlates a trace against a bank of fixed, symmetric Ricker
wavelets. A symmetric wavelet is a poor match for a fast-rise/slow-decay calcium
event: as the wavelet widens the match degrades, so slow asymmetric events are
missed. OrCaNN keeps the wavelet idea but lets the mother rotate inside the
derivative-of-Gaussian family:

```
ψ_θ(t) = cos(θ)·R(t) + sin(θ)·D(t)
  R(t) ∝ (1 − (t/a)²)·exp(−t²/2a²)   symmetric, centre-positive  (−∇²G)
  D(t) ∝ (t/a)·exp(−t²/2a²)          antisymmetric               ( ∇G )
```

At `θ = 0` this is exactly the Ricker wavelet (the 1-D twin of the spatial LoG).
At `θ ≠ 0` the lobes are unequal, giving the matched detector for an asymmetric
transient. R and D are both zero-mean (Gaussian derivatives integrate to zero),
so every `ψ_θ` is zero-mean and DC rejection holds for any `θ`.

The bank stores per-channel log-scales and one shared asymmetry, generates the
`(K, 1, L)` kernel stack on the fly, demeans and unit-L2-normalises each kernel,
then convolves. A scale `a` maps to a timescale in seconds through the Torrence &
Compo (1998) DOG `m=2` Fourier factor, `2π/√2.5 ≈ 3.974`, held fixed.

## The rate model (`TemporalRateModel`)

Two layers. Layer 1 is the wavelet bank above. Layer 2 is a thin head (a 1×1
weighting across scales, a short temporal convolution, then softplus) mapping the
multi-scale response to a non-negative per-bin event rate. `forward` returns the
rate; `response` exposes the layer-1 output for the duration read-out.

During training, whole scale channels are randomly zeroed (`scale_dropout`) so the
head cannot lean on the one or two scales that dominate the training indicators;
this forces a scale-robust read-out that transfers to a held-out indicator. It is
identity at eval, so the default of `0.0` changes nothing.

## Input standardisation (`standardize_trace`)

Every trace is put in SNR units: detrend with a rolling low-percentile baseline,
then divide by the robust noise σ (MAD of successive differences). The same call
runs in training and inference, so the model never sees a shifted input
distribution between the two, and it is well defined for raw F or ΔF/F. This
achieves by normalisation what noise-matching achieves by tuning training noise to
the test condition, without having to measure the organoid noise level.

## Detection and duration (`detect_transients`, `read_durations`)

`detect_transients` is the single detection path used by both the batch runner and
the visualiser. It standardises the trace, runs the model to a rate, then takes
peaks above a height floor (a low percentile of the rate, gating baseline) that
also clear a prominence requirement (the event must stand out locally).

Duration has no labels, so it is read geometrically rather than learned. At each
event the layer-1 response across scales is softmax-weighted and the timescales
combined in log-space (a soft version of "the scale whose shape matches best").
Because the mother is the learned, asymmetric one, asymmetric transients are read
at their true width instead of being pushed to fine scales.

## Choices and limits

- **Rate is supervised, duration is geometric.** Keeping them on separate paths
  means the supervised rate head cannot distort the duration read-out.
- **Honesty at the frame rate.** The target is a smoothed per-bin rate, never
  sub-frame spike times; at 2 Hz the input cannot support finer timing. Reported
  durations are characteristic timescales whose *ordering* across events is
  faithful; treat absolute seconds on out-of-domain organoid data as indicative,
  and prefer rate shape and event structure over absolute counts.
- **SNR normalisation over noise-matching** (see above) avoids needing the
  organoid noise level; training adds noise *before* standardising so the model
  still sees a range of SNRs.
