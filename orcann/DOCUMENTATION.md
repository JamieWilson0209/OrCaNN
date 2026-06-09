# OrCaNN — a single derivative-of-Gaussian operator for spatial and temporal calcium analysis

**Status:** full skeleton implemented, hardened for cluster scale, and tested end-to-end on synthetic data; not yet trained on real recordings. CASCADE ground truth is staged; the annotated `.nd2` recordings are pending.
**Scope of this document:** what the project is, the full mathematical decomposition of every stage, the chain of decisions that produced the current design, the engineering that makes it run cleanly at scale, and an explicit ledger of what is and is not justified by evidence so far.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $`Y(x,t)`$ | movie: fluorescence at spatial position $`x\in\mathbb{R}^2`$, frame $`t`$ |
| $`G_\sigma`$ | Gaussian of scale $`\sigma`$ |
| $`\nabla^2 G_\sigma`$ | Laplacian-of-Gaussian (LoG); the Marr–Hildreth operator |
| $`\nabla G_\sigma`$ | first derivative of Gaussian (antisymmetric) |
| $`*`$, $`*_x`$ | convolution; convolution in the spatial argument only |
| $`\bar y(x)`$ | temporal mean image, $`\mathrm{mean}_t Y(x,t)`$ |
| $`L(x,t)`$ | band-passed movie $`\nabla^2 G_\sigma *_x Y(\cdot,t)`$ |
| $`C(u,v)`$ | pixel–pixel temporal second moment $`\mathrm{mean}_t[Y(u,\cdot)Y(v,\cdot)]`$ |
| $`f_s`$ | frame rate (Hz); target regime is $`2\ \mathrm{Hz}`$ |
| $`a,\ \tau_k,\ \theta`$ | wavelet scale (samples); characteristic timescale (s); mother asymmetry |

The single object underneath the whole project is the **normalised derivative-of-Gaussian family**. Everything below is a deployment of it in a different dimension, at a different scale, pooled at a different temporal moment, or on a different side of a nonlinearity.

---

## 1. The unifying idea: one operator, two domains

The classical pipeline used two methods that looked unrelated: **Laplacian-of-Gaussian (LoG) blob detection** to find neurons in space, and a **Ricker (Mexican-hat) wavelet transform** to find transients in time. They are the same operator.

The Ricker wavelet is the normalised negative second derivative of a Gaussian:

```math
\psi_{\text{Ricker}}(t)\ \propto\ \Big(1-\tfrac{t^2}{a^2}\Big)e^{-t^2/2a^2}\ =\ -\,a^2\,\frac{d^2}{dt^2}G_a(t).
```

The LoG is the same thing in two dimensions:

```math
\nabla^2 G_\sigma(x)\ =\ \frac{|x|^2-2\sigma^2}{\sigma^4}\,G_\sigma(x).
```

Both are $`\nabla^2 G`$. LoG sweeps a **spatial** scale $`\sigma`$ for blobs of varying radius; the Ricker CWT sweeps a **temporal** scale $`a`$ for transients of varying duration. A blob of radius $`r`$ is matched at $`\sigma=r/\sqrt2`$; a transient at the $`a`$ whose Fourier period equals its width.

**Design commitment that follows.** The architecture is *one generating function deployed across a scale group*, not a free bank of independent filters. A learnable bank that drifted from $`\nabla^2 G`$ would still detect things, but it would dissolve the unification, make the duration axis uninterpretable, and break the claim that the two stages are the same operator. Every learnable component **generates** its kernels from $`\nabla^2 G`$ with a few interpretable parameters (scales, plus one temporal asymmetry angle); the weights are never free. Both trained stages turn out to share the shape *learnable derivative-of-Gaussian wavelet → nonlinearity → head*.

---

## 2. The ground-truth reckoning (why the two stages are trained differently)

The two stages have radically different supervision, and the asymmetry is structural.

**Spatial — strong, in-domain.** ~85 manually annotated 1-photon organoid recordings at 2 Hz. The annotations are **ROIs drawn on the max-projection summary image, covering all visible somata regardless of activity** (see §3.7 — this fixes the detection target). Genuine in-domain supervision; the spatial stage earns a real precision/recall claim.

**Temporal — no in-domain labels exist.** No spike-time labels at any frame rate. High-rate paired ephys+calcium ground truth *is* available (CASCADE; staged in `data/public_gt/`), but it is the wrong domain: in-vivo mouse/zebrafish, two-photon, single neurons, GCaMP/OGB indicators — not organoid, not Fluo-4, not widefield. For organoids the native ground-truth instruments are MEA and patch-clamp, which give population/single-cell electrical events, not per-ROI spike trains.

**Decision.** Train the temporal stage on public spike trains **resampled to 2 Hz and standardised** to the regime of the real recordings, trying cross-indicator transfer **first** (LOIO, §5.2) and keeping a synthetic forward model as the documented fallback. The honest target at 2 Hz is a **per-bin rate**, never sub-frame spike times. Because spatial and temporal labels live in different datasets, the stages are **trained separately** and composed only at inference — which, given the elegance priority, is a feature.

---

## 3. Spatial stage — full decomposition

### 3.1 The classical pipeline, stated as operators

Per-frame Gaussian smoothing at $`\sigma_s`$ → nonlinear temporal projection $`R_t\in\{\max,\ \mathrm{std},\ \text{local-corr}\}`$ → LoG detection at $`\sigma_d`$ → Otsu contour per seed:

```math
D(x)\ =\ \nabla^2 G_{\sigma_d} *_x\ R_t\big[(G_{\sigma_s} *_x Y)(\cdot,t)\big].
```

### 3.2 Per-frame smoothing **is** part of the LoG

Two facts collapse smoothing and detection into one operator wherever the projection is linear:
1. **Gaussians compose:** $`G_a * G_b = G_{\sqrt{a^2+b^2}}`$.
2. **The Laplacian commutes with convolution:** $`\nabla^2(f*g)=(\nabla^2 f)*g`$.

Hence $`\nabla^2 G_{\sigma_d} *(G_{\sigma_s} * I) = \nabla^2 G_{\sqrt{\sigma_s^2+\sigma_d^2}} * I`$. The hand-tuned smoothing was never outside the convolution — it shifted the LoG's effective scale and set a hard floor (nothing finer than $`\sim\sigma_s`$ survives). In a learnable bank that floor is just the lower edge of the scale range, so smoothing's role becomes *learned*, not guessed. **There is no `smooth_sigma` parameter in OrCaNN.**

### 3.3 The one part that does **not** fold in — and the covariance view

The projection is nonlinear, and $`\max(\mathrm{smooth}(\cdot))\neq\mathrm{smooth}(\max(\cdot))`$. The obstruction splits by the **algebraic order** of the projection:

- **Mean (linear).** Folds trivially: per-frame and post-projection smoothing are identical; the cascade is a single LoG at $`\sqrt{\sigma_s^2+\sigma_d^2}`$ on $`\bar y`$.
- **Std / local-correlation (quadratic).** Fold cleanly, but onto the **covariance operator**, not the image. With $`f=G_\sigma *_x Y`$,

```math
\mathrm{var}_t f(x)\ =\ \big[(G_\sigma\otimes G_\sigma)\,C\big](x,x)\ -\ \big[(G_\sigma * \bar y)(x)\big]^2 ,
```

i.e. smooth $`C(u,v)`$ in **both** spatial slots, read the diagonal, subtract the smoothed-mean squared. *(Verified numerically: direct vs covariance-operator form agree to $`3\times10^{-16}`$.)* The energy-feature / Wiener–Khinchin principle: a quadratic feature of a linear-filter output is a linear functional of the autocorrelation.
- **Max (order statistic).** Genuinely irreducible — outside the moment algebra. This is **why max is a separately-computed channel** (§3.4), not something folded into the convolution.

### 3.4 The detector is a hierarchy of temporal moments of one band-passed movie

Applying the LoG **per frame**, before the nonlinearity, unifies smoothing and detection (the LoG's own Gaussian *is* the per-frame smoothing). What was a single scattering coefficient is, in general, a **ladder of temporal statistics of the same band-passed movie $`L=\nabla^2 G_\sigma *_x Y`$**, the pooling statistic indexing the channel — all sharing one learnable $`\nabla^2 G`$ bank:

| Channel | Pooling | Detects | Role |
|---|---|---|---|
| **structural** | 0th moment: $`\mathrm{mean}_t L = \nabla^2 G_\sigma*\bar y`$ | all visible somata (active or silent) | **the annotation target** |
| **max** | $`L^\infty`$: $`\nabla^2 G_\sigma`$ on a robust max projection | sparse, low-baseline cells | the ROI-drawing substrate |
| **variance** | 2nd, diagonal: $`\mathrm{var}_t L`$ | active blobs | activity feature |
| **coherence** | 2nd, off-diagonal (§3.5) | coherent activity | noise-robust activity feature |

The **structural** channel is the LoG of the mean image and is the primary detector, because the manual ROIs mark every visible cell (§3.7). The **max** channel is the irreducible order statistic of §3.3, computed explicitly as $`\nabla^2 G_\sigma`$ applied to a **top-$`k`$ robust max projection** ($`k`$-th largest $`\approx`$ the $`(1{-}q)`$ percentile, default $`q{=}0.99`$; top-$`k`$ is a partial sort, ~20× faster than `quantile`, and rejects single-frame hotspots without per-frame smoothing). It matches the substrate the annotators drew on and surfaces sparse low-baseline firers invisible to mean/variance. **Variance** uses the centred 2nd moment because $`\nabla^2 G`$ is zero-mean, so subtracting $`\mathrm{mean}_t L`$ removes static structure and isolates *active* blobs.

Default configuration: `structural + max + variance` on, coherence off.

### 3.5 The correlation-energy (coherence) channel

The variance channel is the **diagonal** of the band-passed temporal covariance; a faint cell competes against the full noise variance. The **off-diagonal** has no such floor:

```math
C_{\mathrm{coh}}(x,\sigma)\ =\ \frac{1}{|\Delta|}\sum_{\delta\in\Delta}\mathrm{Cov}_t\!\big(L(x,\cdot),\,L(x{+}\delta,\cdot)\big).
```

For spatially-independent noise $`\mathbb{E}[\mathrm{Cov}_t]\approx 0`$ ($`\delta\neq0`$), so a faint-but-coherent cell stands out even when buried in variance. This is the principled, in-framework form of the old "local correlation image," and it cannot be reduced to a per-frame filter (off-diagonal information needs cross-pixel temporal products). **Empirical finding (synthetic):** helps faint-cell recall in a moderate-noise band ($`0.40\to0.60`$ at noise sd $`0.12`$), neutral elsewhere, never harmful; the high-noise tie is partly a short-recording artefact ($`\mathrm{Var}`$ of the estimate $`\propto 1/T`$), so long real recordings should widen the useful regime. **Opt-in; the real spatial harness decides.**

### 3.6 Learnable realisation

- **`ParametricLoG2d`** generates $`K`$ kernels from learnable $`\log\sigma`$: $`-\sigma^2\nabla^2 G_\sigma`$ (bright blob → positive), demeaned (DC rejection). Only the scales are free. Shared across all channels. *Note:* the leading $`\sigma^2`$ is the conventional scale-normalisation, but in practice it does **not** equalise peak response across scales (empirically response grows with $`\sigma`$ rather than peaking at the matching blob size); size discrimination is carried by the learned head combining all $`K`$ channels and by the spatial extent of the response, not by a single winning scale.
- **Head:** per-channel **RMS normalisation** (heavy-tailed scattering coefficients; RMS is stable for the zero-mean structural channel where mean-normalisation would blow up) → $`1\times1`$ across channels → $`3\times3`$ refine → $`1\times1`$ → cellness logit. Replaces the hand-set intensity/contrast/local-max gates. ~2.5k trainable parameters.
- **Footprints (replaces Otsu entirely):** thresholded cellness assigned to the nearest peak → irregular per-instance soft masks; non-circular boundaries preserved, but the boundary is *learned from annotations*, not assumed bimodal. Optional **activity-gating** sharpens each footprint with its own top-activity frames — the seam where the temporal stage can later drive frame selection.
- **Detection threshold** is one inference scalar (`det_threshold`, default 0.5) + `min_distance`, not a per-blob Otsu computation.
- **Radius bank** seeded `auto` from the annotation radius distribution — the ROIs carry the cell sizes, so pixel-size metadata is not needed.

### 3.7 Detection target: all somata, not active cells

The annotations mark **all visible cells regardless of activity**. An activity-only detector (`variance`) is therefore the wrong target — it scores 0 recall on silent cells, faithfully, which is a *target mismatch*, not a bug. The structural (0th-moment) channel is the fix: it detects somata by baseline morphology in the mean image, exactly what the annotator traced. Validated on synthetic data with silent-but-baseline-bright cells: `variance`-only → 0.0 silent recall; any `structural`-containing config → 1.0. Detecting all cells and labelling each active/silent (via the activity channels) is strictly richer than activity-only detection, and the active fraction is itself a network-participation readout.

---

## 4. Temporal stage — full decomposition

### 4.1 The classical method and its bias

A bank of **fixed, symmetric** Ricker wavelets, ridge extraction, and a normalised-correlation gate. Failure mode: a fast-rise/slow-decay calcium transient matches a symmetric Mexican hat progressively worse as the wavelet broadens, so slow asymmetric events slip the gate.

### 4.2 The shape-parameterised mother — a rotation in the (∇G, ∇²G) plane

```math
\psi_\theta(t)\ =\ \cos\theta\,R(t)\ +\ \sin\theta\,D(t),\qquad
R\ \propto\ \big(1-\tfrac{t^2}{a^2}\big)e^{-t^2/2a^2}\ (-\nabla^2 G),\quad
D\ \propto\ \tfrac{t}{a}\,e^{-t^2/2a^2}\ (\nabla G),
```

demeaned and unit-$`L^2`$ normalised. $`\theta=0\Rightarrow`$ pure Ricker $`\Rightarrow`$ the exact 1-D twin of the spatial LoG (unification holds at the symmetric setting). Both $`R`$ and $`D`$ integrate to zero, so $`\psi_\theta`$ is zero-mean for **all** $`\theta`$ — admissibility and DC rejection are free. $`\theta\neq0`$ tilts the lobes into the matched detector for an asymmetric transient.

**Why only time gets the extra DoF.** Spatial blobs are isotropic → pure symmetric $`\nabla^2 G`$, scales only. Temporal transients are asymmetric → one shape angle. A genuinely multi-shape regime (e.g. a symmetric burst envelope coexisting with an asymmetric somatic transient) would justify a **small structured set** of mothers, each still dilated across scale — never an unstructured dictionary, and only if the data's residuals demand it.

### 4.3 Scale, timescale, transform

$`\lambda = 2\pi a/\sqrt{2.5}\approx 3.974\,a`$ samples (Torrence & Compo, $`m=2`$); $`\tau=\lambda/f_s`$. Transform $`W(a,t)=(\psi_{\theta,a}*x)(t)`$.

### 4.4 What is learned vs geometry

- **Rate head (supervised):** learned map from the multi-scale response to a non-negative **per-bin rate** $`\hat r=\mathrm{softplus}(\text{head}(W))`$, trained on resampled public spike trains. Loss $`=\text{MSE}(\hat r,r) + (1-\mathrm{corr}(\hat r,r))`$. ~1.7k params.
- **Duration (geometry, label-free):** soft-argmax over scale of $`|W(\cdot,b)|`$ at each event bin → $`\hat\tau(b)=\exp(\sum_k w_k\log\tau_k)`$, read with the *asymmetric* mother. **$`\hat\tau`$ is an inflated multiple of the decay constant $`\tau`$ (≈5×; ≈8× in synthetic tests), not $`\tau`$ itself** — faithful in ordering/separation, not absolute seconds.

Rate (supervised) and duration (geometry) are on separate paths so the rate head cannot distort the duration read-out, and duration needs no labels (none exist).

**Optional regulariser — scale-channel dropout.** `scale_dropout` (default 0, off) randomly zeros whole wavelet-scale channels during training so the head cannot over-rely on the scale(s) that dominate the training indicators, encouraging a scale-robust read-out for transfer. It is identity at eval, so inference and the saved model are unaffected. Evaluate its effect via LOIO (it should help the kinetically-distinct held-out folds most).

### 4.5 Input convention — one standardisation everywhere

The model input is produced by a single `standardize_trace`, used **identically** in temporal training prep and at inference, so the model never faces a shifted input distribution:
1. **detrend** — subtract a rolling low-percentile baseline (removes drift/bleaching, flattens baseline to ~0);
2. **normalise** — divide by the robust noise σ (MAD of differences) → SNR units.

Normalising to unit noise makes the model invariant to absolute noise level — achieving by normalisation what CASCADE achieves by matching training noise to each test condition, *without needing to measure the organoid noise*. Training augments by adding noise **before** standardisation, so the model still sees a range of SNRs.

### 4.6 Discrete transient extraction (post-hoc, not learned)

The model emits a continuous rate; discrete transients are extracted from it by `detect_transients`: a height floor (a low percentile of the rate, gating quiet baseline) **and** a prominence requirement (`min_prominence`, in rate units — the analogue of OASIS's $`s_{\min}`$: a peak must rise clearly above its local surroundings), with a minimum inter-event spacing. Prominence is the main knob and removes the over-firing that a fraction-of-max threshold produces on busy ROIs. **Caveat:** the threshold is *absolute* in CASCADE-calibrated rate units, which is the part least likely to transfer to organoid Fluo-4. The robust options are a relative threshold (e.g. $`k\cdot`$noise per recording, or the valley of the per-recording prominence histogram) and leaning on continuous-rate measures rather than absolute event counts for cross-domain analysis.

---

## 5. Cross-indicator training and the degradation math

### 5.1 Degrading high-rate ground truth to the target regime

- **Trace:** anti-aliased resample (polyphase / bin-average), **not** decimation (which would alias fast structure into the signal).
- **Noise/standardisation:** noise added as SNR augmentation, then `standardize_trace` (§4.5) — detrend + unit-noise.
- **Target:** bin spike times to 2 Hz and Gaussian-smooth (σ ≈ 1 s) → per-bin rate. Never discrete sub-bin spikes.

### 5.2 Leave-one-indicator-out (LOIO) — the go/no-go

Train on all-but-one indicator, test on the held-out one. **Transfer gap** = within-train − held-out median correlation. Small gap + high held-out correlation ⇒ the cross-indicator bet holds; a collapse on one held-out indicator ⇒ fall back to synthetic. **Instrument check (synthetic four-indicator stand-in):** mean held-out corr ≈ 0.8, mean gap ≈ 0.04, with the most kinetically distinct indicator showing the largest gap — the harness localises a weakness rather than rubber-stamping. The real go/no-go is the same table on the staged CASCADE files; **group by indicator AND cell class** (interneuron vs pyramidal kernels differ; CASCADE interneuron sets are DS#22–27).

---

## 6. The inference path

```math
Y\ \xrightarrow{\text{Stage 1 (scattering detector, raw movie)}}\ \text{cellness}
\ \xrightarrow{\text{peaks + nearest-centroid, activity-gated}}\ A_i
\ \xrightarrow{\ C_i=\frac{\sum_x A_i(x)Y(x,t)}{\sum_x A_i(x)}\ }\ C_i
\ \xrightarrow{\text{standardize\_trace}}\ \xrightarrow{\text{Stage 2}}\ \hat r_i,\ \hat\tau_i
```

A single per-recording process (`scripts/run_infer.py`) runs the whole chain, composing the spatial detector, the non-learned extraction helpers (`extract.py`), and the shared `detect_transients`. The spatial detector consumes the **raw movie directly** (no projection step). Traces are standardised by the same `standardize_trace` the temporal model trained on. Silent cells flow through and yield ~0 rate. The two trained stages drop in unchanged.

---

## 7. How we got here — the decision lineage

1. **Unification.** LoG (space) and Ricker (time) are the same $`\nabla^2 G`$ → one operator, two stages.
2. **Ground-truth split.** Spatial = in-domain ROIs; temporal = domain-mismatched public spikes → resample + LOIO first, synthetic fallback. Different datasets ⇒ separate training ⇒ separable design.
3. **Single generating function, not a filter bank** — to preserve the unification and the interpretable duration axis. Shape-parameterised in time (asymmetry angle) to fix the symmetric-Ricker bias; pure symmetric in space.
4. **Per-frame smoothing → covariance/scattering decomposition** → detector as a hierarchy of temporal moments of the band-passed movie; per-frame smoothing absorbed into the scale bank.
5. **All-cells annotation target** → the **structural (0th-moment) channel** becomes the primary detector; activity channels become features.
6. **ROIs drawn on the max projection** → the **max ($`L^\infty`$) channel** (top-$`k`$ robust max), the irreducible order statistic, matching the annotation substrate and surfacing sparse cells.
7. **Coherence channel** — in-framework faint-cell fix; conditional, opt-in.
8. **ΔF/F unified** into one `standardize_trace` across training and inference (noise-invariance by normalisation).

End state: both stages are the same operator — learnable derivative-of-Gaussian wavelet → nonlinearity → head — space pools temporal moments, time predicts rate.

---

## 8. Implementation, scale, and deployment

**Package** (`orcann/`):

| Module | Role |
|---|---|
| `spatial_log.py` | `ParametricLoG2d`; cellness targets, loss, instance extraction |
| `spatial_scatter.py` | `SpatialScatterDetector` — the moment-hierarchy detector (structural · max · variance · coherence) |
| `temporal_dog.py` | `ParametricDoGWavelet1d`, `TemporalRateModel`; `standardize_trace`, `detect_transients`, degradation helpers, geometric durations |
| `extract.py` | non-learned plumbing: movie I/O (`.npy`/`.tif`/`.nd2`), soft footprints, trace extraction |
| `figures.py` | per-ROI transient panels and the max-projection detection overlay |
| `train_loio.py` | dataset interface, CASCADE `.mat` loader, synthetic indicator bank, LOIO train/eval, `train_final` |
| `train_spatial.py` | annotated-recording loader (incl. ImageJ `RoiSet.zip`), synthetic bank, streaming patch training, metrics |

**Runners** (`scripts/`): `run_infer.py` (full per-recording pipeline), `run_transients.py` (trace-only), `run_train_spatial.py`, `run_train_temporal_loio.py`, `visualize_transients.py`, `check_annotation.py` — the training/inference runners carry a `--synthetic` self-test, and the suite in `tests/` exercises every module on synthetic data.

**HPC deployment** (`hpc/`): `config.sh` (all cluster-specific values), `setup_hpc.sh` (conda prefix env + CUDA-matched torch via `CUDA_BUILD`, default `cu121`; `PYTHONNOUSERSITE`), `make_workspace.sh`, and SGE jobs `check_gpu.sh`, `train_spatial.sh` (GPU), `train_temporal_loio.sh` (CPU), `infer_array.sh` (CPU array over a manifest).

**Scale/robustness hardening (verified):**
- **GPU-correct:** training detects CUDA and moves model + batches to device; kernels/accumulators follow parameter device.
- **Bounded memory:** recordings stream from disk one at a time; per-chunk LoG response ~101 MB at 512² (ample on an 80 GB A100). Patch training (default 128 px, 16 patches/recording, 256 energy frames) gives field coverage with bounded cost.
- **Fast max:** top-$`k`$ (0.28 s) replaces `torch.quantile` (5.9 s) at realistic size.
- **Checkpoint every epoch** so a queue-killed job is not a total loss.
- **`.nd2` reader** via the `nd2` package, using named axes (`sizes`) to form $`(T,H,W)`$; multi-channel/Z/point axes take index 0 with a warning.
- **Stable model persistence.** Models are saved as `{kind, config, state_dict}` (plain data + tensors, via `orcann.io`), not pickled objects, so a saved model loads after a package rename or a class refactor (a later constructor arg with a default is filled in). Whole-object `torch.save(model)` breaks on both.
- Inference runs on CPU (frees the GPU queue; loads models with `map_location="cpu"`).

**Per-recording outputs.** Each recording produces `run_<JOBID>/<recording>/` with two folders. `data/`: `spatial_footprints.npz`, `centroids.npy`, `temporal_traces.npy`, `rates.npy`, `events.npz` (long-format: `roi, time_s, duration_s, amplitude`), `max_projection.npy`, `meta.json`. `figures/`: one `roi_<i>.png` per ROI (trace + scalogram + rate) and `max_projection_detections.png` (centroids + footprint contours on the max image). The array files share the ROI axis; `events.npz` rows index into it. The cross-recording group analysis reads this contract and is a separate module. Annotation intake supports ImageJ/FIJI `RoiSet.zip` (rasterised with explicit $`(x,y)\to(\text{col},\text{row})`$ handling); `scripts/check_annotation.py` overlays it on the max projection as the orientation pre-flight.

**Status:** every module smoke, every runner self-test, all shell scripts, and a realistic-scale forward+backward pass clean; edge cases (flat/silent trace, zero detection) produce no NaN/crash. **No real *organoid* recording trained on or evaluated yet** (temporal LOIO ran on real CASCADE ground truth; spatial + organoid domain pending).

---

## 9. Honesty ledger (consolidated caveats)

- **2 Hz timing.** Sub-frame spike timing is not identifiable from 2 Hz input. Temporal target is a per-bin rate; discrete spikes are a downstream thresholding step.
- **Detection target settled, not free.** The detector matches the all-somata annotations via the structural channel; this was a correction (an activity-only detector would have under-counted silent cells). Active/silent is reported as a derived label.
- **Max is in, as an explicit channel** (not dropped). It is the irreducible order statistic, computed as $`\nabla^2 G`$ on a top-$`k`$ robust max projection, and matches the ROI-drawing substrate.
- **Temporal domain shift.** Public ground truth is GCaMP/OGB, 2-photon, in-vivo cortical — not Fluo-4 widefield organoid. LOIO (completed on CASCADE) is a **qualified pass**: transfer is strong for mainstream GECIs/dyes and GCaMP8 (often held-out ≈ within-train), weak for SST/VIP interneurons, and a clear fail for spinal cord (excluded as a distinct domain). What transfers is the transient *shape*; the *absolute rate scale* is CASCADE-calibrated and is the part that does not — so absolute event thresholds and counts are the fragile quantities on organoid data (see §4.6).
- **Scale normalisation is nominal, not effective.** The LoG bank carries the $`\sigma^2`$ factor but per-scale peak response is not equalised across scales (it grows with $`\sigma`$); size discrimination is done by the learned head, not by scale-selection. The duration read-out and detector both rely on the head/extent, not on a winning scale.
- **Duration ≠ τ.** Reported durations are wavelet characteristic timescales — an inflated, roughly constant multiple of the decay constant. Ordering/separation faithful; absolute seconds not.
- **Coherence is regime-dependent.** Helps in moderate noise, neutral elsewhere, never hurts; real-data value unmeasured.
- **ROI ↔ movie orientation is the top unverified risk.** ROIs were drawn on a max-projection summary; if its $`(Y,X)`$ orientation differs from the loader's (a transpose/flip/crop when the summary was made), targets are mislocated and training fails *silently* (low score, no error). Must be confirmed by overlaying ROI centroids on `movie.max(0)` before trusting any spatial number.
- **`.nd2` reader unverified on a real file** (no `.nd2` available during development); confirm axes and the orientation overlay on the first real recording.
- **Temporal validation in-domain is unavailable** with current data; checkable only against the old wavelet/OASIS method or future high-rate organoid acquisition.
- **What has and hasn't seen real data.** The temporal LOIO ran on real public CASCADE ground truth (§5.2 verdict above). The spatial stage and the organoid (Fluo-4) domain are still synthetic/pending: no annotated organoid recording has been trained on or evaluated, so spatial accuracy numbers and in-domain temporal numbers await the real harnesses.

---

## 10. References

- Marr D, Hildreth E (1980). Theory of edge detection. *Proc. R. Soc. Lond. B* — the $`\nabla^2 G`$ operator.
- Lindeberg T (1998). Feature detection with automatic scale selection. *IJCV* — scale-normalised LoG.
- Torrence C, Compo GP (1998). A Practical Guide to Wavelet Analysis. *BAMS* — scale↔Fourier-period.
- Friedrich J, Zhou P, Paninski L (2017). Fast online deconvolution of calcium imaging data. *PLoS Comp. Biol.* — OASIS.
- Rupprecht P, et al. (2021). A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging. *Nature Neuroscience* — CASCADE; resampling and noise-matching.
- Bruna J, Mallat S (2013). Invariant scattering convolution networks. *IEEE TPAMI* — wavelet→modulus→pool.
