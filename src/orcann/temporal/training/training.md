# Temporal training

`orcann/temporal/training.py`. Fits the rate head on public ground-truth spike
trains and evaluates it leave-one-indicator-out (LOIO) before any organoid data is
involved.

## The question LOIO answers

The deployment plan trains the rate head on public ground truth (a *set* of
calcium indicators) and applies it to organoid Fluo-4. LOIO tests whether that
transfer works at all: train on all-but-one indicator, test on the held-out one,
at the 2 Hz target rate with noise matched to the real recordings. If transfer is
weak here the public route fails early and the fallback is a synthetic forward
model; if it holds, the route is justified.

## Data interface

Everything runs through one type, `GroundTruthNeuron` (a ΔF/F trace, its frame
rate, true spike times, and an indicator label). Two producers implement it:

- `load_cascade_mat` reads real CASCADE `.mat` files (`CAttached` structs with
  `fluo_time`, `fluo_mean`, `events_AP`). It handles both v7 (`scipy.io.loadmat`)
  and v7.3/HDF5 (`h5py`) files, flattens whatever container shape the dataset
  uses, auto-detects the `events_AP` time unit against the fluorescence span (it
  is not crisply documented and varies), and skips malformed entries so one odd
  file cannot abort a run.
- `synthetic_indicator_bank` builds several indicators with distinct rise/decay
  kinetics, so the harness is fully testable without the multi-GB download. Swap
  in `load_cascade_mat` on the cluster.

## Preprocessing recipe (`preprocess`)

Each neuron becomes a `(trace, rate_target)` pair at the target rate:

1. **Anti-aliased resample** to the target rate (polyphase, not decimation, which
   would alias fast structure into the signal).
2. **Noise match**: add white noise up to a measured target SD. A binned-down
   high-rate trace is cleaner than a native low-rate one, so this restores a
   realistic SNR; it runs *before* standardisation, giving SNR augmentation.
3. **Standardise** with the same `standardize_trace` used at inference, so train
   and test inputs match.

The supervised target is a Gaussian-smoothed per-bin spike rate, never sub-frame
spike times. Silent neurons keep an all-zero target.

## LOIO and the deployable model

`run_loio` runs the fold loop and reports, per held-out indicator, within-train
and held-out correlation, the transfer gap, and the learned asymmetry θ.
Correlation is measured only over windows that actually contain firing, since it
is undefined for an all-zero target; silent windows still train the model through
the MSE term of the loss. `train_final` fits one model on all supplied neurons
(no held-out split) and is the model used for inference.
