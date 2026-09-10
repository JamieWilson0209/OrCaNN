"""Training-progress frames: the model's current soma probability drawn over one
training and one held-out recording, written at a tapering epoch schedule so the
series assembles into a movie afterwards. See docs/spatial/training.md.

The subject is fixed for the whole run and rendered in ``eval()``, so successive
frames differ by what the model learned and by nothing else: in ``train()`` the
energy front-end draws a fresh random frame subset per forward, which would move
the picture on its own.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# The gamma stretch and per-pixel alpha of the infer stage's QC image
# (figures.prob_overlay_figure), but one flat hue rather than that figure's magma:
# magma runs to white at high probability and the somata are already white, so a
# confident detection read as nothing but a slightly warmer bright blob. Cyan is
# far enough from cream to separate the prediction from the cell under it, and the
# alpha stays low so the cell is still the thing you see.
GAMMA = 0.45
PROB_ALPHA = 0.5
PROB_RGB = (0.10, 0.90, 1.00)      # cyan: the model's soma probability
EDGE_RGB = (0.85, 0.25, 0.35)      # red: the border the LoG bank cannot read


def progress_epochs(epochs: int, first: int = 10, taper: float = 0.15) -> List[int]:
    """Epochs to render: every one below ``first``, then gaps growing by
    ``taper``. Dense early because that is where the fit moves, sparse later
    because it does not, and always the last epoch so the series ends on the
    model that gets saved."""
    out: List[int] = []
    nxt = 0
    for ep in range(epochs):
        if ep < first or ep >= nxt:
            out.append(ep)
            nxt = ep + max(1, int(ep * taper))
    if out and out[-1] != epochs - 1:
        out.append(epochs - 1)
    return out


def prepare_subject(rec, n_energy_frames: Optional[int]
                    ) -> Tuple[np.ndarray, np.ndarray, str]:
    """(frames, label, name) for one recording, holding only the frames the model
    will actually pool. ``eval()`` reads an even stride of ``n_energy_frames``, so
    taking that stride here and passing it whole is the same forward pass on a
    fraction of the memory -- a 601-frame recording is 630 MB, its 64-frame
    stride 67 MB, and two subjects are held for the length of the run."""
    movie = rec.movie
    T = movie.shape[0]
    if n_energy_frames and T > n_energy_frames:
        idx = np.linspace(0, T - 1, n_energy_frames).round().astype(int)
        movie = movie[idx]
    return (np.ascontiguousarray(movie), (rec.label > 0).astype(bool),
            os.path.basename(rec.rid))


def _panel(frames: np.ndarray, prob: np.ndarray, half: int) -> np.ndarray:
    """(H, W, 3) uint8: the recording's max projection in grey, under the model's
    current soma probability in cyan at ``PROB_ALPHA``, with the border the bank
    reads against zero padding ringed in red."""
    base = frames.max(axis=0)
    lo, hi = np.percentile(base, (1.0, 99.5))
    grey = np.clip((base - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.repeat(grey[..., None], 3, axis=2)

    alpha = (np.clip(prob, 0.0, 1.0) ** GAMMA * PROB_ALPHA)[..., None]
    rgb = rgb * (1.0 - alpha) + np.asarray(PROB_RGB) * alpha

    H, W = grey.shape
    if half and 2 * half < min(H, W):
        for ys, xs in ((slice(half, half + 1), slice(half, W - half)),
                       (slice(H - half - 1, H - half), slice(half, W - half)),
                       (slice(half, H - half), slice(half, half + 1)),
                       (slice(half, H - half), slice(W - half - 1, W - half))):
            rgb[ys, xs] = EDGE_RGB
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def write_progress_frame(model, subjects: Sequence[Tuple[np.ndarray, np.ndarray, str]],
                         titles: Sequence[str], epoch: int, path: str,
                         device: Optional[torch.device] = None) -> List[float]:
    """Render one frame of the series and return each panel's soft IoU at 0.3.

    Restores ``train()`` before returning: the caller is mid-fit.
    """
    from orcann.spatial.training.training import soft_iou
    device = device or next(model.parameters()).device
    half = model.front.log._half

    was_training = model.training
    model.eval()
    panels, ious = [], []
    with torch.no_grad():
        for frames, truth, _ in subjects:
            x = torch.from_numpy(frames).to(device)[None]
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            panels.append(_panel(frames, prob, half))
            interior = (slice(half, prob.shape[0] - half),
                        slice(half, prob.shape[1] - half))
            ious.append(soft_iou(prob[interior], truth[interior], 0.3))
    if was_training:
        model.train()

    H, W = panels[0].shape[:2]
    strip = 46
    fig = plt.figure(figsize=((W * len(panels)) / 100.0, (H + strip) / 100.0), dpi=100)
    fig.patch.set_facecolor("#0d0d10")
    for i, (img, title) in enumerate(zip(panels, titles)):
        ax = fig.add_axes([i / len(panels), strip / (H + strip),
                           1.0 / len(panels), H / (H + strip)])
        ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
        ax.text(0.015, 0.982, f"{title}  ·  {subjects[i][2]}", color="w",
                fontsize=9, va="top", ha="left", transform=ax.transAxes)
        ax.text(0.985, 0.982, f"IoU@0.3 {ious[i]:.3f}", color="w", fontsize=9,
                va="top", ha="right", transform=ax.transAxes)
    fig.text(0.5, (strip * 0.5) / (H + strip), f"epoch {epoch}", color="w",
             fontsize=13, ha="center", va="center")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ious
