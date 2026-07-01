"""Spatial stage: detect *where* the cells are.

A single ∇²G operator (`laplacian`) feeds a scattering-energy front-end
(`scattering`), which feeds the trained `segmenter` (U-Net -> per-pixel soma
probability). `training` fits the segmenter; `annotations` loads manual ROIs.
"""
from orcann.spatial.detection.laplacian import (
    ParametricLoG2d, extract_instances, centroids_from_masks,
)
from orcann.spatial.detection.scattering import SpatialScatterDetector
from orcann.spatial.detection.segmenter import (
    SpatialSegmenter, focal_dice_loss, predict_prob, segment_instances,
)
from orcann.spatial.training import (
    SegRecording, load_seg_recording, soft_iou, best_iou,
    train_segmenter, synthetic_sources,
)
from orcann.spatial.training.annotations import _load_annotation

__all__ = [
    "ParametricLoG2d", "extract_instances", "centroids_from_masks",
    "SpatialScatterDetector",
    "SpatialSegmenter", "focal_dice_loss", "predict_prob", "segment_instances",
    "SegRecording", "load_seg_recording", "soft_iou", "best_iou",
    "train_segmenter", "synthetic_sources", "_load_annotation",
]
