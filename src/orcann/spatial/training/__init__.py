"""Spatial training: fit the segmenter; load manual ROI annotations."""
from orcann.spatial.training.training import (
    SegRecording, load_seg_recording, soft_iou, best_iou,
    train_segmenter, synthetic_sources)
from orcann.spatial.training.annotations import _load_annotation
