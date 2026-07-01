"""Spatial detection: ∇²G operator -> scattering energy -> trained segmenter."""
from orcann.spatial.detection.laplacian import (
    ParametricLoG2d, extract_instances, centroids_from_masks)
from orcann.spatial.detection.scattering import SpatialScatterDetector
from orcann.spatial.detection.segmenter import (
    SpatialSegmenter, focal_dice_loss, predict_prob, segment_instances)
