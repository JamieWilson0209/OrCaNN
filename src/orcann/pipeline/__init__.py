"""Shared pipeline: the seam between the two stages, plus stage-agnostic plumbing.

Kept intentionally import-light (no eager submodule imports) so that the
caiman-only motion-correction path can pull ``pipeline.extraction`` and
``pipeline.motion_correction`` without dragging in torch. Import the piece you
need directly:

    from orcann.pipeline import inference                 # the seam (segment->extract->detect)
    from orcann.pipeline.model_io import load_model, save_model
    from orcann.pipeline.extraction import _load_movie
    from orcann.pipeline.figures import roi_figure, max_projection_figure
    from orcann.pipeline.motion_correction import correct_motion   # caiman env only
"""
