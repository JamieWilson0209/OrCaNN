"""Group-level analysis pipeline: loading, metric computation, statistics
orchestration.

Module layout:
  - loading       — DatasetMetrics, load_dataset_metrics, name extractors
  - metrics       — per-neuron / per-recording metric helpers
  - stats         — statistical test orchestration + figure dispatch
  - orchestrate   — top-level run_analysis entry point
"""
