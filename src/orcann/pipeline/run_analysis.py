"""Analysis stage: cross-recording group analysis over the activity outputs.

A thin config->call wrapper around the calcium pipeline's group analysis
(`orcann.analysis.orchestrate.run_analysis`). It reads every calcium-format
recording folder in the activity store this config names,
scores neuron quality, deduplicates ROIs, computes per-recording functional
summaries (spike rate, amplitude, pairwise correlation, synchrony, network
bursts, active fraction), and compares them across genotype and developmental
day, writing figures + tables to results/analysis/.

Unlike the per-recording stages this is a single aggregate run over one activity
store, so it is submitted directly (not as an array). Genotype and day are parsed
from each recording folder name by the analysis package's own name parsers
(D-line convention: line token starting with '3' is Control); the content tag the
identity ends in is a trailing field they do not read.
"""
import logging

logger = logging.getLogger(__name__)


def run(cfg, force=False):
    from orcann.analysis.orchestrate import run_analysis

    from orcann.pipeline import provenance as prov

    ap = cfg.analysis
    try:
        chain = prov.chain(cfg)
        src = prov.require_store(cfg.paths.activity, chain.activity, "activity")
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    out = prov.store_dir(cfg.paths.analysis, chain.analysis)
    have = prov.list_dir_records(src, prov.RUN_INFO, prov.STAGE_ACTIVITY)
    if not have:
        print(f"analysis: no recordings with an activity record in {src} "
              f"(run activity first)")
        return False

    # One acquisition must not enter the cohort twice. It can: a raw recording
    # replaced by a re-export or a re-acquisition under the same filename takes a
    # new identity, and the results made from the old contents stay where they
    # are. Both are honest records of different data, but the group statistics
    # would count one organoid as two, so the run stops rather than averaging
    # them.
    by_stem = {}
    for rec in have:
        by_stem.setdefault(prov.stem_of(rec), []).append(rec)
    twice = {k: v for k, v in by_stem.items() if len(v) > 1}
    if twice:
        for stem, recs in sorted(twice.items()):
            print(f"analysis: {stem} is present {len(recs)} times: "
                  f"{', '.join(sorted(recs))}")
        print(f"analysis: each of those is one acquisition under two different "
              f"contents. Delete the folders you do not want from {src}.")
        return False

    print(f"analysis: {len(have)} recording(s)  {src} -> {out}")
    results = run_analysis(
        results_dir=src,
        output_dir=out,
        frame_rate=cfg.imaging.frame_rate,
        motion_max_threshold=ap.motion_max_threshold,
        motion_residual_threshold=ap.motion_residual_threshold,
        drift_threshold=ap.drift_threshold,
        inactive_file=ap.inactive_file,
        min_roi_distance=ap.min_roi_distance,
        roi_peak_figures=ap.roi_peak_figures,
        mutant_label=ap.mutant_label,
        # analysis.dev gates orcann.dev; the deconvolution settings ride along
        # because the dev transient-decay measurement only means anything on the
        # robust detector's own onset threshold (it refuses to run otherwise).
        dev=ap.dev,
        deconv_method=cfg.deconvolution.method,
        robust_k_onset=cfg.deconvolution.robust_k_onset,
    )
    if not results:
        print("analysis: no datasets loaded - check that activity ran and folder "
              "names carry genotype/day (see dataset_features.csv)")
        return False

    print(f"analysis outputs -> {out}/  ({results.get('n_datasets', '?')} datasets)")

    # A stage that failed leaves its section out of analysis_results.json. Say so
    # here and report it through the exit code, so a scheduler does not record a
    # partial run as a clean one.
    report = results.get("stage_report", {})
    failed = report.get("failed_stages") or []
    if failed:
        print(f"analysis: {len(failed)} stage(s) FAILED: {', '.join(failed)}")
        print(f"analysis: {out}/data/analysis_results.json is INCOMPLETE "
              f"(see stage_report in that file for tracebacks)")
        return False
    return True
