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
(D-line convention: line token starting with '3' is Control).

Its key carries the cohort — every recording it read, with the digest of that
recording's activity record — because an aggregate has no recording of its own
and would otherwise read as current after a recording changed underneath it.
"""
import logging
import os

logger = logging.getLogger(__name__)


def run(cfg, force=False):
    from orcann.analysis.orchestrate import run_analysis

    from orcann.pipeline import provenance as prov

    ap = cfg.analysis
    try:
        chain = prov.chain(cfg)
        keys = prov.stage_keys(cfg, chain)
        src = prov.require_store(cfg.paths.activity, prov.STAGE_ACTIVITY,
                                 chain.run_key, keys[prov.STAGE_ACTIVITY],
                                 prov.RUN_INFO)
    except prov.ProvenanceError as e:
        raise SystemExit(str(e))
    out = prov.output_store(cfg.paths.analysis, prov.STAGE_ANALYSIS,
                            chain.run_key, keys[prov.STAGE_ANALYSIS], None)
    have = prov.list_dir_records(src, prov.RUN_INFO, prov.STAGE_ACTIVITY)
    if not have:
        print(f"analysis: no recordings with an activity record in {src} "
              f"(run activity first)")
        return False

    # The cohort as read, not as configured: which recordings arrived and what
    # each of them was made from. A recording that changed, arrived or dropped
    # out is then a changed key, and the group figures are rebuilt.
    cohort = sorted(
        [rec, prov.record_of(src, rec, prov.RUN_INFO,
                             prov.STAGE_ACTIVITY).get("key_digest")]
        for rec in have)
    key = dict(keys[prov.STAGE_ANALYSIS], cohort=cohort)
    record = os.path.join(out, prov.RUN_INFO)
    ok, why = prov.is_current(record, prov.STAGE_ANALYSIS, key)
    if ok and not force:
        print(f"analysis: current for {len(have)} recording(s) in {out}; "
              f"nothing to do")
        return True
    if not ok and why != "no record":
        print(f"analysis: rebuilding ({why})")

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

    # The record last, and only on a complete run: it is what marks this store
    # done, so a rebuild that lost a stage is redone rather than trusted.
    from orcann.run_info import write_record
    write_record(record, prov.STAGE_ANALYSIS, key, chain.run_key,
                 {"n_datasets": results.get("n_datasets"),
                  "n_recordings": len(have),
                  "upstream_store": src})
    return True
