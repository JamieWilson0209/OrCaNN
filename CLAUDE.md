# OrCaNN

Calcium-imaging pipeline for neural organoids, run on the university HPC cluster.
Stages run in order: `motion_correct` → `segment` → `infer` → `activity` →
`analysis`. Entry point is `orcann <stage>` (`src/orcann/pipeline/cli.py`);
`config.yaml` holds every knob.

## Comments

A comment is read by someone who has never seen the commit that prompted it.
Write what the code does now, not what changed.

- **No history.** Not "this used to fall back to 2.0", not "since 10df777", not
  "(v2.0)". Those name a version the reader cannot look at, and they rot the
  moment the code moves again.
- **No stale claims.** A comment describing behaviour the code no longer has is
  worse than no comment, because it is believed. If you change a line, re-read
  the comment above it.
- **Line by line, not blocks.** Explain the line in front of you. Long blocks
  belong in a module docstring, and only when the rationale is genuinely
  load-bearing — `activity/deconvolution.py`'s transient-duration gate is the
  standard to aim at.
- **Say why, not what.** `# clip negatives` restates the code. `# anti-correlated
  neurons do not contribute to synchrony` says why the clip is there.
- **Record open questions honestly.** Where a method has not been validated, the
  comment should say so rather than assert a rationale.

Before committing:

```bash
python scripts/check_comments.py          # exits 1 on a violation
```

It catches the mechanical cases — history phrases, commit hashes, version
stamps. Stale prose it cannot see, so read what you touched. Where the history
genuinely *is* the message — telling someone a config value they still name was
removed and what to use instead — mark it `history-ok` in the comment.

## Conventions this pipeline holds to

**Fail loudly; never substitute a plausible value.** A wrong number that looks
right is worse than a crash. There are no fallback detectors, no default
amplitude method, no assumed frame rate. A recording that cannot be read is
refused by name, with what to do about it.

**`NaN` means "not measured".** It is not zero and must not be imputed. It
reaches the CSV as an empty field and excludes a recording from anything needing
complete rows, which says so in the log.

**Every stage record goes through `orcann/run_info.py`.** One writer, one
reader, one schema. Records open with `schema_version` then `stage`; the reader
refuses anything older rather than guessing what its keys meant. There are no
raw `json.dump` calls outside that module — keep it that way, or NaN handling
and version stamping get forgotten at the new call site.

**`config.yaml` is the source of truth.** The frame rate, the baseline method
and the thresholds come from config and nowhere else; per-recording metadata is
provenance, not an input.

**Prefer removal over patching.** A silently broken path is deleted, not guarded.
Several detectors, baseline methods and quality gates have gone this way.

## Environment

The `orcann` conda env lacks `pandas` and `sklearn`, so `analysis/stats.py`,
`analysis/orchestrate.py` and `figures/overview.py` will not import under it.
Use `orcann-app`, which has the full set:

```bash
/Users/Jamie/miniforge3/envs/orcann-app/bin/python -c "import orcann"
```

## Tests

There is no test suite in this repository. Verify a change by running the code
against fixtures you build, and say plainly in the commit what you ran. Work in
progress whose method is still being decided lives behind `analysis.dev` in
`src/orcann/dev/`; local notes and port plans go in `dev/`, which is gitignored.
