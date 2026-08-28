"""A record of which pipeline stages ran, and what happened to each.

Stages that produce optional outputs — figures, an extra statistical test, a dev
experiment — are isolated so one failure cannot cost the run its remaining
results.  Isolation alone would hide the failure: a stage that raises leaves its
key absent from ``analysis_results.json``, which is indistinguishable from a
stage that ran and found nothing to report.

Every stage runs through :meth:`StageReport.run`, which records the outcome
either way.  The report is written into the results file, so the artefact
carries its own provenance, and :attr:`StageReport.ok` gives the caller an exit
code.

Usage::

    report = StageReport()
    stats = report.run("statistical_tests", run_statistical_tests, datasets, out)
    ...
    results["stage_report"] = report.to_dict()
    return results, report
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["StageOutcome", "StageReport"]


@dataclass
class StageOutcome:
    """What happened to one stage."""

    name: str
    status: str                              # "ok" | "failed" | "skipped"
    detail: Optional[str] = None             # exception repr, or why it was skipped
    traceback_text: Optional[str] = None     # full traceback, failures only

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status}
        if self.detail:
            d["detail"] = self.detail
        if self.traceback_text:
            d["traceback"] = self.traceback_text
        return d


@dataclass
class StageReport:
    """Runs named stages, isolating failures without hiding them."""

    outcomes: List[StageOutcome] = field(default_factory=list)

    # ---------------------------------------------------------------- running
    def run(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run ``fn`` and record the outcome.

        Returns whatever ``fn`` returned, or ``None`` if it raised.  The
        exception is logged and recorded; it is not re-raised, so a failing
        stage costs only its own output.
        """
        try:
            value = fn(*args, **kwargs)
        except Exception as e:                       # noqa: BLE001 - recorded, not hidden
            logger.error(f"{name} failed: {e}")
            tb = traceback.format_exc()
            logger.error(tb)
            self.outcomes.append(StageOutcome(name, "failed", repr(e), tb))
            return None
        self.outcomes.append(StageOutcome(name, "ok"))
        return value

    def skip(self, name: str, reason: str) -> None:
        """Record a stage that was deliberately not run."""
        self.outcomes.append(StageOutcome(name, "skipped", reason))

    # ---------------------------------------------------------------- reading
    @property
    def failed(self) -> List[str]:
        return [o.name for o in self.outcomes if o.status == "failed"]

    @property
    def ok(self) -> bool:
        """True when no stage failed.  Skipped stages are not failures."""
        return not self.failed

    def to_dict(self) -> Dict[str, Any]:
        """The block to embed in the results JSON."""
        return {
            "ok": self.ok,
            "failed_stages": self.failed,
            "stages": {o.name: o.to_dict() for o in self.outcomes},
        }

    def log_summary(self) -> None:
        n_ok = sum(1 for o in self.outcomes if o.status == "ok")
        n_skip = sum(1 for o in self.outcomes if o.status == "skipped")
        if self.ok:
            logger.info(f"All stages completed ({n_ok} ok, {n_skip} skipped)")
        else:
            logger.error(
                f"{len(self.failed)} stage(s) FAILED: {', '.join(self.failed)} "
                f"({n_ok} ok, {n_skip} skipped) — results are incomplete"
            )
