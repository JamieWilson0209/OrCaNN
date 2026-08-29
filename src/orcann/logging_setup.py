"""Where the pipeline's log output goes, configured once at the entry point.

Modules throughout the package call ``logging.getLogger(__name__)`` and log
freely, which does nothing on its own: with no handler installed, Python's
last-resort handler prints WARNING and above to stderr and discards everything
below it. The per-recording quality-gating table, the count of duplicate
detections removed, the reason a recording was refused — all of it is INFO, and
all of it is what a reader needs when a run produces a number they did not
expect. This module installs the handler that lets it out.

Only ``pipeline/cli.py`` calls :func:`configure`. A library module that
configures logging steals the decision from whoever imported it, so nothing
under ``analysis/``, ``activity/`` or ``figures/`` may do so.

**Output goes to stdout, deliberately.** The cluster's job scripts direct stdout
and stderr to separate files (``#$ -o logs/`` and ``#$ -e logs/``), and the
stages also ``print`` their per-recording progress. Sending the log to stderr
would split one run's narrative across two files, interleaved with neither half
readable in order. On stdout it sits with the progress lines it explains, and
stderr is left to carry tracebacks alone.

**INFO lines are printed bare.** The messages already carry their own
indentation, and a level prefix on every line would fight it and bury the few
lines that matter. WARNING and above are prefixed, so the exclusions and
refusals stand out in a log that is mostly narrative.
"""

from __future__ import annotations

import logging
import sys

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

DEFAULT_LEVEL = "INFO"


class _StageFormatter(logging.Formatter):
    """Bare text for the narrative, a level prefix for anything that interrupts it."""

    def __init__(self) -> None:
        super().__init__("%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)          # includes traceback when exc_info is set
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {text}"
        return text


def configure(level: str = DEFAULT_LEVEL, stream=None) -> None:
    """Install the pipeline's log handler on the root logger.

    Replaces any handler already installed, so calling twice in one process
    leaves one handler rather than doubling every line.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(_StageFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    # matplotlib and PIL log per-glyph and per-chunk at DEBUG, which buries the
    # pipeline's own output the moment anyone asks for it.
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
