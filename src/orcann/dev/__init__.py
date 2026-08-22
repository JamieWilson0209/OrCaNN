"""Development-only analyses: gated behind ``analysis.dev`` in the config.

Nothing here is part of the supported pipeline. These modules are for
investigations whose method is still being decided — they may make stronger
assumptions than the main analysis does, may only apply to one deconvolution
method, and may change or disappear without a deprecation path. Code graduating
out of here moves into ``orcann.analysis`` / ``orcann.figures`` and loses the
flag.

The flag is off by default, so a normal ``orcann analysis`` run never touches
this package.
"""

from .transient_decay import run_transient_decay          # noqa: F401

__all__ = ["run_transient_decay"]
