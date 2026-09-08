#!/usr/bin/env python
"""Exercise the intake policy in ``pipeline/extraction.py``; exit 1 on a violation.

Every case here is a file the pipeline might be handed. The point is not that the
loader parses them — it is that it refuses the ones it cannot honestly read, and says
so by name rather than reading them wrongly and reporting success.

    python scripts/check_intake.py                 # synthetic cases only
    python scripts/check_intake.py --real DIR      # also round-trip real recordings

With ``--real``, one file of each extension found in DIR is read twice, once whole and
once by frame index, and the two are compared. That is the check that matters after any
change to the readers: the frames a caller asks for must be the frames it would have
got from the whole array.
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from orcann.pipeline.cli import _SIDECAR_SUFFIXES                    # noqa: E402
from orcann.pipeline.extraction import IntakeError, open_recording   # noqa: E402

FAILURES = []


def expect_ok(label, fn):
    try:
        fn()
        print(f"  ok       {label}")
    except Exception as e:
        FAILURES.append(label)
        print(f"  FAILED   {label}: {type(e).__name__}: {e}")


def expect_refused(label, fn, must_mention=None):
    try:
        fn()
    except IntakeError as e:
        if must_mention and must_mention not in str(e):
            FAILURES.append(label)
            print(f"  FAILED   {label}: refused, but the message omits {must_mention!r}")
        else:
            print(f"  ok       {label}")
        return
    except Exception as e:
        FAILURES.append(label)
        print(f"  FAILED   {label}: raised {type(e).__name__}, not IntakeError")
        return
    FAILURES.append(label)
    print(f"  FAILED   {label}: accepted, and should not have been")


def synthetic(tmp):
    a = np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8)
    p = lambda n: os.path.join(tmp, n)                              # noqa: E731

    np.save(p("plain.npy"), a)
    tifffile.imwrite(p("q.tif"), a)
    tifffile.imwrite(p("t.tif"), a, imagej=True,
                     metadata={"axes": "TYX", "finterval": 0.5})
    tifffile.imwrite(p("z.tif"), a, imagej=True, metadata={"axes": "ZYX"})
    tifffile.imwrite(p("c.tif"), a, imagej=True, metadata={"axes": "CYX"})
    np.save(p("flat.npy"), a[0])
    dirty = a.astype(np.float32).copy()
    dirty[2, 3, 3] = np.nan
    np.save(p("nan.npy"), dirty)

    print("accepted, with the right provenance")
    expect_ok("plain .npy reads as assumed TYX",
              lambda: _assert(open_recording(p("plain.npy")).meta.axes_source == "assumed"))
    expect_ok("undeclared QYX tif reads as assumed",
              lambda: _assert(open_recording(p("q.tif")).meta.axes_source == "assumed"))
    expect_ok("declared TYX tif reads as declared",
              lambda: _assert(open_recording(p("t.tif")).meta.axes_source == "declared"))
    expect_ok("declared tif carries its frame interval",
              lambda: _assert(open_recording(p("t.tif")).meta.frame_interval_s == 0.5))
    expect_ok("a file that states no interval reports None, not a default",
              lambda: _assert(open_recording(p("q.tif")).meta.frame_interval_s is None))

    print("\nrefused by name")
    expect_refused("a z-stack is not a time series",
                   lambda: open_recording(p("z.tif")), must_mention="Z")
    expect_refused("a channel stack is not a time series",
                   lambda: open_recording(p("c.tif")), must_mention="C")
    expect_refused("a 2-D image is not a movie", lambda: open_recording(p("flat.npy")))
    expect_refused("an unsupported extension", lambda: open_recording(p("x.foo")))
    expect_refused("a frame index outside the recording",
                   lambda: open_recording(p("plain.npy")).frames([99]))

    print("\nfiniteness is the caller's policy, supplied on request")
    expect_ok("array() loads a dirty movie when not asked to check",
              lambda: _assert(np.isnan(open_recording(p("nan.npy")).array()).any()))
    expect_refused("array(require_finite=True) refuses it",
                   lambda: open_recording(p("nan.npy")).array(require_finite=True))
    expect_ok("a clean subset of a dirty movie passes",
              lambda: open_recording(p("nan.npy")).frames([0, 1], require_finite=True))

    print("\nthe lazy reads agree with the whole array")
    rec = open_recording(p("plain.npy"))
    full = rec.array()
    expect_ok("frames(idx) == array()[idx]",
              lambda: _assert(np.array_equal(rec.frames([0, 3, 5]), full[[0, 3, 5]])))
    expect_ok("stream() concatenates back to array()",
              lambda: _assert(np.array_equal(np.concatenate(list(rec.stream(2))), full)))
    rec.close()

    print("\nhandles are released")
    expect_ok("close() is idempotent", lambda: _close_twice(p("plain.npy")))
    expect_ok("200 open/close cycles do not leak descriptors",
              lambda: _no_fd_leak(p("q.tif")))


def real(dirpath):
    print(f"\nreal recordings in {dirpath}")
    seen = set()
    for ext in (".nd2", ".tif", ".tiff", ".npy"):
        for f in sorted(glob.glob(os.path.join(dirpath, "*" + ext))):
            # the same sidecars cli.list_recordings drops, so the checker sees the
            # files the pipeline would actually be handed
            if os.path.basename(f).endswith(_SIDECAR_SUFFIXES):
                continue
            if ext in seen:
                break
            seen.add(ext)
            try:
                rec = open_recording(f)
            except IntakeError as e:
                FAILURES.append(os.path.basename(f))
                print(f"  FAILED   {os.path.basename(f)[:44]:46} refused: {e}")
                continue
            with rec:
                m = rec.meta
                idx = [0, m.n_frames // 2, m.n_frames - 1]
                sub = rec.frames(idx)
                whole = rec.array()
                same = np.array_equal(sub, whole[idx])
                label = (f"{os.path.basename(f)[:38]:40} {m.n_frames}x{m.height}x{m.width} "
                         f"axes={m.axes_source} interval={m.frame_interval_s}")
                if same:
                    print(f"  ok       {label}")
                else:
                    FAILURES.append(label)
                    print(f"  FAILED   {label}: frames(idx) != array()[idx]")
            del whole, sub


def _assert(cond):
    if not cond:
        raise AssertionError("expectation not met")


def _close_twice(path):
    rec = open_recording(path)
    rec.close()
    rec.close()


def _no_fd_leak(path):
    import gc
    before = len(os.listdir("/dev/fd"))
    for _ in range(200):
        with open_recording(path):
            pass
    gc.collect()
    after = len(os.listdir("/dev/fd"))
    _assert(after <= before + 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", metavar="DIR",
                    help="also round-trip one real recording per extension found here")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        synthetic(tmp)
    if a.real:
        real(os.path.expanduser(a.real))
    print()
    if FAILURES:
        print(f"{len(FAILURES)} intake check(s) failed.")
        return 1
    print("intake policy holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
