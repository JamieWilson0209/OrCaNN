#!/bin/bash
# =============================================================================
# One-time environment setup on an SGE / Grid Engine HPC cluster. Builds the
# conda env(s) OrCaNN needs. Run from anywhere on a login node (installs are fine
# there; never train on the login node):
#
#     bash hpc/setup.sh            # main torch + orcann env (always needed)
#     bash hpc/setup.sh all        # also build the caiman env (motion correction)
#     bash hpc/setup.sh caiman     # only the caiman env
#
# Targets:
#   main    ENV_PREFIX  : python 3.11 + CUDA torch + `pip install -e .` (orcann).
#                         Used by every stage EXCEPT motion correction.
#   caiman  CAIMAN_ENV  : python 3.11 + caiman (conda-forge) + nd2 + tifffile +
#                         orcann (editable, --no-deps, for the `orcann` command).
#                         Used ONLY by `orcann motion_correction`. Kept separate
#                         so caiman's large pinned stack never constrains the
#                         torch env. Optional: skip it unless you motion-correct
#                         raw recordings.
#   all     both of the above.
#
# Env locations, module names and the torch CUDA build all come from
# hpc/config.sh. Each env is created only if its directory does not already
# exist, so re-running is safe.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"   # the repo root (this is the workspace)
source "${HERE}/config.sh"

TARGET="${1:-main}"
case "${TARGET}" in
    main|caiman|all) ;;
    *) echo "usage: bash hpc/setup.sh [main|caiman|all]   (default: main)" >&2
       exit 2 ;;
esac

# Caches live on scratch (see config.sh). Create them before conda runs: if the
# conda package cache falls back to a quota-limited home directory, the solve
# dies while writing repodata with a long, misleading "unexpected error" report.
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_DIRS}" "${PIP_CACHE_DIR}"

# Preflight: a full home directory breaks conda in ways its own error message
# does not explain. Fail here with something actionable instead.
if ! touch "${HOME}/.orcann_quota_probe" 2>/dev/null; then
    echo "ERROR: cannot write to \$HOME (${HOME}). Your home quota is likely full." >&2
    echo "  Check usage:  quota -s ; du -sh ~/.conda ~/.cache" >&2
    echo "  Reclaim:      conda clean --all --yes" >&2
    exit 1
fi
rm -f "${HOME}/.orcann_quota_probe"

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"

# Some conda activation hooks (e.g. caiman's Intel-MPI mpivars.activate.sh)
# reference unbound shell variables; `set -u` would abort the script on them.
# Activate with nounset off, exactly as the job scripts do.
activate() {
    set +u
    source activate "$1"
    set -u
}

setup_main() {
    echo "=== main env: ${ENV_PREFIX} ==="
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    if [ ! -d "${ENV_PREFIX}" ]; then
        conda create --yes --prefix "${ENV_PREFIX}" python=3.11
    fi
    activate "${ENV_PREFIX}"
    # CUDA torch build from config.sh (CUDA_BUILD). cu121 is a safe default for
    # recent datacenter GPUs; verify with jobs/check_gpu.sh and bump if the node
    # reports a newer CUDA. PYTHONNOUSERSITE (from config) stops ~/.local leaking
    # into the env.
    python -m pip install --upgrade pip
    python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_BUILD}"
    python -m pip install -e "${REPO_ROOT}"
    # cuda False is EXPECTED on a login node (no GPU is attached there). Verify
    # the CUDA build on a GPU node with jobs/check_gpu.sh.
    python -c "import torch, orcann; print('  main env ready | torch', torch.__version__, '| cuda visible here:', torch.cuda.is_available(), '(False is normal on a login node)')"
}

setup_caiman() {
    echo "=== caiman env: ${CAIMAN_ENV} ==="
    # mamba resolves the caiman stack much faster than conda; use it if present.
    local solver
    solver="$(command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)"
    mkdir -p "$(dirname "${CAIMAN_ENV}")"
    if [ ! -d "${CAIMAN_ENV}" ]; then
        # one solve: conda-forge resolves caiman and a compatible stack. opencv is
        # pinned below 5 (opencv 4.x is long-tested with caiman) and libjxl is
        # named EXPLICITLY: conda-forge's opencv (4.x and 5.x alike) links
        # libjxl.so.0.11, but the library is not always materialised as a
        # transitive dependency, which surfaces later as an ImportError when
        # caiman imports cv2. Listing libjxl forces conda to install the file.
        "${solver}" create --yes --prefix "${CAIMAN_ENV}" -c conda-forge caiman "opencv<5" libjxl "python=3.11"
    fi
    activate "${CAIMAN_ENV}"
            # opencv is pulled in by caiman (caiman/base/movies.py imports cv2) and links
    # libjxl.so.0.11. conda-forge does not always materialise libjxl as a
    # transitive dependency, so caiman's first cv2 import fails with:
    #   ImportError: libjxl.so.0.11: cannot open shared object file
    # Naming opencv (<5, the caiman-tested line) AND libjxl explicitly forces both
    # onto disk at a matching version. Runs every time (idempotent), so it repairs
    # an env created before this fix, not just fresh creates.
    "${solver}" install --yes --prefix "${CAIMAN_ENV}" -c conda-forge "opencv<5" libjxl
    if ! python -c "import cv2" 2>/dev/null; then
        echo "ERROR: cv2 still fails to import in ${CAIMAN_ENV}." >&2
        echo "  libjxl may be registered but not extracted; force it:" >&2
        echo "  ${solver} install --prefix ${CAIMAN_ENV} -c conda-forge --force-reinstall 'libjxl=0.11'" >&2
        echo "  then re-run setup." >&2
        exit 1
    fi
    # The motion-correction path is deliberately torch-free (pipeline/__init__ is
    # import-light; caiman is lazy-imported), so we can give the caiman env the
    # `orcann` command WITHOUT pulling the torch/segmentation stack. This env runs
    # TWO stages: motion_correction and activity (OASIS deconvolution is CaImAn's
    # constrained_foopsi, plus the dF/F0 baseline and the HTML gallery). Install:
    #   - nd2 (Nikon reader) + tifffile: the movie-I/O extras on top of caiman.
    #   - pyyaml: configLoader needs it to read the config (caiman does not
    #     guarantee it).
    #   - pillow: base64 PNG encoding for the activity stage's HTML gallery.
    #   - orcann itself, editable and --no-deps: registers the `orcann` console
    #     script (so every job script invokes stages uniformly) while pip leaves
    #     caiman's pinned numpy/scipy/scikit-image/tifffile untouched. The MC and
    #     activity paths import only numpy/scipy/scikit-image/matplotlib/pillow/
    #     tifffile/nd2/yaml/caiman, all present here (scikit-learn/pandas are only
    #     needed by the analysis stage, which runs in the torch env).
    # A CUDA torch wheel is still NOT installed; caiman brings what it needs.
    python -m pip install --upgrade pip
    python -m pip install "nd2>=0.10" "tifffile>=2023.7" "pyyaml>=6.0" "pillow>=10.0"
    python -m pip install -e "${REPO_ROOT}" --no-deps
    python -c "import caiman, numpy, nd2, tifffile, yaml, PIL, orcann; print('  caiman env ready |', caiman.__version__)"
    command -v orcann >/dev/null && echo "  orcann command: $(command -v orcann)" \
        || { echo "  ERROR: orcann not on PATH after install" >&2; exit 1; }
}

if [ "${TARGET}" = "main" ] || [ "${TARGET}" = "all" ]; then
    setup_main
fi
if [ "${TARGET}" = "caiman" ] || [ "${TARGET}" = "all" ]; then
    setup_caiman
fi

echo "Done (${TARGET})."
