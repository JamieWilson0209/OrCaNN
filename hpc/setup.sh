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
    python -c "import torch, orcann; print('  main env ready | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
}

setup_caiman() {
    echo "=== caiman env: ${CAIMAN_ENV} ==="
    # mamba resolves the caiman stack much faster than conda; use it if present.
    local solver
    solver="$(command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)"
    mkdir -p "$(dirname "${CAIMAN_ENV}")"
    if [ ! -d "${CAIMAN_ENV}" ]; then
        # one solve: conda-forge resolves caiman and a compatible stack.
        "${solver}" create --yes --prefix "${CAIMAN_ENV}" -c conda-forge caiman "python=3.11"
    fi
    activate "${CAIMAN_ENV}"
    # The motion-correction path is deliberately torch-free (pipeline/__init__ is
    # import-light; caiman is lazy-imported), so we can give the caiman env the
    # `orcann` command WITHOUT pulling the torch/segmentation stack. Install:
    #   - nd2 (Nikon reader) + tifffile: the movie-I/O extras on top of caiman.
    #   - pyyaml: configLoader needs it to read the config (caiman does not
    #     guarantee it).
    #   - orcann itself, editable and --no-deps: registers the `orcann` console
    #     script (so every job script invokes stages uniformly) while pip leaves
    #     caiman's pinned numpy/scipy/scikit-image/tifffile untouched. The MC path
    #     imports only numpy/tifffile/nd2/yaml/caiman, all present here.
    # A CUDA torch wheel is still NOT installed; caiman brings what it needs.
    python -m pip install --upgrade pip
    python -m pip install "nd2>=0.10" "tifffile>=2023.7" "pyyaml>=6.0"
    python -m pip install -e "${REPO_ROOT}" --no-deps
    python -c "import caiman, numpy, nd2, tifffile, yaml, orcann; print('  caiman env ready |', caiman.__version__)"
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
