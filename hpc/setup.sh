#!/bin/bash
# =============================================================================
# One-time environment setup on an SGE / Grid Engine HPC cluster. Builds the
# conda env(s) OrCaNN needs. Run from anywhere on a login node (installs are fine
# there; never train on the login node):
#
#     bash hpc/setup.sh            # BOTH envs (the default; this is what you want)
#     bash hpc/setup.sh main       # only the torch env
#     bash hpc/setup.sh caiman     # only the caiman env
#     bash hpc/setup.sh doctor     # diagnose only: caches, quota, env health
#
# The default builds BOTH because the caiman env is not optional for a normal
# run: it carries motion_correction AND activity (OASIS deconvolution is
# CaImAn's constrained_foopsi). Building only the torch env leaves a setup that
# looks complete and then fails at the first stage: activity now refuses to run
# OASIS without CaImAn rather than silently substituting another detector.
#
# Targets:
#   main    ENV_PREFIX  : python 3.11 + CUDA torch + `pip install -e .` (orcann).
#                         Used by every stage EXCEPT motion correction.
#   caiman  CAIMAN_ENV  : python 3.11 + caiman (conda-forge) + nd2 + tifffile +
#                         orcann (editable, --no-deps, for the `orcann` command).
#                         Used ONLY by `orcann motion_correction`. Kept separate
#                         so caiman's large pinned stack never constrains the
#                         torch env. REQUIRED for motion_correction and activity;
#                         skip it only if you are running neither.
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

TARGET="${1:-all}"
case "${TARGET}" in
    main|caiman|all|doctor) ;;
    *) echo "usage: bash hpc/setup.sh [all|main|caiman|doctor]   (default: all)" >&2
       exit 2 ;;
esac

. /etc/profile.d/modules.sh
module load "${ANACONDA_MODULE}"

# =============================================================================
# PREFLIGHT — every install failure seen on this cluster so far has been an
# environment problem, not a package problem: a cache conda wrote to home anyway,
# a Windows checkout, or a scratch purge that emptied an env in place. Each check
# below exists because one of those actually happened. All are cheap; all run
# before any solve, so a failure costs seconds rather than dying deep in conda.
#
# Note what is deliberately NOT here any more: a probe that home is writable.
# With the caches, TMPDIR and env registration (CONDA_REGISTER_ENVS) all
# redirected in config.sh, setup writes nothing to home, so a full home quota no
# longer breaks an install and must not be allowed to fail one. `doctor` still
# reports it, because it stays useful context when something else misbehaves.
# =============================================================================

# conda keeps channel notices in <XDG_CACHE_HOME>/conda/notices, which is NOT
# under pkgs_dirs, so relocating the package cache does not move it. A write
# truncated by a full quota leaves a 0-byte file there, and every later conda
# call dies reading it back:
#     json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
# Notices are core rather than a plugin, so --no-plugins does not help. Delete
# any empty or unparseable file; conda refetches on demand. The legacy ~/.cache
# path is swept too, since that is where the damage lands when XDG_CACHE_HOME was
# not yet set — i.e. exactly the runs that caused it.
clean_notices_cache() {
    local dir file removed=0
    for dir in "${XDG_CACHE_HOME}/conda/notices" "${HOME}/.cache/conda/notices"; do
        [ -d "${dir}" ] || continue
        for file in "${dir}"/*; do
            [ -f "${file}" ] || continue
            if [ ! -s "${file}" ] || ! python -c \
                 "import json,sys; json.load(open(sys.argv[1]))" "${file}" >/dev/null 2>&1; then
                rm -f "${file}"
                removed=$((removed + 1))
            fi
        done
    done
    if [ "${removed}" -gt 0 ]; then
        echo "  cleared ${removed} corrupt conda notices cache file(s)"
    fi
    return 0
}

# A Windows checkout with git's default core.autocrlf=true rewrites every LF to
# CRLF, and those files reach the cluster byte-for-byte over rsync/scp. bash then
# reads `set -euo pipefail\r` and rejects `pipefail<CR>` as an invalid option
# name. The repo's .gitattributes prevents this at checkout — but only for clones
# made after it was committed, so strip any CR that is here regardless.
# Idempotent, and a no-op on a clean clone.
#
# `tr` output is copied back with `cat` rather than `mv` so the file keeps its
# inode and its executable bit.
strip_crlf_scripts() {
    local file tmp fixed=0
    for file in "${REPO_ROOT}"/hpc/*.sh "${REPO_ROOT}"/hpc/jobs/*.sh; do
        [ -f "${file}" ] || continue
        grep -q $'\r' "${file}" 2>/dev/null || continue
        tmp="$(mktemp)"
        if tr -d '\r' < "${file}" > "${tmp}" && cat "${tmp}" > "${file}"; then
            fixed=$((fixed + 1))
        fi
        rm -f "${tmp}"
    done
    if [ "${fixed}" -gt 0 ]; then
        echo "  stripped CRLF from ${fixed} shell script(s) — this clone came from a"
        echo "  Windows checkout; see the line-endings section of hpc/README_HPC.md"
    fi
    return 0
}

# "The directory exists" is not "the env works". Scratch purges delete files by
# age from *inside* a tree, so a purged env keeps its directory and loses its
# interpreter; a `[ ! -d ]` guard then skips creation and the failure surfaces
# much later, as a confusing pip or import error. Test the interpreter.
env_ok() {
    [ -x "$1/bin/python" ] && "$1/bin/python" -c "import sys" >/dev/null 2>&1
}

# Refuse to work on a half-present env, and say exactly how to clear it. Deleting
# is left to the operator on purpose: this script never removes an env tree.
require_usable_env() {
    local prefix="$1" label="$2" target="$3"
    if [ -d "${prefix}" ] && ! env_ok "${prefix}"; then
        echo "ERROR: the ${label} at ${prefix} exists but its interpreter does not run." >&2
        echo "  A scratch purge removing files from inside the env does exactly this." >&2
        echo "  Delete the broken tree and re-run (this destroys the env):" >&2
        echo "    rm -rf '${prefix}' && bash hpc/setup.sh ${target}" >&2
        exit 1
    fi
}

# Caches live on scratch (see config.sh). Create them before conda runs: if the
# conda package cache falls back to a quota-limited home directory, the solve
# dies while writing repodata with a long, misleading "unexpected error" report.
preflight() {
    echo "=== preflight ==="
    mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_DIRS}" "${PIP_CACHE_DIR}" \
             "${TMPDIR}" "${XDG_CACHE_HOME}"
    echo "  pkgs cache : ${CONDA_PKGS_DIRS}"
    echo "  envs dir   : ${CONDA_ENVS_DIRS}"
    echo "  pip cache  : ${PIP_CACHE_DIR}"
    echo "  xdg cache  : ${XDG_CACHE_HOME}"
    clean_notices_cache
    strip_crlf_scripts
}

# Diagnose without installing anything: what the caches resolve to, whether home
# has room, and whether each env is actually usable. First thing to run when a
# job reports an env problem.
doctor() {
    echo "=== doctor ==="
    echo "  conda      : $(command -v conda || echo 'NOT FOUND')"
    echo "  pkgs cache : ${CONDA_PKGS_DIRS}"
    echo "  envs dir   : ${CONDA_ENVS_DIRS}"
    echo "  pip cache  : ${PIP_CACHE_DIR}"
    echo "  xdg cache  : ${XDG_CACHE_HOME}"
    echo "  notices    : ${CONDA_NUMBER_CHANNEL_NOTICES} (0 = disabled)"
    echo "  register   : ${CONDA_REGISTER_ENVS} (false = no ~/.conda write)"
    echo "  TMPDIR     : ${TMPDIR}"
    local prefix label
    for prefix in "${ENV_PREFIX}" "${CAIMAN_ENV}"; do
        if [ ! -d "${prefix}" ]; then
            label="ABSENT (not built, or purged entirely)"
        elif env_ok "${prefix}"; then
            label="ok — $(${prefix}/bin/python --version 2>&1)"
        else
            label="BROKEN (directory present, interpreter does not run)"
        fi
        echo "  env        : ${prefix}"
        echo "               ${label}"
    done
    if command -v quota >/dev/null 2>&1; then
        # quota exits non-zero when the user is OVER quota, so capture first and
        # test the text: piping straight through would print the real numbers and
        # the "nothing reported" fallback together, in exactly the case that
        # matters most.
        local q
        q="$(quota -s 2>/dev/null || true)"
        echo "  home quota :"
        if [ -n "${q}" ]; then
            echo "${q}" | sed 's/^/    /'
        else
            echo "    (quota reported nothing)"
        fi
    fi
    local probe="${HOME}/.orcann_setup_probe.$$"
    if (echo ok > "${probe}") 2>/dev/null; then
        rm -f "${probe}"
        echo "  home write : ok"
    else
        echo "  home write : FAILING — quota full. Setup does not need home, so"
        echo "               this will not block an install; it will break other"
        echo "               tools. See hpc/README_HPC.md"
    fi
}

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
    require_usable_env "${ENV_PREFIX}" "main env" "main"
    mkdir -p "$(dirname "${ENV_PREFIX}")"
    if ! env_ok "${ENV_PREFIX}"; then
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
    require_usable_env "${CAIMAN_ENV}" "caiman env" "caiman"
    mkdir -p "$(dirname "${CAIMAN_ENV}")"
    if ! env_ok "${CAIMAN_ENV}"; then
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

if [ "${TARGET}" = "doctor" ]; then
    doctor
    exit 0
fi

preflight

if [ "${TARGET}" = "main" ] || [ "${TARGET}" = "all" ]; then
    setup_main
fi
if [ "${TARGET}" = "caiman" ] || [ "${TARGET}" = "all" ]; then
    setup_caiman
fi

# Closing summary. The failure this prevents: a setup that reports success
# having built one env, followed hours later by a queued job dying on the other.
# Stage-to-env mapping is stated explicitly because it is not guessable — OASIS
# deconvolution lives in CaImAn, so `activity` needs the caiman env even though
# nothing about it looks like motion correction.
echo
echo "=== environments ==="
for spec in "main:${ENV_PREFIX}:infer, segment, analysis, train_spatial, train_temporal" \
            "caiman:${CAIMAN_ENV}:motion_correction, activity"; do
    name="${spec%%:*}"; rest="${spec#*:}"
    prefix="${rest%%:*}"; stages="${rest#*:}"
    if env_ok "${prefix}"; then
        echo "  [ok]      ${name} env  -> ${stages}"
    elif [ -d "${prefix}" ]; then
        echo "  [BROKEN]  ${name} env  -> ${stages}"
        echo "            ${prefix} exists but its interpreter does not run;"
        echo "            rm -rf '${prefix}' && bash hpc/setup.sh ${name}"
    else
        echo "  [MISSING] ${name} env  -> ${stages}"
        echo "            these stages will fail until: bash hpc/setup.sh ${name}"
    fi
done
echo

echo "Done (${TARGET})."
