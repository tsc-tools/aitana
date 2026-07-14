#!/usr/bin/env bash
# Run the EGU 2026 example workflow inside Docker.
#
# Usage:
#   ./scripts/run_egu2026.sh [-h] [-o OUTDIR] [-c CORES]
#
# Options:
#   -o OUTDIR   Directory for workflow outputs (default: ./egu_2026_output)
#   -c CORES    Number of parallel cores for Snakemake
#               (default: all logical CPUs on the host)
#   -h          Show this help message and exit
#
# The Docker image is built from docker/Dockerfile on first run and cached by
# Docker. Conda environments for individual workflow rules are cached in a named
# Docker volume (aitana-conda-envs) and reused across runs.
#
# The poster generation step uses Apptainer inside Docker to run an Inkscape
# container.  Apptainer requires elevated capabilities; the flags below mirror
# the runArgs in .devcontainer/devcontainer.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$PROJECT_ROOT/egu_2026_output"
CORES="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
IMAGE_NAME="aitana-workflow"
CONDA_CACHE_VOLUME="aitana-conda-envs"

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
    exit 0
}

while getopts ":o:c:h" opt; do
    case $opt in
        o) OUTDIR="$OPTARG" ;;
        c) CORES="$OPTARG" ;;
        h) usage ;;
        :) echo "Option -${OPTARG} requires an argument." >&2; exit 1 ;;
        \?) echo "Unknown option: -${OPTARG}" >&2; exit 1 ;;
    esac
done

OUTDIR="$(mkdir -p "$OUTDIR" && cd "$OUTDIR" && pwd)"

echo "==> Building Docker image '${IMAGE_NAME}' ..."
docker build \
    --build-arg D_UID="$(id -u)" \
    --build-arg D_GID="$(id -g)" \
    -t "$IMAGE_NAME" \
    -f "$PROJECT_ROOT/docker/Dockerfile" \
    "$PROJECT_ROOT"

echo "==> Running workflow (cores=${CORES}) → ${OUTDIR}"
docker run --rm \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --device=/dev/fuse \
    -v "$PROJECT_ROOT:/project:ro" \
    -v "$OUTDIR:/results" \
    -v "$CONDA_CACHE_VOLUME:/opt/conda/envs" \
    "$IMAGE_NAME" \
    bash -c "
        set -euo pipefail
        echo '--> Installing aitana from source ...'
        pip install /project --quiet
        echo '--> Launching volcanobench ...'
        volcanobench run whakaari /results --cores ${CORES}
    "

echo "==> Done. Results written to: ${OUTDIR}"
