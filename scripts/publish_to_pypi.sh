#!/usr/bin/env bash
set -euo pipefail

# Publish one or all aitana distributions to PyPI.
# Usage: publish_to_pypi.sh [distribution-name]
# If no argument given, publishes all in dependency order.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Dependency-ordered list
ALL_DISTS=(aitana-access aitana-features aitana-volcanoes aitana-bench)

die() { echo "ERROR: $*" >&2; exit 1; }

check_clean_tree() {
    git -C "$REPO_ROOT" diff --exit-code --quiet \
        || die "Uncommitted changes in working tree. Commit or stash before publishing."
    git -C "$REPO_ROOT" diff --cached --exit-code --quiet \
        || die "Staged changes present. Commit before publishing."
}

build_and_upload() {
    local dist="$1"
    local dist_dir="$REPO_ROOT/$dist"
    [[ -d "$dist_dir" ]] || die "Distribution directory not found: $dist_dir"

    echo "==> Building $dist"
    rm -rf "$dist_dir/dist"
    uv run python -m build "$dist_dir"

    echo "==> Uploading $dist"
    uv run python -m twine upload "$dist_dir/dist/*"
}

check_clean_tree

if [[ $# -eq 0 ]]; then
    for dist in "${ALL_DISTS[@]}"; do
        build_and_upload "$dist"
    done
elif [[ $# -eq 1 ]]; then
    build_and_upload "$1"
else
    echo "Usage: $(basename "$0") [distribution-name]" >&2
    echo "Available distributions: ${ALL_DISTS[*]}" >&2
    exit 1
fi

echo "Done."
