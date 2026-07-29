#!/usr/bin/env bash
# Run Claude Code unattended, inside the hardened sandbox in docker/Dockerfile_cc.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... ./scripts/run_claude_sandboxed.sh [-h] [-w WORKDIR] [-d DOMAINS] [-m MEMORY] [-c CPUS] -- [claude args...]
#
# Options:
#   -w WORKDIR  Directory bind-mounted read-write at /workspace (default: current directory)
#   -d DOMAINS  Extra space-separated domains to allow through the egress proxy,
#               on top of the built-in defaults in docker/init-firewall.sh
#   -m MEMORY   Memory limit, docker --memory syntax (default: 4g)
#   -c CPUS     CPU limit, docker --cpus syntax (default: 4)
#   -h          Show this help message and exit
#
# What this guards against:
#   - claude runs as a non-root user with no Linux capabilities of its own
#     (docker/entrypoint.sh drops privileges via setpriv after setup)
#   - its only network path is a local allowlisting proxy; anything not on
#     the allowlist gets a 403, and direct egress is firewalled off entirely
#     (docker/init-firewall.sh)
#   - the container gets no capabilities beyond what setup needs (CHOWN,
#     SETUID/SETGID to drop privileges, NET_ADMIN/NET_RAW for the firewall),
#     can't gain new ones (no-new-privileges), and is capped on memory,
#     CPU, and process count
#   - only WORKDIR is mounted -- no host $HOME, no ~/.ssh, no docker socket,
#     no real ~/.claude credentials
#
# What it does NOT do: the root filesystem is left writable (not --read-only).
# Claude Code writes session state, shell snapshots, and caches under
# ~/.claude at runtime; making that survive a read-only rootfs needs the
# entrypoint to seed a tmpfs before dropping privileges, which isn't done
# here. The layers above are the real boundary; read-only rootfs would be
# defense in depth on top, not instead of them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR="$(pwd)"
MEMORY="4g"
CPUS="4"
EXTRA_DOMAINS=""
IMAGE_NAME="claude-sandbox"

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
    exit 0
}

while getopts ":w:d:m:c:h" opt; do
    case $opt in
        w) WORKDIR="$OPTARG" ;;
        d) EXTRA_DOMAINS="$OPTARG" ;;
        m) MEMORY="$OPTARG" ;;
        c) CPUS="$OPTARG" ;;
        h) usage ;;
        :) echo "Option -${OPTARG} requires an argument." >&2; exit 1 ;;
        \?) echo "Unknown option: -${OPTARG}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

[[ -n "${ANTHROPIC_API_KEY:-}" ]] || {
    echo "ANTHROPIC_API_KEY must be set -- this sandbox authenticates with a scoped" >&2
    echo "API key, not your host's claude.ai login (which would hand it a real session)." >&2
    exit 1
}

WORKDIR="$(cd "$WORKDIR" && pwd)"

echo "==> Building image '${IMAGE_NAME}' (UID=$(id -u) GID=$(id -g)) ..."
docker build \
    --build-arg D_UID="$(id -u)" \
    --build-arg D_GID="$(id -g)" \
    -t "$IMAGE_NAME" \
    -f "$PROJECT_ROOT/docker/Dockerfile_cc" \
    "$PROJECT_ROOT"

echo "==> Running claude sandboxed. Workspace: ${WORKDIR} -> /workspace (rw)"
docker run --rm -it \
    --cap-drop=ALL \
    --cap-add=CHOWN \
    --cap-add=SETUID \
    --cap-add=SETGID \
    --cap-add=NET_ADMIN \
    --cap-add=NET_RAW \
    --security-opt=no-new-privileges \
    --pids-limit=1024 \
    --memory="$MEMORY" \
    --cpus="$CPUS" \
    -e ANTHROPIC_API_KEY \
    -e CLAUDE_EXTRA_DOMAINS="$EXTRA_DOMAINS" \
    -v "$WORKDIR:/workspace" \
    "$IMAGE_NAME" \
    "$@"
