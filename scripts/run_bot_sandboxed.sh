#!/usr/bin/env bash
# Run the aitana Slack bot inside the hardened sandbox in docker/Dockerfile_bot.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... SLACK_BOT_TOKEN=xoxb-... SLACK_APP_TOKEN=xapp-... \
#   [GRAFANA_MCP_URL=http://host:port/mcp] \
#   ./scripts/run_bot_sandboxed.sh [-h] [-d DOMAINS] [-m MEMORY] [-c CPUS]
#
# GRAFANA_MCP_URL is optional; without it the bot has no Grafana tools.
#
# Options:
#   -d DOMAINS  Extra space-separated domains to allow through the egress proxy,
#               on top of the built-in defaults in docker/init-firewall.sh
#   -m MEMORY   Memory limit, docker --memory syntax (default: 4g)
#   -c CPUS     CPU limit, docker --cpus syntax (default: 4)
#   -h          Show this help message and exit
#
# What this guards against:
#   - the bot, and the code its agent writes and runs, run as a non-root user
#     with no Linux capabilities of its own (docker/entrypoint.sh drops
#     privileges via setpriv after setup)
#   - its only network path is a local allowlisting proxy; anything not on
#     the allowlist gets a 403, and direct egress is firewalled off entirely
#     (docker/init-firewall.sh)
#   - the container gets no capabilities beyond what setup needs (CHOWN,
#     SETUID/SETGID to drop privileges, NET_ADMIN/NET_RAW for the firewall),
#     can't gain new ones (no-new-privileges), and is capped on memory,
#     CPU, and process count
#   - nothing from the host is mounted -- the aitana code is baked into the
#     image, owned by root
#
# What it does NOT do: the root filesystem is left writable (not --read-only).
# The agent writes scripts and plots under /tmp, and matplotlib caches under
# $HOME; making that survive a read-only rootfs needs tmpfs mounts for those
# paths, which isn't done here. The layers above are the
# real boundary; read-only rootfs would be defense in depth on top, not
# instead of them. Commands the agent runs don't inherit the tokens, but they
# run as the same user as the bot, so they could still read them from /proc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MEMORY="4g"
CPUS="4"
EXTRA_DOMAINS=""
IMAGE_NAME="aitana-bot-sandbox"

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
    exit 0
}

while getopts ":d:m:c:h" opt; do
    case $opt in
        d) EXTRA_DOMAINS="$OPTARG" ;;
        m) MEMORY="$OPTARG" ;;
        c) CPUS="$OPTARG" ;;
        h) usage ;;
        :) echo "Option -${OPTARG} requires an argument." >&2; exit 1 ;;
        \?) echo "Unknown option: -${OPTARG}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

for var in ANTHROPIC_API_KEY SLACK_BOT_TOKEN SLACK_APP_TOKEN; do
    [[ -n "${!var:-}" ]] || { echo "$var must be set" >&2; exit 1; }
done

echo "==> Building image '${IMAGE_NAME}' (UID=$(id -u) GID=$(id -g)) ..."
docker build \
    --build-arg D_UID="$(id -u)" \
    --build-arg D_GID="$(id -g)" \
    -t "$IMAGE_NAME" \
    -f "$PROJECT_ROOT/docker/Dockerfile_bot" \
    "$PROJECT_ROOT"

echo "==> Running the Slack bot sandboxed (Ctrl-C to stop)"
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
    -e SLACK_BOT_TOKEN \
    -e SLACK_APP_TOKEN \
    -e GRAFANA_MCP_URL \
    -e BOT_EXTRA_DOMAINS="$EXTRA_DOMAINS" \
    "$IMAGE_NAME"
