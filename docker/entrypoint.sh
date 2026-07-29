#!/usr/bin/env bash
# Container entrypoint. Starts as root only long enough to lock down the
# network (init-firewall.sh), then permanently drops to the unprivileged
# `claude` user before running Claude Code -- root never touches anything
# the agent controls (prompts, fetched content, task output).
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "entrypoint: expected to start as root to configure the network sandbox" >&2
    exit 1
fi

/usr/local/sbin/init-firewall.sh

# Note: --reset-env instead of these explicit vars would also wipe whatever
# the caller passed with `docker run -e` (ANTHROPIC_API_KEY included).
exec env HOME=/home/claude USER=claude LOGNAME=claude SHELL=/bin/bash \
    setpriv --reuid=claude --regid=claude --clear-groups --inh-caps=-all -- \
    claude "$@"
