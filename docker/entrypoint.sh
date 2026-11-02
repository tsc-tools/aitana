#!/usr/bin/env bash
# Container entrypoint. Starts as root only long enough to lock down the
# network (init-firewall.sh), then permanently drops to the unprivileged
# `bot` user before starting the Slack bot -- root never touches anything
# the agent controls (prompts, fetched content, code it writes, output).
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "entrypoint: expected to start as root to configure the network sandbox" >&2
    exit 1
fi

/usr/local/sbin/init-firewall.sh

# Note: --reset-env instead of these explicit vars would also wipe whatever
# the caller passed with `docker run -e` (the API key and Slack tokens).
# The proxy vars route the bot's traffic through the allowlist; it can't
# reach the network any other way.
exec env HOME=/home/bot USER=bot LOGNAME=bot SHELL=/bin/bash \
    HTTPS_PROXY=http://127.0.0.1:8888 HTTP_PROXY=http://127.0.0.1:8888 \
    setpriv --reuid=bot --regid=bot --clear-groups --inh-caps=-all -- \
    /opt/aitana/.venv/bin/python -m aitana.bot.slack_bot
