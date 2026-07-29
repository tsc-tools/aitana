#!/usr/bin/env bash
# Runs once, as root, before entrypoint.sh drops to the unprivileged
# `claude` user. Locks down all outbound traffic behind a domain allowlist:
# only `proxyuser` (running tinyproxy) may reach the network directly, and
# tinyproxy refuses CONNECT/requests to anything not on the filter list.
# `claude` can therefore only leave the container through that allowlist,
# matched by exact hostname (no subdomain/substring bypass).
#
# Fails closed: any error here aborts the container instead of leaving
# egress unrestricted.
set -euo pipefail

# Domains Claude Code itself needs to function -- do not remove.
# Source: https://code.claude.com/docs/en/network-config.md
CORE_DOMAINS=(
    api.anthropic.com
    claude.ai
    claude.com
    platform.claude.com
    mcp-proxy.anthropic.com
)

# Domains this workspace's tasks typically need (git remote, package
# installs). Tune per project -- narrower is safer.
DEV_DOMAINS=(
    github.com
    objects.githubusercontent.com
    raw.githubusercontent.com
    pypi.org
    files.pythonhosted.org
)

# Additional space-separated domains from the run script, e.g.
# `-e CLAUDE_EXTRA_DOMAINS="foo.example.com bar.example.com"`.
read -ra EXTRA_DOMAINS <<< "${CLAUDE_EXTRA_DOMAINS:-}"

ALL_DOMAINS=("${CORE_DOMAINS[@]}" "${DEV_DOMAINS[@]}" "${EXTRA_DOMAINS[@]}")

PROXY_PORT=8888
RUNTIME_DIR=/run/init-firewall
mkdir -p "$RUNTIME_DIR"

filter_file="$RUNTIME_DIR/filter"
printf '%s\n' "${ALL_DOMAINS[@]}" > "$filter_file"

cat > "$RUNTIME_DIR/tinyproxy.conf" <<CONF
User proxyuser
Group proxyuser
Port ${PROXY_PORT}
Listen 127.0.0.1
Timeout 600
Filter "${filter_file}"
FilterType fnmatch
FilterDefaultDeny Yes
ConnectPort 443
ConnectPort 80
PidFile "${RUNTIME_DIR}/tinyproxy.pid"
LogFile "${RUNTIME_DIR}/tinyproxy.log"
CONF

chown -R proxyuser:proxyuser "$RUNTIME_DIR"

setpriv --reuid=proxyuser --regid=proxyuser --clear-groups --inh-caps=-all -- \
    tinyproxy -c "$RUNTIME_DIR/tinyproxy.conf"

for _ in $(seq 1 50); do
    [[ -f "${RUNTIME_DIR}/tinyproxy.pid" ]] && break
    sleep 0.1
done
[[ -f "${RUNTIME_DIR}/tinyproxy.pid" ]] || {
    echo "init-firewall: tinyproxy failed to start" >&2
    cat "${RUNTIME_DIR}/tinyproxy.log" >&2 2>/dev/null || true
    exit 1
}

# Loopback: needed both ways -- claude -> 127.0.0.1:8888 is OUTPUT+lo, and
# tinyproxy's accept() on that socket is INPUT+lo, in the same netns.
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -i lo -j ACCEPT

# Only proxyuser may leave the container directly (web ports + DNS).
iptables -A OUTPUT -m owner --uid-owner proxyuser -p tcp --dport 443 -j ACCEPT
iptables -A OUTPUT -m owner --uid-owner proxyuser -p tcp --dport 80 -j ACCEPT
iptables -A OUTPUT -m owner --uid-owner proxyuser -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -m owner --uid-owner proxyuser -p tcp --dport 53 -j ACCEPT

# Return traffic for connections proxyuser initiated.
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

iptables -P OUTPUT DROP
iptables -P INPUT DROP
iptables -P FORWARD DROP

echo "init-firewall: egress locked to ${#ALL_DOMAINS[@]} allowlisted domains via 127.0.0.1:${PROXY_PORT}"
