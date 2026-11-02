#!/usr/bin/env bash
# Runs once, as root, before entrypoint.sh drops to the unprivileged
# `bot` user. Locks down all outbound traffic behind a domain allowlist:
# only `proxyuser` (running tinyproxy) may reach the network directly, and
# tinyproxy refuses CONNECT/requests to anything not on the filter list.
# `bot` can therefore only leave the container through that allowlist,
# matched by exact hostname (no subdomain/substring bypass).
#
# Fails closed: any error here aborts the container instead of leaving
# egress unrestricted.
set -euo pipefail

# Domains the bot itself needs: the Claude API, and Slack's Web API,
# Socket Mode websockets and file uploads (for plots).
BOT_DOMAINS=(
    api.anthropic.com
    slack.com
    wss-primary.slack.com
    wss-backup.slack.com
    files.slack.com
)

# GeoNet data the agent's aitana code fetches: Tilde, WFS, FDSN waveforms
# and the open-data S3 archive.
DATA_DOMAINS=(
    tilde.geonet.org.nz
    wfs.geonet.org.nz
    service.geonet.org.nz
    service-nrt.geonet.org.nz
    geonet-open-data.s3.amazonaws.com
)

# The Grafana MCP server, if the bot uses one, from the URL it connects to. It
# may listen on a port other than 80/443, so that port is opened below as well.
GRAFANA_DOMAINS=()
GRAFANA_PORT=""
if [[ -n "${GRAFANA_MCP_URL:-}" ]]; then
    grafana_url="$GRAFANA_MCP_URL"
    grafana_hostport="${grafana_url#*://}"
    grafana_hostport="${grafana_hostport%%/*}"
    GRAFANA_DOMAINS=("${grafana_hostport%%:*}")
    if [[ "$grafana_hostport" == *:* ]]; then
        GRAFANA_PORT="${grafana_hostport##*:}"
    elif [[ "$grafana_url" == https://* ]]; then
        GRAFANA_PORT=443
    else
        GRAFANA_PORT=80
    fi
    [[ "$GRAFANA_PORT" =~ ^[0-9]+$ ]] || {
        echo "init-firewall: can't read a port from GRAFANA_MCP_URL=${grafana_url}" >&2
        exit 1
    }
fi

# Additional space-separated domains from the run script, e.g.
# `-e BOT_EXTRA_DOMAINS="foo.example.com bar.example.com"`.
read -ra EXTRA_DOMAINS <<< "${BOT_EXTRA_DOMAINS:-}"

ALL_DOMAINS=("${BOT_DOMAINS[@]}" "${DATA_DOMAINS[@]}" "${GRAFANA_DOMAINS[@]}" "${EXTRA_DOMAINS[@]}")

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
${GRAFANA_PORT:+ConnectPort ${GRAFANA_PORT}}
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
if [[ -n "$GRAFANA_PORT" ]]; then
    iptables -A OUTPUT -m owner --uid-owner proxyuser -p tcp --dport "$GRAFANA_PORT" -j ACCEPT
fi
iptables -A OUTPUT -m owner --uid-owner proxyuser -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -m owner --uid-owner proxyuser -p tcp --dport 53 -j ACCEPT

# Return traffic for connections proxyuser initiated.
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

iptables -P OUTPUT DROP
iptables -P INPUT DROP
iptables -P FORWARD DROP

echo "init-firewall: egress locked to ${#ALL_DOMAINS[@]} allowlisted domains via 127.0.0.1:${PROXY_PORT}"
