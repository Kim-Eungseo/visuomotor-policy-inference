#!/usr/bin/env bash
# dg5f_ip.sh — resolve the DG5F (left hand) Modbus/502 IP by trying its known addresses.
#
# The hand answers on ONE of two addresses depending on how its ethernet is configured:
#   192.168.4.73     — main robot subnet (same switch as arm/cameras), if reconfigured there
#   169.254.186.73   — Tesollo native link-local default (what the ROS driver ships with)
# This picks whichever actually answers on 502, adding the matching host alias on enp4s0 as
# needed (both aliases are lost on reboot). Source it, then use $DG5F_IP.
#
#   source ./dg5f_ip.sh
#   resolve_dg5f_ip && echo "hand at $DG5F_IP" || echo "hand not reachable"
#
# Env overrides: DG5F_NIC (default enp4s0), DG5F_PORT (default 502),
#                DG5F_CANDIDATES ("ip=host_alias/cidr ..." pairs, first match wins).

# candidate hand-IP -> host alias needed to route to it
DG5F_CANDIDATES="${DG5F_CANDIDATES:-192.168.4.73=192.168.4.55/24 169.254.186.73=169.254.186.55/24}"
DG5F_PORT="${DG5F_PORT:-502}"
DG5F_NIC="${DG5F_NIC:-enp4s0}"
DG5F_IP=""

_dg5f_port_open() { timeout 2 bash -c "echo > /dev/tcp/$1/$DG5F_PORT" 2>/dev/null; }

# Sets DG5F_IP to the first candidate that answers on $DG5F_PORT. Returns 0 on success.
resolve_dg5f_ip() {
  local pair ip alias
  for pair in $DG5F_CANDIDATES; do
    ip="${pair%%=*}"; alias="${pair#*=}"
    # the host must NOT hold the hand's own IP (that's a conflict) -> drop it
    if ip -br addr show "$DG5F_NIC" 2>/dev/null | grep -qw "$ip"; then
      sudo ip addr del "$ip/24" dev "$DG5F_NIC" 2>/dev/null || true
    fi
    # ensure the host has an address on this subnet so it can reach the hand
    ip -br addr show "$DG5F_NIC" 2>/dev/null | grep -qw "${alias%/*}" \
      || sudo ip addr add "$alias" dev "$DG5F_NIC" 2>/dev/null || true
    if _dg5f_port_open "$ip"; then DG5F_IP="$ip"; return 0; fi
  done
  DG5F_IP=""
  return 1
}
