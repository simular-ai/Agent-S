#!/usr/bin/env bash
# Stop an evolution run completely.
#
# Killing the tmux session and the driver is NOT enough: the driver launches
# agent_s4.eval.osworld_v2 as a separate process tree that keeps running,
# keeps its VMs, finishes every task (as imputed zeros if the model endpoint is
# broken) and writes latest_summary.json. A later run then resumes from that
# summary and the variant is handed a fabricated score. Kill the evaluator too,
# and verify it is gone before anyone touches the eval cache.
set -euo pipefail
NAMESPACE="${1:-}"
[[ -n "$NAMESPACE" ]] || { printf 'usage: %s <namespace> [--terminate-vms]\n' "$0" >&2; exit 2; }
TERMINATE=0
[[ "${2:-}" == "--terminate-vms" ]] && TERMINATE=1

tmux kill-session -t "evo15" 2>/dev/null || true
for pattern in "agent_s4\.cli run" "agent_s4\.eval\.osworld"; do
    pkill -TERM -f "$pattern" 2>/dev/null || true
done
sleep 10
for pattern in "agent_s4\.cli run" "agent_s4\.eval\.osworld"; do
    if pgrep -f "$pattern" >/dev/null; then
        printf 'escalating to SIGKILL for %s\n' "$pattern"
        pkill -KILL -f "$pattern" 2>/dev/null || true
    fi
done
sleep 3
still=0
for pattern in "agent_s4\.cli run" "agent_s4\.eval\.osworld"; do
    if pgrep -f "$pattern" >/dev/null; then
        printf 'ERROR: processes matching %s are still alive\n' "$pattern" >&2
        still=1
    fi
done
(( still == 0 )) || exit 1
printf 'driver and evaluator are stopped\n'

if (( TERMINATE )); then
    # Task VMs are matched by owner tag alone. Filtering on Name=shadow-evol-*
    # missed every VM launched by the test-split runner, which does not set a
    # Name tag -- eight of them were left running after a stop that reported
    # success. Long-lived infrastructure is excluded by explicit id instead,
    # which fails safe: an unknown instance is kept, never killed.
    if [[ -z "${SHADOW_EVOL_OWNER_TAG:-}" ]]; then
        printf 'ERROR: --terminate-vms matches task VMs by their Owner tag; set SHADOW_EVOL_OWNER_TAG.\n' >&2
        exit 2
    fi
    KEEP="${SHADOW_EVOL_KEEP_INSTANCES:-}"
    mapfile -t IDS < <(aws ec2 describe-instances \
        --filters Name=instance-state-name,Values=running,pending \
                  Name=tag:Owner,Values="$SHADOW_EVOL_OWNER_TAG" \
        --query 'Reservations[].Instances[].[InstanceId,LaunchTime]' --output text \
        | sort -k2 | awk -v keep="$KEEP" '
            BEGIN { n = split(keep, k, /[, ]+/); for (i = 1; i <= n; i++) skip[k[i]] = 1 }
            !($1 in skip) { print $1 }')
    if (( ${#IDS[@]} )); then
        aws ec2 terminate-instances --instance-ids "${IDS[@]}" \
            --query 'TerminatingInstances[].CurrentState.Name' --output text
    else
        printf 'no task VMs to terminate\n'
    fi
fi
