#!/usr/bin/env bash
# Evaluate one evolved shadow-interface variant on a held-out OSWorld V2 task
# set with Prime Agent (default) or OpenCode (SHADOW_EVOL_TEST_HARNESS).
#
# The evolution driver never touches the test split — held_out_tasks_path is
# only a leak guard — so a reportable test number requires this separate run.
#
# The variant is materialized from its git tag into an isolated directory
# rather than reusing the evolution clone's working tree, so the run is
# reproducible and cannot perturb (or be perturbed by) evolution state.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# SHADOW_EVOL_SHARED_ROOT points the launcher at another checkout's .venv and
# .env; by default both come from this project directory.
SHARED_ROOT="${SHADOW_EVOL_SHARED_ROOT:-$ROOT}"
ENV_FILE="${ENV_FILE:-$SHARED_ROOT/.env}"
DOTENV="$SHARED_ROOT/.venv/bin/dotenv"
PYTHON="$SHARED_ROOT/.venv/bin/python"
VENV_BIN="$SHARED_ROOT/.venv/bin"
# Launch from this project's venv (osworld_setup/s4/.venv; see README.md).
export PATH="$VENV_BIN:$PATH"
# agent_s4 imports from the Agent-S repository root, installed or not.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
# Settings this launcher checks itself are read from .env when not exported.
_env_default() {
    local name value
    for name in "$@"; do
        [[ -n "${!name:-}" ]] && continue
        value="$("$VENV_BIN/dotenv" -f "$ENV_FILE" get "$name" 2>/dev/null || true)"
        if [[ -n "$value" ]]; then export "$name=$value"; fi
    done
    return 0
}
_env_default AWS_PROFILE AWS_REGION AWS_SUBNET_ID AWS_SECURITY_GROUP_ID \
    SHADOW_EVOL_OWNER_TAG

NAMESPACE="${NAMESPACE:-prime25v15r4}"
VARIANT="${VARIANT:-if-006}"
NUM_ENVS="${NUM_ENVS:-20}"
# The held-out manifest. This project's split is 25 train / 15 val / 68 test,
# so the default is the 68-task remainder of the 108-task pool, not OSWorld's
# canonical 33-task test split.
TASKS_FILE="${TASKS_FILE:-eval_sets/osworld_v2_heldout_68.json}"
MAX_AGENT_STEPS=300
EVAL_TIMEOUT_COMPAT=1200
INFRA_RETRIES="${OSWORLD_INFRA_RETRIES:-2}"

AWS_PROFILE_NAME="${AWS_PROFILE:-}"
AWS_REGION_NAME="${AWS_REGION:-us-east-1}"
AWS_SUBNET="${AWS_SUBNET_ID:-}"
# The security group must admit this host on the task VMs' control port: a
# group that only admits the VPC CIDR silently hangs every task in setup when
# the orchestrator runs off-VPC (checked below).
AWS_SECURITY_GROUP="${AWS_SECURITY_GROUP_ID:-}"
ASSET_DIR="${OSWORLD_FILE_BASE_URL:-$ROOT/OSWorldV2_Base/cache/osworld_v2_assets}"
PROXY_CONFIG="${PROXY_CONFIG_FILE:-$ROOT/OSWorldV2_Base/evaluation_examples/settings/proxy/dataimpulse.json}"
WEBSITE_SUFFIX="${WEBSITE_HOST_SUFFIX:-site.hku.icu}"
OWNER_TAG="${SHADOW_EVOL_OWNER_TAG:-}"
MODEL="${SHADOW_EVOL_MODEL:-openrouter/qwen/qwen3.8-27b}"
# Agent harness. A non-default harness gets its own result tree (suffix
# _<harness>) so its runs never merge with Prime runs of the same variant.
HARNESS="${SHADOW_EVOL_TEST_HARNESS:-prime_agent}"
case "$HARNESS" in
    prime_agent) HARNESS_SUFFIX="" ;;
    opencode) HARNESS_SUFFIX="_opencode" ;;
    *) printf 'ERROR: SHADOW_EVOL_TEST_HARNESS=%s (prime_agent|opencode).\n' "$HARNESS" >&2; exit 2 ;;
esac
ALLOW_PLACEHOLDER_PROXY="${SHADOW_EVOL_ALLOW_PLACEHOLDER_PROXY:-1}"
# Bare-harness mode: run Prime Agent with NO shadow interface at all -- no
# upload, no setup.sh, no eval constraints, and the stock NO_TOOL system
# prompt instead of TOOL_SYSTEM_PROMPT (agent_s4/eval/osworld.py:716,777).
# This is the floor the whole experiment is measured against, so it
# deliberately materializes no variant and writes to its own result
# namespace; --no_tool also moves results under shadow_eval_no_tool/ rather
# than shadow_eval/, which any reader of these numbers has to account for.
NO_TOOL=0
case "${SHADOW_EVOL_NO_TOOL:-0}" in
    1|true|True|TRUE|yes|on) NO_TOOL=1 ;;
    0|false|False|FALSE|no|off|"") NO_TOOL=0 ;;
    *) printf 'ERROR: SHADOW_EVOL_NO_TOOL=%s is not a boolean.\n' \
           "$SHADOW_EVOL_NO_TOOL" >&2; exit 2 ;;
esac

CONFIRMED=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes) CONFIRMED=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --tasks-file) TASKS_FILE="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# AWS resources come from the environment (or .env); nothing account-specific
# is defaulted here. AWS_PROFILE is optional: leave it unset to use boto3's
# normal credential chain.
if [[ -n "$AWS_PROFILE_NAME" ]]; then export AWS_PROFILE="$AWS_PROFILE_NAME"; else unset AWS_PROFILE; fi
for _required in AWS_SUBNET:AWS_SUBNET_ID AWS_SECURITY_GROUP:AWS_SECURITY_GROUP_ID \
                 OWNER_TAG:SHADOW_EVOL_OWNER_TAG; do
    _var="${_required%%:*}"
    if [[ -z "${!_var}" ]]; then
        printf 'ERROR: set %s (see .env.example).\n' "${_required#*:}" >&2
        exit 2
    fi
done

if [[ "$NO_TOOL" == "1" ]]; then
    VARIANT="bare"
fi
CLONE="$ROOT/OSWorldV2_Evol_$NAMESPACE"
TAG="evol/$NAMESPACE/$VARIANT"
VARIANT_ID="$NAMESPACE-$VARIANT"

if [[ "$NO_TOOL" != "1" ]]; then
    [[ -d "$CLONE/.git" ]] || { printf 'ERROR: no evolution clone at %s\n' "$CLONE" >&2; exit 1; }
    git -C "$CLONE" rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
        || { printf 'ERROR: tag %s not found in %s\n' "$TAG" "$CLONE" >&2; exit 1; }
fi

# Read the held-out manifest. Held-out isolation is verified at run time, not
# just at manifest-authoring time: a variant tuned on a leaked task would make
# this number meaningless. (A multi-line process substitution followed by a
# heredoc confuses bash's heredoc collection, so this stays on one line.)
# ${VAR-default}, not ${VAR:-default}: an explicitly empty value means "this
# run deliberately evaluates a split the variant has seen" (a baseline
# measurement, say). That is never the default and never silent.
LEAK_GUARD_SETS="${SHADOW_EVOL_LEAK_GUARD_SETS-osworld_v2_train_25.json,osworld_v2_val_15.json}"
if [[ -z "$LEAK_GUARD_SETS" ]]; then
    printf 'WARNING: leak guard disabled. This number is NOT a held-out result.\n' >&2
fi
mapfile -t TASK_IDS < <(python3 -c 'import json,sys,pathlib; r=pathlib.Path(sys.argv[1]); m=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert len(m)==len(set(m)), "held-out manifest has duplicate ids"; [sys.exit(f"ERROR: {n} leaks into the held-out set: {sorted(set(json.loads((r/n).read_text())) & set(m))}") for n in sys.argv[3].split(",") if n if set(json.loads((r/n).read_text())) & set(m)]; print("\n".join(m))' "$ROOT/eval_sets" "$ROOT/$TASKS_FILE" "$LEAK_GUARD_SETS")
(( ${#TASK_IDS[@]} > 0 )) || { printf 'ERROR: no task ids read from %s\n' "$TASKS_FILE" >&2; exit 1; }

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
TASKS_TAG="$(basename "$TASKS_FILE" .json)"
# A context-window sweep evaluates the SAME variant on the SAME tasks more
# than once, so the window has to be part of the result path. Without it the
# runs land in sibling run-id directories and any "newest result per task"
# reader silently merges two different configurations into one score.
CW_SUFFIX=""
[[ -n "${CONTEXT_WINDOW_OVERRIDE:-}" ]] && CW_SUFFIX="_cw${CONTEXT_WINDOW_OVERRIDE}"
# keep_recent_tokens is the other half of the compaction budget, and it is not
# independent of the window: Prime refuses to compact when the compactable
# prefix estimates below keepRecentTokens, so a window small enough to trigger
# before that threshold never compacts at all. Sweeping it needs the same
# path separation as the window, for the same reason.
[[ -n "${KEEP_RECENT_OVERRIDE:-}" ]] && CW_SUFFIX="${CW_SUFFIX}_keep${KEEP_RECENT_OVERRIDE}"
RESULT_ROOT="$ROOT/results/osworld_v2_test_${NAMESPACE}_${VARIANT}${HARNESS_SUFFIX}/${TASKS_TAG}${CW_SUFFIX}/$RUN_ID"
SHADOW_DIR="$RESULT_ROOT/shadow_interface"

# Resume support. A run that was interrupted leaves genuine per-task results in
# earlier run directories of the SAME split; re-running those tasks wastes VM
# time and, because each attempt is an independent sample, silently replaces a
# measurement with a different one. With SHADOW_EVOL_SKIP_SCORED=1 the task list
# drops every task that already carries a genuine result, using the same rule
# the report applies: a result.txt with no failure.json beside it and no
# trajectory ending "finalError":"Connection error.". Infrastructure failures
# are therefore NOT skipped -- they are exactly what a resume has to redo.
case "${SHADOW_EVOL_SKIP_SCORED:-0}" in
    1|true|True|TRUE|yes|on)
        mapfile -t TASK_IDS < <("$PYTHON" - "$(dirname "$RESULT_ROOT")" "${TASK_IDS[@]}" <<'SKIPPY'
import sys
from pathlib import Path

split_dir, wanted = Path(sys.argv[1]), sys.argv[2:]


def genuine(task_dir):
    if (task_dir / "failure.json").is_file():
        return False
    trajectory = task_dir / "claude_trajectory.jsonl"
    if trajectory.is_file():
        with trajectory.open("rb") as handle:
            for line in handle:
                if b'"finalError":"Connection error.' in line:
                    return False
    return True


scored = set()
if split_dir.is_dir():
    for result in split_dir.glob("*/*/*/*/*/*/result.txt"):
        if result.parent.name in wanted and genuine(result.parent):
            scored.add(result.parent.name)
remaining = [task for task in wanted if task not in scored]
print(f"skip-scored: {len(scored)} already measured, {len(remaining)} to run",
      file=sys.stderr)
print("\n".join(remaining))
SKIPPY
)
        (( ${#TASK_IDS[@]} > 0 )) || {
            printf 'Nothing to do: every task in %s already has a genuine result.\n' \
                "$TASKS_FILE"; exit 0; }
        ;;
    0|false|False|FALSE|no|off|"") ;;
    *) printf 'ERROR: SHADOW_EVOL_SKIP_SCORED=%s is not a boolean.\n' \
           "$SHADOW_EVOL_SKIP_SCORED" >&2; exit 2 ;;
esac
mkdir -p "$SHADOW_DIR"

# Materialize the variant from its tag. --strip-components drops the
# shadow_interface/ prefix so the layout matches what the evaluator expects
# (the same shape as seeds/combined).
if [[ "$NO_TOOL" == "1" ]]; then
    printf 'Bare run: no shadow interface is materialized or uploaded.\n'
else
    git -C "$CLONE" archive "$TAG" shadow_interface \
        | tar -x --strip-components=1 -C "$SHADOW_DIR"
    printf 'Materialized %s (%s) into %s\n' "$TAG" \
        "$(git -C "$CLONE" rev-parse --short "$TAG")" "$SHADOW_DIR"
fi

# Optional context-window sweep. The evolved interface owns this value, so
# the override is applied to the materialized copy only -- the variant's
# git tag is never rewritten.
if [[ -n "${CONTEXT_WINDOW_OVERRIDE:-}" ]]; then
    "$PYTHON" - "$SHADOW_DIR/context_compaction.json" "$CONTEXT_WINDOW_OVERRIDE" <<'CWPY'
import json, sys
from pathlib import Path
path, value = Path(sys.argv[1]), int(sys.argv[2])
config = json.loads(path.read_text())
before = config.get("context_window_tokens")
config["context_window_tokens"] = value
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"context_window_tokens override: {before} -> {value}")
CWPY
fi

if [[ -n "${KEEP_RECENT_OVERRIDE:-}" ]]; then
    "$PYTHON" - "$SHADOW_DIR/context_compaction.json" "$KEEP_RECENT_OVERRIDE" <<'KRPY'
import json, sys
from pathlib import Path
path, value = Path(sys.argv[1]), int(sys.argv[2])
config = json.loads(path.read_text())
before = config.get("keep_recent_tokens")
config["keep_recent_tokens"] = value
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"keep_recent_tokens override: {before} -> {value}")
KRPY
fi

if [[ -z "${SHADOW_PRIME_AGENT_BIN:-}" ]]; then
    if [[ -x /opt/prime-agent/bin/prime-agent ]]; then
        export SHADOW_PRIME_AGENT_BIN=/opt/prime-agent/bin/prime-agent
    elif [[ -x "$ROOT/vm_assets/prime_agent_host/opt/prime-agent/bin/prime-agent" ]]; then
        export SHADOW_PRIME_AGENT_BIN="$ROOT/vm_assets/prime_agent_host/opt/prime-agent/bin/prime-agent"
    fi
fi

with_prime_env() {
    \
    AWS_REGION="$AWS_REGION_NAME" \
    AWS_DEFAULT_REGION="$AWS_REGION_NAME" \
    AWS_SUBNET_ID="$AWS_SUBNET" \
    AWS_SECURITY_GROUP_ID="$AWS_SECURITY_GROUP" \
    WEBSITE_HOST_SUFFIX="$WEBSITE_SUFFIX" \
    OSWORLD_FILE_BASE_URL="$ASSET_DIR" \
    PROXY_CONFIG_FILE="$PROXY_CONFIG" \
    SHADOW_EVOL_OWNER_TAG="$OWNER_TAG" \
    SHADOW_EVOL_ALLOW_PLACEHOLDER_PROXY="$ALLOW_PLACEHOLDER_PROXY" \
    SHADOW_EVOL_RUN_NAMESPACE="${NAMESPACE}test${HARNESS_SUFFIX}" \
    SHADOW_EVOL_EVAL_NUM_ENVS="$NUM_ENVS" \
    OSWORLD_INFRA_RETRIES="$INFRA_RETRIES" \
    PYTHONPATH="$ROOT:$PYTHONPATH" "$DOTENV" -f "$ENV_FILE" run -- \
        bash -c '
            : "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from .env}"
            : "${OPENAI_BASE_URL:?OPENAI_BASE_URL is missing from .env}"
            : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is missing from .env}"
            # Only demanded when a judge actually reads it: the OpenRouter
            # routing passes *_API_KEY_ENV=OPENROUTER_API_KEY.
            if [[ "${OSWORLD_EVAL_MODEL_API_KEY_ENV:-ANTHROPIC_API_KEY}" == ANTHROPIC_API_KEY \
                  || "${OSWORLD_USER_SIM_API_KEY_ENV:-ANTHROPIC_API_KEY}" == ANTHROPIC_API_KEY ]]; then
                : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required for grader + user simulator}"
            fi
            export OSWORLD_EVAL_MODEL_PROVIDER="${OSWORLD_EVAL_MODEL_PROVIDER:-anthropic}"
            export OSWORLD_EVAL_MODEL_NAME="${OSWORLD_EVAL_MODEL_NAME:-claude-sonnet-4-6}"
            export OSWORLD_EVAL_MODEL_API_KEY_ENV="${OSWORLD_EVAL_MODEL_API_KEY_ENV:-ANTHROPIC_API_KEY}"
            export OSWORLD_EVAL_MODEL_BASE_URL="${OSWORLD_EVAL_MODEL_BASE_URL:-https://api.anthropic.com}"
            export OSWORLD_USER_SIM_PROVIDER="${OSWORLD_USER_SIM_PROVIDER:-anthropic}"
            export OSWORLD_USER_SIM_MODEL="${OSWORLD_USER_SIM_MODEL:-claude-sonnet-4-6}"
            export OSWORLD_USER_SIM_API_KEY_ENV="${OSWORLD_USER_SIM_API_KEY_ENV:-ANTHROPIC_API_KEY}"
            export OSWORLD_USER_SIM_BASE_URL="${OSWORLD_USER_SIM_BASE_URL:-https://api.anthropic.com}"
            exec "$@"
        ' shadow-evol-test "$@"
}

CMD=(
    "$PYTHON" -m agent_s4.eval.osworld_v2
    --benchmark osworld_v2
    --harness "$HARNESS"
    --model "$MODEL"
    --task_ids "${TASK_IDS[@]}"
    --num_envs "$NUM_ENVS"
    --shadow_dir "$SHADOW_DIR"
    --result_dir "$RESULT_ROOT"
    --test_config_base_dir "$ROOT/OSWorldV2_Base/evaluation_examples"
    --region "$AWS_REGION_NAME"
    --timeout "$EVAL_TIMEOUT_COMPAT"
    --max-steps "$MAX_AGENT_STEPS"
    --no-time-limit
)
[[ "$NO_TOOL" == "1" ]] && CMD+=(--no_tool)

# The egress proxy exists so tasks that fetch from the public web look like they
# come from the expected country. dataimpulse.json is still a placeholder, so
# with the proxy ON those tasks fail outright -- the preflight names them. When
# the orchestrator already sits in the right country the proxy is pure overhead,
# and disabling it turns those forced failures into real measurements.
case "${SHADOW_EVOL_DISABLE_PROXY:-0}" in
    1|true|True|TRUE|yes|on) CMD+=(--disable-proxy) ;;
    0|false|False|FALSE|no|off|"") CMD+=(--enable-proxy) ;;
    *) printf 'ERROR: SHADOW_EVOL_DISABLE_PROXY=%s is not a boolean.\n' \
           "$SHADOW_EVOL_DISABLE_PROXY" >&2; exit 2 ;;
esac

# An orchestrator outside the benchmark VPC cannot route the private 172.31.x
# address the AWS provider hands back by default, and the failure is silent:
# every task sits in vm_allocation until it times out and records an imputed
# zero. The evolution path gets this flag from ubuntu_vm.py, which translates
# SHADOW_EVOL_USE_PUBLIC_IP into --use-public-ip; this script builds its own
# argv and so has to do the same translation itself. Default off, so an
# in-VPC run keeps private addressing and past results stay reproducible.
case "${SHADOW_EVOL_USE_PUBLIC_IP:-false}" in
    1|true|True|TRUE|yes|on) CMD+=(--use-public-ip) ;;
    0|false|False|FALSE|no|off|"") ;;
    *) printf 'ERROR: SHADOW_EVOL_USE_PUBLIC_IP=%s is not a boolean.\n' \
           "$SHADOW_EVOL_USE_PUBLIC_IP" >&2; exit 2 ;;
esac

printf 'OSWorld V2 held-out eval (%s tasks from %s)\n' "${#TASK_IDS[@]}" "$TASKS_FILE"
printf '  variant:            %s (%s)\n' "$VARIANT_ID" "$TAG"
printf '  model / harness:    %s / %s\n' "$MODEL" "$HARNESS"
printf '  concurrency:        %s VMs\n' "$NUM_ENVS"
printf '  agent budget:       %s steps, no episode wall clock\n' "$MAX_AGENT_STEPS"
printf '  infra retries:      %s\n' "$INFRA_RETRIES"
printf '  result root:        %s\n' "$RESULT_ROOT"
if [[ "$ALLOW_PLACEHOLDER_PROXY" == "1" ]]; then
    printf '  proxy:              UNCONFIGURED; the preflight below enumerates\n'
    printf '                      the tasks that cannot pass without egress\n'
fi
printf '  command:'; printf ' %q' "${CMD[@]}"; printf '\n'

[[ "$DRY_RUN" == "1" ]] && exit 0
if [[ "$CONFIRMED" != "1" ]]; then
    printf 'ERROR: live execution requires --yes.\n' >&2
    exit 2
fi

SG_ADDR_MODE="${SHADOW_EVOL_USE_PUBLIC_IP:-false}"
AWS_REGION="$AWS_REGION_NAME" \
AWS_DEFAULT_REGION="$AWS_REGION_NAME" \
"$PYTHON" - "$AWS_SECURITY_GROUP" "$AWS_REGION_NAME" "$SG_ADDR_MODE" <<'SGCHECK'
import ipaddress, subprocess, sys, urllib.request
import boto3

group_id, region, use_public_ip = sys.argv[1], sys.argv[2], sys.argv[3] in {
    "1", "true", "True", "TRUE", "yes", "on"}
CONTROL_PORTS = (5000, 9222)
group = boto3.client("ec2", region_name=region).describe_security_groups(
    GroupIds=[group_id])["SecurityGroups"][0]
if use_public_ip:
    address = urllib.request.urlopen(
        "https://checkip.amazonaws.com", timeout=15).read().decode().strip()
    described = f"this host's egress address {address}"
else:
    address = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
    described = f"this host's private address {address}"
here = ipaddress.ip_address(address)
for port in CONTROL_PORTS:
    for permission in group["IpPermissions"]:
        low, high = permission.get("FromPort"), permission.get("ToPort")
        if low is None or not (low <= port <= high):
            continue
        if any(here in ipaddress.ip_network(e["CidrIp"])
               for e in permission.get("IpRanges", [])):
            print(f"port {port} reachable: {group['GroupName']} admits {described}")
            break
    else:
        raise SystemExit(
            f"ERROR: security group {group['GroupName']} ({group_id}) does not "
            f"admit {described} on port {port}.\n"
            "       Every task would hang in setup and record a zero.\n"
            "       Point AWS_SECURITY_GROUP_ID at a group that admits it."
        )
SGCHECK

# V2-native task classes import OSWorld V2's desktop_env, which only resolves
# from a V2 clone. Use the pristine template clone rather than an island clone
# so a test run can never perturb evolution state.
cd "$ROOT/OSWorldV2_Evol"
with_prime_env "$PYTHON" -m agent_s4.osworld_v2_preflight \
    --release osworld-v2-2026.08.08 \
    --harness "$HARNESS" \
    --assets-dir "$ASSET_DIR" \
    --website-suffix "$WEBSITE_SUFFIX" \
    --proxy-config "$PROXY_CONFIG"

with_prime_env "${CMD[@]}"
