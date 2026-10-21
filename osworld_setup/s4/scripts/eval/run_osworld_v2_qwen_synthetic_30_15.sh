#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
VENV_BIN="$ROOT/.venv/bin"
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
AWS_PROFILE_NAME="${AWS_PROFILE:-}"
AWS_REGION_NAME="${AWS_REGION:-us-east-1}"
AWS_SUBNET="${AWS_SUBNET_ID:-}"
AWS_SECURITY_GROUP="${AWS_SECURITY_GROUP_ID:-}"
USE_PUBLIC_IP="${SHADOW_EVOL_USE_PUBLIC_IP:-true}"
ASSET_DIR="${OSWORLD_FILE_BASE_URL:-$ROOT/OSWorldV2_Base/cache/osworld_v2_assets}"
PROXY_CONFIG="${PROXY_CONFIG_FILE:-$ROOT/OSWorldV2_Base/evaluation_examples/settings/proxy/dataimpulse.json}"
WEBSITE_SUFFIX="${WEBSITE_HOST_SUFFIX:-site.hku.icu}"
OWNER_TAG="${SHADOW_EVOL_OWNER_TAG:-}"

# Split overrides exist for small smoke runs; formal runs keep the defaults.
TRAIN_SET="${SHADOW_EVOL_TRAIN_SET:-synthetic_train_30.json}"
SYNTHETIC_ROOT="${OSWORLD_SYNTHETIC_ROOT:-$ROOT/data/osworld_synthetic_30}"
# The synthetic tasks are downloaded separately as data/<name>.tar.xz (see
# data/README.md); unpack them on first use.
if [[ ! -e "$SYNTHETIC_ROOT" && -f "$SYNTHETIC_ROOT.tar.xz" ]]; then
    tar -xJf "$SYNTHETIC_ROOT.tar.xz" -C "$(dirname "$SYNTHETIC_ROOT")"
fi
PROMPT_TEMPLATE="$ROOT/prompts/synthetic_trajectories.md"
VAL_SET="${SHADOW_EVOL_VAL_SET:-osworld_v2_val_15.json}"
RUN_NAMESPACE="${SHADOW_EVOL_RUN_NAMESPACE:-qwen_synthetic30v15}"
ISLAND_NAME="$RUN_NAMESPACE"
MAX_EVOLUTION_STEPS="${SHADOW_EVOL_MAX_STEPS:-2}"

# One harness drives both the Builder and the task agent. OpenCode is a
# separate experimental condition (native tools, not Prime's IPython tool):
# give it its own namespace and never resume a Prime namespace with it.
HARNESS="${SHADOW_EVOL_HARNESS:-prime_agent}"
if [[ "$HARNESS" != prime_agent && "$HARNESS" != opencode ]]; then
    printf 'ERROR: SHADOW_EVOL_HARNESS must be prime_agent or opencode, got %s\n' "$HARNESS" >&2
    exit 2
fi
BUILDER_HARNESS="$HARNESS"
EVAL_HARNESS="$HARNESS"
BUILDER_TIMEOUT="${SHADOW_EVOL_BUILDER_TIMEOUT:-7200}"
if [[ "$HARNESS" == prime_agent ]]; then
    if [[ -z "${SHADOW_PRIME_AGENT_BIN:-}" && -x /opt/prime-agent/bin/prime-agent ]]; then
        export SHADOW_PRIME_AGENT_BIN=/opt/prime-agent/bin/prime-agent
    elif [[ -z "${SHADOW_PRIME_AGENT_BIN:-}" ]]; then
        export SHADOW_PRIME_AGENT_BIN="$ROOT/vm_assets/prime_agent_host/opt/prime-agent/bin/prime-agent"
    fi
    # Recorded in the run config under its historical key.
    HARNESS_BINARY="$SHADOW_PRIME_AGENT_BIN"
else
    HARNESS_BINARY="$(cd "$ROOT" && .venv/bin/python -c 'from agent_s4.harness.opencode.source import resolve_binary; print(resolve_binary())')"
fi

EVAL_TIMEOUT=7200
EVAL_MAX_AGENT_STEPS=300
EVAL_NUM_ENVS="${SHADOW_EVOL_EVAL_NUM_ENVS:-45}"
EVAL_BATCH_TIMEOUT="${SHADOW_EVOL_EVAL_BATCH_TIMEOUT:-36000}"
INFRA_RETRIES="${OSWORLD_INFRA_RETRIES:-2}"
SEED=42

SCHEDULER_SPEC="${SHADOW_EVOL_SCHEDULER_SPEC:-deterministic:0,0,0}"
PARENT_SELECTION="${SHADOW_EVOL_PARENT_SELECTION:-latest}"
# 1 gives each Builder its own V2 VM to test its code on before it finishes.
SCRATCH_VM="${SHADOW_EVOL_SCRATCH_VM:-0}"

OR_PROVIDER_ORDER="novita"
OR_ALLOW_FALLBACKS="false"
OR_ZDR="true"
OR_DATA_COLLECTION="deny"
CONTEXT_WINDOW_SOURCE="${SHADOW_PRIME_CONTEXT_WINDOW_SOURCE:-interface}"

PREFLIGHT_ONLY=0
DRY_RUN=0
CONFIRMED=0
ALLOW_PLACEHOLDER_PROXY=0
LOCAL_CHECK=0
MODEL="${SHADOW_EVOL_MODEL:-}"

usage() {
    cat <<EOF
Usage: $0 [--local-check | --dry-run | --preflight-only | --yes]
  --local-check              Offline package/config checks; print resolved command
  --dry-run, --preflight-only Also check pinned benchmark/runtime (no API/VM calls)
  --yes                      Launch seed + 2 mutation steps (Qwen builder)
  --model MODEL              Qwen model for both builder and task agent
  --namespace NAME           Independent run namespace (default: $RUN_NAMESPACE)
  --aws-profile NAME         AWS profile
  --region REGION            AWS region
  --allow-placeholder-proxy  Diagnostic mode, inherited from older runs
Environment: SHADOW_EVOL_HARNESS=prime_agent|opencode, SHADOW_EVOL_MAX_STEPS=1 or 2,
  SHADOW_EVOL_EVAL_NUM_ENVS=45,
  SHADOW_EVOL_BUILDER_TIMEOUT=7200, SHADOW_EVOL_EVAL_BATCH_TIMEOUT=36000.
Synthetic training scores are ungraded; only the older validation scores measure performance.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes) CONFIRMED=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --local-check) LOCAL_CHECK=1; shift ;;
        --allow-placeholder-proxy) ALLOW_PLACEHOLDER_PROXY=1; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --namespace) RUN_NAMESPACE="$2"; ISLAND_NAME="$2"; shift 2 ;;
        --namespace=*) RUN_NAMESPACE="${1#*=}"; ISLAND_NAME="$RUN_NAMESPACE"; shift ;;
        --aws-profile) AWS_PROFILE_NAME="$2"; shift 2 ;;
        --aws-profile=*) AWS_PROFILE_NAME="${1#*=}"; shift ;;
        --region) AWS_REGION_NAME="$2"; shift 2 ;;
        --region=*) AWS_REGION_NAME="${1#*=}"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
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

for required_dir in \
    "$ROOT/OSWorldV2_Evol" \
    "$ROOT/OSWorldV2_Base" \
    "$ROOT/seeds/combined" \
    "$ASSET_DIR"; do
    if [[ ! -d "$required_dir" ]]; then
        printf 'ERROR: required directory not found: %s\n' "$required_dir" >&2
        exit 1
    fi
done
for required_file in \
    "$ENV_FILE" \
    "$PROXY_CONFIG" \
    "$ROOT/eval_sets/$TRAIN_SET" \
    "$ROOT/eval_sets/$VAL_SET" \
    "$ROOT/eval_sets/osworld_v2_train_50.json" \
    "$ROOT/eval_sets/osworld_v2_val_25.json" \
    "$ROOT/eval_sets/osworld_v2_test.json" \
    "$PROMPT_TEMPLATE"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'ERROR: required file not found: %s\n' "$required_file" >&2
        exit 1
    fi
done

if [[ ! -d "$SYNTHETIC_ROOT" ]]; then
    printf 'ERROR: synthetic tasks not found at %s\n  Download them as described in %s/data/README.md, or set OSWORLD_SYNTHETIC_ROOT.\n' \
        "$SYNTHETIC_ROOT" "$ROOT" >&2
    exit 1
fi
if ! python3 "$ROOT/scripts/eval/prepare_osworld_synthetic.py" --root "$SYNTHETIC_ROOT" --check; then
    printf 'Configure the synthetic tasks for your mock websites first:\n  python3 scripts/eval/prepare_osworld_synthetic.py [--scheme http] [--origin APP=URL ...]\n' >&2
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    configured_model="$(dotenv -f "$ENV_FILE" get OPENROUTER_MODEL 2>/dev/null || true)"
    if [[ -n "$configured_model" ]]; then
        MODEL="openrouter/${configured_model#openrouter/}"
    else
        configured_model="$(dotenv -f "$ENV_FILE" get OPENAI_MODEL 2>/dev/null || true)"
        if [[ -z "$configured_model" ]]; then
            printf 'ERROR: neither OPENROUTER_MODEL nor OPENAI_MODEL is configured in %s\n' "$ENV_FILE" >&2
            exit 1
        fi
        MODEL="openai/$configured_model"
    fi
fi
OPENAI_MODEL_FROM_ENV="$(dotenv -f "$ENV_FILE" get OPENAI_MODEL 2>/dev/null || true)"
# An allowlist, not a format check: this experiment's results are only
# comparable across steps if every step ran the same family of model, and a
# typo'd slug would otherwise be discovered only after a VM fleet had run.
# openai/<name> is a self-hosted deployment behind OPENAI_BASE_URL; its name is
# whatever OPENAI_MODEL in .env says.
if [[ "$MODEL" != openrouter/qwen/qwen3.8-27b \
      && "$MODEL" != openai/*/Qwen/Qwen3.8 \
      && "$MODEL" != "openai/${OPENAI_MODEL_FROM_ENV:-}" ]]; then
    printf 'ERROR: unsupported Qwen evaluator model slug: %s\n' "$MODEL" >&2
    exit 2
fi
BUILDER_MODEL="$MODEL"
for name in MAX_EVOLUTION_STEPS BUILDER_TIMEOUT EVAL_BATCH_TIMEOUT EVAL_NUM_ENVS; do
    if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'ERROR: %s must be a positive integer, got %s\n' "$name" "${!name}" >&2
        exit 2
    fi
done
if [[ ! "$INFRA_RETRIES" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: OSWORLD_INFRA_RETRIES must be a non-negative integer, got %s\n' \
        "$INFRA_RETRIES" >&2
    exit 2
fi
if [[ "$PARENT_SELECTION" != "tournament" && "$PARENT_SELECTION" != "latest" ]]; then
    printf 'ERROR: SHADOW_EVOL_PARENT_SELECTION must be tournament or latest, got %s\n' \
        "$PARENT_SELECTION" >&2
    exit 2
fi
if [[ "$SCRATCH_VM" != 0 && "$SCRATCH_VM" != 1 ]]; then
    printf 'ERROR: SHADOW_EVOL_SCRATCH_VM must be 0 or 1, got %s\n' "$SCRATCH_VM" >&2
    exit 2
fi
if [[ "$SCRATCH_VM" == 1 && -z "$OWNER_TAG" ]]; then
    # The account can only terminate instances launched with our Owner tag.
    printf 'ERROR: SHADOW_EVOL_SCRATCH_VM=1 needs SHADOW_EVOL_OWNER_TAG\n' >&2
    exit 2
fi
if [[ ! "$SCHEDULER_SPEC" =~ ^deterministic(:[0-9]+,[0-9]+,[0-9]+)?$ \
      && ! "$SCHEDULER_SPEC" =~ ^probabilistic:.+$ ]]; then
    printf 'ERROR: SHADOW_EVOL_SCHEDULER_SPEC is not a scheduler spec: %s\n' \
        "$SCHEDULER_SPEC" >&2
    exit 2
fi
if [[ ! "$RUN_NAMESPACE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'ERROR: namespace must contain only letters, digits, underscores, or hyphens: %s\n' \
        "$RUN_NAMESPACE" >&2
    exit 2
fi

python3 - "$EVAL_TIMEOUT" "$EVAL_BATCH_TIMEOUT" "$EVAL_NUM_ENVS" 45 <<'PY'
import math
import sys

per_task, batch, num_envs, tasks = (int(x) for x in sys.argv[1:5])
waves = math.ceil(tasks / num_envs)
floor = waves * per_task
if batch <= floor:
    raise SystemExit(
        f"ERROR: the batch watchdog ({batch}s) is not above the {waves} wave(s) "
        f"of {num_envs} environment(s) that {tasks} tasks at a {per_task}s "
        f"per-task cap need ({floor}s of agent time alone, before provisioning, "
        "grading and infra retries).\n"
        "       Raise SHADOW_EVOL_EVAL_BATCH_TIMEOUT, raise the environment "
        "count, or lower the per-task cap."
    )
print(
    f"Per-task cap {per_task}s x {waves} wave(s) = {floor}s of agent time fits "
    f"inside the {batch}s batch watchdog ({batch - floor}s of slack)."
)
PY

if [[ "$HARNESS" == prime_agent ]]; then
    (cd "$ROOT" && .venv/bin/python -c 'from agent_s4.harness.prime.builder import resolve_prime_agent_binary; print("Prime builder runtime:", resolve_prime_agent_binary())')
else
    (cd "$ROOT" && .venv/bin/python -c 'from agent_s4.harness.opencode.evaluator import _resolve_runtime_cache; from agent_s4.harness.opencode.source import validate_runtime_cache; validate_runtime_cache(_resolve_runtime_cache())')
    printf 'OpenCode builder runtime: %s (VM cache validated)\n' "$HARNESS_BINARY"
fi
# Match the island argument exactly. `pgrep -f "agent_s4.cli.*$RUN_NAMESPACE"`
# is a substring test, so a namespace that is a PREFIX of another one aborts on
# the other one's driver: RUN_NAMESPACE=foo matched a live foo_scratch run
# and refused to start, although those are
# separate islands with separate databases. The driver spells the island as
# `--island name=<ns>,...`, so anchoring on `name=<ns>,` is exact; the trailing
# comma is what makes the prefix stop matching. pgrep -f also matches the shell
# running this script, hence the [n] character class.
if ps -eo args= | grep -q "agent_s4[.]cli run .*[n]ame=$RUN_NAMESPACE,"; then
    printf 'ERROR: an orchestrator for %s is already running.\n' "$RUN_NAMESPACE" >&2
    exit 1
fi

if [[ "$LOCAL_CHECK" == 0 && "$PREFLIGHT_ONLY" == 0 && "$DRY_RUN" == 0 && "$CONFIRMED" == 1 ]]; then
AWS_REGION="$AWS_REGION_NAME" \
AWS_DEFAULT_REGION="$AWS_REGION_NAME" \
python - "$AWS_SECURITY_GROUP" "$AWS_REGION_NAME" "$USE_PUBLIC_IP" <<'SGCHECK'
import datetime
import glob
import ipaddress
import json
import os
import subprocess
import sys
import urllib.request

import boto3
import botocore.exceptions

group_id, region, use_public_ip = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
CONTROL_PORT = 5000
VPC_CIDR = "172.31.0.0/16"
SSO_LOGIN_HINT = (
    "       Refresh the session first, in a terminal that can open a browser:\n"
    "           aws sso login --profile %s"
    % os.environ.get("AWS_PROFILE", "<profile>")
)

try:
    ec2 = boto3.client("ec2", region_name=region)
    group = ec2.describe_security_groups(GroupIds=[group_id])["SecurityGroups"][0]
except botocore.exceptions.TokenRetrievalError as exc:
    raise SystemExit(
        f"ERROR: boto3 cannot mint credentials for this run: {exc}\n"
        "       Note the `aws` CLI may still work -- it caches minted role\n"
        "       credentials that outlive the SSO token boto3 needs -- so a\n"
        "       working CLI is not evidence this run can authenticate.\n"
        f"{SSO_LOGIN_HINT}"
    ) from None
except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as exc:
    detail = str(exc)[:300]
    hint = SSO_LOGIN_HINT if "sso" in detail.lower() or "Token" in detail else ""
    raise SystemExit(f"ERROR: could not read {group_id}: {detail}\n{hint}") from None

tokens = []
for path in glob.glob(os.path.expanduser("~/.aws/sso/cache/*.json")):
    try:
        with open(path) as stream:
            cached = json.load(stream)
    except (OSError, json.JSONDecodeError):
        continue
    if "startUrl" not in cached or not cached.get("expiresAt"):
        continue
    try:
        expiry = datetime.datetime.fromisoformat(
            cached["expiresAt"].replace("Z", "+00:00")
        )
    except ValueError:
        continue
    tokens.append((expiry, bool(cached.get("refreshToken"))))
if tokens:
    expiry, renewable = max(tokens)
    hours = (expiry - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 3600
    if renewable:
        print(
            f"AWS SSO access token expires in {hours:.1f}h and carries a "
            "refresh token, so botocore renews it for the life of the SSO "
            "session."
        )
    else:
        print(
            f"WARNING: the AWS SSO token expires in {hours:.1f}h and has no "
            "refresh token, so it is a hard stop.\n"
            "         Every task still running past it fails VM allocation and "
            "records a zero that counts as fitness. Re-run `aws sso login` "
            "before it lapses."
        )

if use_public_ip:
    try:
        address = urllib.request.urlopen(
            "https://checkip.amazonaws.com", timeout=15,
        ).read().decode().strip()
    except OSError as exc:
        raise SystemExit(f"ERROR: could not determine this host's egress address: {exc}")
    described = f"this host's egress address {address}"
else:
    address = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True,
    ).stdout.split()[0]
    described = f"this host's private address {address}"
    if ipaddress.ip_address(address) not in ipaddress.ip_network(VPC_CIDR):
        raise SystemExit(
            f"ERROR: the run is configured for private VM addressing, but this "
            f"host ({address}) is not inside {VPC_CIDR}, so it cannot route to "
            "the VMs.\n"
            "       Set SHADOW_EVOL_USE_PUBLIC_IP=true for an off-VPC host."
        )

for permission in group["IpPermissions"]:
    low, high = permission.get("FromPort"), permission.get("ToPort")
    if low is None or not (low <= CONTROL_PORT <= high):
        continue
    for entry in permission.get("IpRanges", []):
        if ipaddress.ip_address(address) in ipaddress.ip_network(entry["CidrIp"]):
            print(
                f"Control server port {CONTROL_PORT} is reachable: "
                f"{group['GroupName']} admits {described} via {entry['CidrIp']}"
            )
            raise SystemExit(0)

raise SystemExit(
    f"ERROR: security group {group['GroupName']} ({group_id}) does not admit "
    f"{described} on port {CONTROL_PORT}.\n"
    "       Every task would hang in vm_allocation, burn its retries, and be "
    "recorded as a zero score.\n"
    "       Add an ingress rule for that address, or point "
    "AWS_SECURITY_GROUP_ID at a group that already has one."
)
SGCHECK
fi


with_run_env() {
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
    SHADOW_EVOL_RUN_NAMESPACE="$RUN_NAMESPACE" \
    SHADOW_EVOL_SYNTHETIC_ROOT_PIN="$SYNTHETIC_ROOT" \
    SHADOW_EVOL_MODEL_RESOLVED="$MODEL" \
    SHADOW_EVOL_EVAL_NUM_ENVS="$EVAL_NUM_ENVS" \
    SHADOW_EVOL_EVAL_BATCH_TIMEOUT="$EVAL_BATCH_TIMEOUT" \
    SHADOW_PRIME_CONTEXT_WINDOW_SOURCE="$CONTEXT_WINDOW_SOURCE" \
    OSWORLD_INFRA_RETRIES="$INFRA_RETRIES" \
    SHADOW_EVOL_INFRA_RETRIES_PIN="$INFRA_RETRIES" \
    SHADOW_EVOL_USE_PUBLIC_IP="$USE_PUBLIC_IP" \
    SHADOW_EVOL_PINNED_OPENROUTER_PROVIDER_ORDER="$OR_PROVIDER_ORDER" \
    SHADOW_EVOL_PINNED_OPENROUTER_ALLOW_FALLBACKS="$OR_ALLOW_FALLBACKS" \
    SHADOW_EVOL_PINNED_OPENROUTER_ZDR="$OR_ZDR" \
    SHADOW_EVOL_PINNED_OPENROUTER_DATA_COLLECTION="$OR_DATA_COLLECTION" \
    dotenv -f "$ENV_FILE" run -- \
        bash -c '
            export OSWORLD_SYNTHETIC_ROOT="$SHADOW_EVOL_SYNTHETIC_ROOT_PIN"
            # .env may carry its own OSWORLD_INFRA_RETRIES; the recorded value wins.
            export OSWORLD_INFRA_RETRIES="$SHADOW_EVOL_INFRA_RETRIES_PIN"
            : "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from .env}"
            : "${OPENAI_BASE_URL:?OPENAI_BASE_URL is missing from .env}"
            : "${OPENAI_MODEL:?OPENAI_MODEL is missing from .env}"
            if [[ "$SHADOW_EVOL_MODEL_RESOLVED" == openrouter/* ]]; then
                : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is missing from .env}"
            elif [[ "openai/$OPENAI_MODEL" != "$SHADOW_EVOL_MODEL_RESOLVED" ]]; then
                printf "ERROR: model %s does not match OPENAI_MODEL %s\n" \
                    "$SHADOW_EVOL_MODEL_RESOLVED" "$OPENAI_MODEL" >&2
                exit 1
            fi
            # Routing pins arrive under their own names because `dotenv run`
            # overrides caller-exported variables; applying them here, after
            # the env file is loaded, is what makes a pin hold.
            for pinned in PROVIDER_ORDER ALLOW_FALLBACKS ZDR DATA_COLLECTION; do
                pin_var="SHADOW_EVOL_PINNED_OPENROUTER_$pinned"
                if [[ -n "${!pin_var:-}" ]]; then
                    export "OPENROUTER_$pinned=${!pin_var}"
                fi
            done
            export OSWORLD_EVAL_MODEL_PROVIDER="${OSWORLD_EVAL_MODEL_PROVIDER:-anthropic}"
            export OSWORLD_EVAL_MODEL_NAME="${OSWORLD_EVAL_MODEL_NAME:-claude-sonnet-4-6}"
            export OSWORLD_EVAL_MODEL_API_KEY_ENV="${OSWORLD_EVAL_MODEL_API_KEY_ENV:-ANTHROPIC_API_KEY}"
            if [[ "$OSWORLD_EVAL_MODEL_PROVIDER" == "anthropic" ]]; then
                : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required for the OSWorld grader}"
                export OSWORLD_EVAL_MODEL_BASE_URL="${OSWORLD_EVAL_MODEL_BASE_URL:-https://api.anthropic.com}"
            else
                : "${OSWORLD_EVAL_MODEL_BASE_URL:?OSWORLD_EVAL_MODEL_BASE_URL is required for a non-Anthropic grader}"
            fi
            export OSWORLD_USER_SIM_PROVIDER="${OSWORLD_USER_SIM_PROVIDER:-anthropic}"
            export OSWORLD_USER_SIM_MODEL="${OSWORLD_USER_SIM_MODEL:-claude-sonnet-4-6}"
            export OSWORLD_USER_SIM_API_KEY_ENV="${OSWORLD_USER_SIM_API_KEY_ENV:-ANTHROPIC_API_KEY}"
            if [[ "$OSWORLD_USER_SIM_PROVIDER" == "anthropic" ]]; then
                : "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required for the OSWorld user simulator}"
                export OSWORLD_USER_SIM_BASE_URL="${OSWORLD_USER_SIM_BASE_URL:-https://api.anthropic.com}"
            else
                : "${OSWORLD_USER_SIM_BASE_URL:?OSWORLD_USER_SIM_BASE_URL is required for a non-Anthropic user simulator}"
            fi
            exec "$@"
        ' shadow-evol-qwen-synthetic "$@"
}

cd "$ROOT"

PREFLIGHT_TASK_ARGS=()
while IFS= read -r task_id; do
    PREFLIGHT_TASK_ARGS+=(--task-id "$task_id")
done < <(python3 - "$ROOT/eval_sets/$TRAIN_SET" "$ROOT/eval_sets/$VAL_SET" <<'TASKIDS'
import json, sys
for path in sys.argv[1:]:
    print("\n".join(json.load(open(path))))
TASKIDS
)
if [[ "$LOCAL_CHECK" == 0 ]]; then
    with_run_env python -m agent_s4.osworld_v2_preflight \
        --release osworld-v2-2026.08.08 --harness "$EVAL_HARNESS" \
        --assets-dir "$ASSET_DIR" --website-suffix "$WEBSITE_SUFFIX" \
        --proxy-config "$PROXY_CONFIG" "${PREFLIGHT_TASK_ARGS[@]}"
fi

# Keep --no-scratch-vm at its original position: the run config pins the
# command, so moving it would read as drift for existing namespaces.
SCRATCH_VM_ARGS=()
[[ "$SCRATCH_VM" == 1 ]] || SCRATCH_VM_ARGS=(--no-scratch-vm)
CMD=(
    python -m agent_s4.cli run
    --benchmark osworld_v2
    "${SCRATCH_VM_ARGS[@]}"
    --island "name=$ISLAND_NAME,evaluator_harness=$EVAL_HARNESS,eval_model=$MODEL"
    --seed-dir "$ROOT/seeds/combined"
    --prompt-template "$PROMPT_TEMPLATE"
    --builder-model "$BUILDER_MODEL"
    --builder-harness "$BUILDER_HARNESS"
    --builder-timeout "$BUILDER_TIMEOUT"
    --eval-timeout "$EVAL_TIMEOUT"
    --eval-max-steps "$EVAL_MAX_AGENT_STEPS"
    --train-set "$TRAIN_SET"
    --val-set "$VAL_SET"
    --allow-v2-noncanonical-splits
    --selection-signal val
    --parent-selection "$PARENT_SELECTION"
    --scheduler "$SCHEDULER_SPEC"
    --elimination-count 5
    --max-steps "$MAX_EVOLUTION_STEPS"
    --region "$AWS_REGION_NAME"
    --seed "$SEED"
)

CONFIG_FILE="$ROOT/runs/${RUN_NAMESPACE}_run_config.json"
mkdir -p "$ROOT/runs"
resolved_cmd="$(printf ' %q' "${CMD[@]}")"
resolved_cmd="${resolved_cmd# }"

xover_cadence="$(printf '%s' "$SCHEDULER_SPEC" | awk -F'[:,]' '{print $3}')"
if [[ "$SCHEDULER_SPEC" == deterministic:0,0,0 ]]; then
    schedule_story="every step is a mutation (no crossover, migration or elimination)"
elif [[ "$xover_cadence" =~ ^[1-9][0-9]*$ ]]; then
    schedule_story="crossover every $xover_cadence step(s), mutation otherwise"
else
    schedule_story="$SCHEDULER_SPEC"
fi
if [[ "$PARENT_SELECTION" == "latest" ]]; then
    parent_selection_story="latest — strict linear chain, depth $MAX_EVOLUTION_STEPS"
else
    parent_selection_story="tournament (k=2) on val_score — branching lineage"
fi
route_headline="crossover every 2 steps"
[[ "$SCHEDULER_SPEC" == deterministic:0,0,0 && "$PARENT_SELECTION" == "latest" ]] \
    && route_headline="linear route"

printf '\nOSWorld V2 evolution — Qwen/%s Builder, 2h per-task cap, %s\n' "$HARNESS" "$route_headline"
printf '  namespace:          %s\n' "$RUN_NAMESPACE"
printf '  state database:     shadow_evolution_osworld_v2_%s.db\n' "$RUN_NAMESPACE"
printf '  builder:            %s on %s\n' "$BUILDER_HARNESS" "$BUILDER_MODEL"
printf '  builder time limit: %ss\n' "$BUILDER_TIMEOUT"
printf '  evaluator:          %s on %s\n' "$EVAL_HARNESS" "$MODEL"
printf '  context window:     %s\n' "$CONTEXT_WINDOW_SOURCE"
printf '  provider (pinned):  %s (allow_fallbacks=%s, zdr=%s, data_collection=%s)\n' \
    "$OR_PROVIDER_ORDER" "$OR_ALLOW_FALLBACKS" "$OR_ZDR" "$OR_DATA_COLLECTION"
printf '  split:              synthetic train=30 (ungraded), old val=15, canonical test=33 held out\n'
printf '  schedule:           %s\n' "$schedule_story"
printf '  parent selection:   %s\n' "$parent_selection_story"
printf '  expected episodes:  %s (seed + %s children, each on 45 tasks)\n' \
    "$(( (MAX_EVOLUTION_STEPS + 1) * 45 ))" "$MAX_EVOLUTION_STEPS"
printf '  agent budget:       %s steps/task, %ss wall clock/task\n' \
    "$EVAL_MAX_AGENT_STEPS" "$EVAL_TIMEOUT"
printf '                      (one clock per task, shared across phases and\n'
printf '                       ASK_USER turns, started once the VM is ready)\n' 
printf '  batch watchdog:     %ss\n' "$EVAL_BATCH_TIMEOUT"
printf '  VM concurrency:     %s environments\n' "$EVAL_NUM_ENVS"
printf '  infra retries:      %s per task, each on a fresh VM\n' "$INFRA_RETRIES"
printf '  security group:     %s\n' "$AWS_SECURITY_GROUP"
printf '  VM addressing:      %s\n' \
    "$([ "$USE_PUBLIC_IP" = true ] && echo 'public (off-VPC host)' || echo 'private (in-VPC host)')"
printf '  scratch VM:         %s\n' \
    "$([ "$SCRATCH_VM" = 1 ] && echo 'one per Builder, torn down after it exits' || echo disabled)"
if [[ "$ALLOW_PLACEHOLDER_PROXY" == "1" ]]; then
    printf '  proxy:              UNCONFIGURED diagnostic mode\n'
else
    printf '  proxy:              no selected task requires proxy credentials\n'
fi
printf '  run config:         %s\n' "$CONFIG_FILE"
printf '  command:            %s\n' "$resolved_cmd"

if [[ "$LOCAL_CHECK" == "1" ]]; then
    printf '\nLocal checks passed; full preflight and VM behavior remain untested.\n'
    exit 0
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    printf '\nPreflight complete; no VM was contacted.\n'
    exit 0
fi
if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nOffline preflight passed; live AWS/network checks run only with --yes.\n'
    exit 0
fi
if [[ "$CONFIRMED" != "1" ]]; then
    printf '\nERROR: live execution requires --yes; use --dry-run to inspect it first.\n' >&2
    exit 2
fi

python3 - "$CONFIG_FILE" "$resolved_cmd" <<PY
import datetime
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

path, orchestrator_command = Path(sys.argv[1]), sys.argv[2]
command_parts = shlex.split(orchestrator_command)
budget_index = command_parts.index("--max-steps")
current = {
    "namespace": "$RUN_NAMESPACE",
    "command_without_step_budget": command_parts[:budget_index] + command_parts[budget_index + 2:],
    "synthetic_root": "$SYNTHETIC_ROOT",
    "island_name": "$ISLAND_NAME",
    "region": "$AWS_REGION_NAME",
    "aws_profile": "$AWS_PROFILE_NAME",
    "aws_subnet": "$AWS_SUBNET",
    "website_suffix": "$WEBSITE_SUFFIX",
    "asset_dir": "$ASSET_DIR",
    "proxy_config": "$PROXY_CONFIG",
    "prime_binary": "$HARNESS_BINARY",
    "seed_sha256": {str(p.relative_to(Path("$ROOT/seeds/combined"))): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("$ROOT/seeds/combined").rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
    "synthetic_score_semantics": "ungraded_constant_zero",
    "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path("$SYNTHETIC_ROOT/evaluation_examples/synthetic_assets/PACKAGE_MANIFEST.json"),
        Path("$PROMPT_TEMPLATE"), Path("$ROOT/eval_sets/$TRAIN_SET"), Path("$ROOT/eval_sets/$VAL_SET"),
        Path("$ROOT/../../agent_s4/synthetic_tasks.py")
    ]},
    "builder_harness": "$BUILDER_HARNESS",
    "builder_model": "$BUILDER_MODEL",
    "builder_timeout": $BUILDER_TIMEOUT,
    "eval_model": "$MODEL",
    "eval_timeout": $EVAL_TIMEOUT,
    "eval_max_steps": $EVAL_MAX_AGENT_STEPS,
    "eval_num_envs": $EVAL_NUM_ENVS,
    "eval_batch_timeout": $EVAL_BATCH_TIMEOUT,
    "scheduler": "$SCHEDULER_SPEC",
    "parent_selection": "$PARENT_SELECTION",
    "max_steps": $MAX_EVOLUTION_STEPS,
    "openrouter_provider_order": "$OR_PROVIDER_ORDER",
    "openrouter_allow_fallbacks": "$OR_ALLOW_FALLBACKS",
    "openrouter_zdr": "$OR_ZDR",
    "openrouter_data_collection": "$OR_DATA_COLLECTION",
    "context_window_source": "$CONTEXT_WINDOW_SOURCE",
    "aws_security_group": "$AWS_SECURITY_GROUP",
    "use_public_ip": "$USE_PUBLIC_IP",
    "infra_retries": $INFRA_RETRIES,
    "allow_placeholder_proxy": $ALLOW_PLACEHOLDER_PROXY,
    "orchestrator_command": orchestrator_command,
}
if path.is_file():
    recorded = json.loads(path.read_text())
    drift = {k: (recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v and k not in {"max_steps", "orchestrator_command", "config_history"}}
    if current["max_steps"] < recorded.get("max_steps", 0):
        drift["max_steps"] = (recorded.get("max_steps"), current["max_steps"])
    # A deliberate mid-run change (switching the model endpoint, say) is
    # sometimes what the operator wants. It still has to be named key by key
    # and is recorded in the config, because a variant evolved under one
    # configuration and one evolved under another are not comparable and the
    # run config is the only place that fact survives.
    accepted = {k.strip() for k in os.environ.get("SHADOW_EVOL_ACCEPT_CONFIG_DRIFT", "").split(",") if k.strip()}
    unaccepted = {k: v for k, v in drift.items() if k not in accepted}
    if unaccepted:
        lines = "\n".join(
            f"  {k}: recorded={was!r} now={now!r}" for k, (was, now) in sorted(unaccepted.items())
        )
        raise SystemExit(
            "ERROR: this run's configuration drifted from its first launch.\n"
            f"{lines}\n"
            "Relaunch with the original values, start a new namespace, or name\n"
            "the keys in SHADOW_EVOL_ACCEPT_CONFIG_DRIFT to accept the change\n"
            f"and have it recorded. ({path})"
        )
    stale = accepted - set(drift)
    if stale:
        # A warning, not an error: once a change is recorded it no longer
        # drifts, so a relaunch after a transient failure (an expired SSO
        # token, say) legitimately carries the same flag with nothing left to
        # accept. Failing there would block exactly the retry it should allow.
        print(
            f"NOTE: SHADOW_EVOL_ACCEPT_CONFIG_DRIFT names {sorted(stale)}, which "
            "did not drift (already recorded, or never differed)."
        )
    current["config_history"] = recorded.get("config_history", [])
    if drift:
        current["config_history"] = current["config_history"] + [{
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "steps_completed_before_change": recorded.get("max_steps"),
            "accepted_drift": {k: {"was": was, "now": now} for k, (was, now) in sorted(drift.items())},
        }]
        print("Accepted configuration change, recorded in config_history:")
        for k, (was, now) in sorted(drift.items()):
            print(f"  {k}: {was!r} -> {now!r}")
    path.write_text(json.dumps(current, indent=2) + "\n")
    if not drift:
        print(f"Configuration matches {path}")
else:
    if Path("$ROOT/shadow_evolution_osworld_v2_$RUN_NAMESPACE.db").exists():
        raise SystemExit("ERROR: existing namespace DB has no run config; choose a new namespace")
    path.write_text(json.dumps(current, indent=2) + "\n")
    print(f"Recorded run configuration to {path}")
PY

with_run_env "${CMD[@]}"
