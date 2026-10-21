#!/usr/bin/env bash
# ScienceBoard evolution on 48 of the 50 ungraded synthetic training tasks, selected
# on 15 official validation tasks. Everything that is not ScienceBoard-specific
# is copied from the OSWorld V2 Prime run: Qwen3.8-27B as both Builder and
# task agent through Prime Agent,
# crossover every 2 steps with tournament parent selection, 30 steps, 300 agent
# steps per task, 2 infra retries, temperature 1 and the 720 s idle budget.
#
# What differs, and why:
#   - per-task wall clock 600 s by default (V2: 7200 s); SHADOW_EVOL_EVAL_TIMEOUT
#     overrides it (qwen_sb_syn50v15_60m uses 3600 s)
#   - tasks run through agent_s4.eval.scienceboard_prime: one fresh EC2 per
#     task from the ScienceBoard AMI; official tasks are scored by
#     ScienceBoard's own programmatic evaluators, so no LLM grader is involved
#   - the Builder clone carries no official task tree (evaluate blocks and
#     answers) and runs under the same Landlock boundary as V2
set -euo pipefail

# ROOT (scienceboard_setup/s4) is this run's workspace: ScienceBoard checkouts,
# prompts, splits, data and every run artefact. SHARED_ROOT (osworld_setup/s4)
# holds what both benchmarks use: the venv, the Prime runtime and the seed.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHARED_ROOT="${SHADOW_EVOL_SHARED_ROOT:-$(cd "$ROOT/../../osworld_setup/s4" && pwd)}"
export SHADOW_EVOL_PROJECT_ROOT="$ROOT" SHADOW_EVOL_SHARED_ROOT="$SHARED_ROOT"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
VENV_BIN="$SHARED_ROOT/.venv/bin"
# Launch from the repository's venv (osworld_setup/s4/.venv; see README.md).
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
    SHADOW_EVOL_OWNER_TAG SB_AMI_ID
AWS_PROFILE_NAME="${AWS_PROFILE:-}"
AWS_REGION_NAME="${AWS_REGION:-us-east-1}"
AWS_SUBNET="${AWS_SUBNET_ID:-}"
AWS_SECURITY_GROUP="${AWS_SECURITY_GROUP_ID:-}"
USE_PUBLIC_IP="${SHADOW_EVOL_USE_PUBLIC_IP:-true}"
OWNER_TAG="${SHADOW_EVOL_OWNER_TAG:-}"
SB_AMI="${SB_AMI_ID:-}"
# t3.medium (4 GB) runs out of memory once VS Code starts Lean over Mathlib:
# the guest daemon stops answering and the task dies in setup.
SB_INSTANCE="${SB_INSTANCE_TYPE:-t3.large}"

# Two tasks are excluded because their own initialization fails on the guest
# every time (deterministic setup failure, no trajectory): sb_grassgis_010's
# setup_inputs.py fails its region check, and sb_chimerax_006's `open` of its
# .dx input returns a ChimeraX error.
TRAIN_SET="${SHADOW_EVOL_TRAIN_SET:-scienceboard_synthetic_train_48.json}"
VAL_SET="${SHADOW_EVOL_VAL_SET:-scienceboard_val_15.json}"
SYNTHETIC_ROOT="${SCIENCEBOARD_SYNTHETIC_ROOT:-$ROOT/data/scienceboard_synthetic_50}"
# The synthetic tasks are downloaded separately as data/<name>.tar.xz (see
# data/README.md); unpack them on first use.
if [[ ! -e "$SYNTHETIC_ROOT" && -f "$SYNTHETIC_ROOT.tar.xz" ]]; then
    tar -xJf "$SYNTHETIC_ROOT.tar.xz" -C "$(dirname "$SYNTHETIC_ROOT")"
fi
# The Builder prompt states the task agent's wall clock, so it must match
# EVAL_TIMEOUT (checked below): 600 s -> the original prompt, 3600 s -> _60m.
EVAL_TIMEOUT="${SHADOW_EVOL_EVAL_TIMEOUT:-600}"
if [[ -z "${SHADOW_EVOL_PROMPT_TEMPLATE:-}" && "$EVAL_TIMEOUT" == 600 ]]; then
    PROMPT_TEMPLATE="$ROOT/prompts/scienceboard_synthetic_trajectories.md"
else
    PROMPT_TEMPLATE="${SHADOW_EVOL_PROMPT_TEMPLATE:-$ROOT/prompts/scienceboard_synthetic_trajectories_$((EVAL_TIMEOUT / 60))m.md}"
fi
# Keep this short: the island clone is ScienceBoard_Evol_<ns>, and Prime's
# daemon socket path overflows sun_path when the clone path is too long.
RUN_NAMESPACE="${SHADOW_EVOL_RUN_NAMESPACE:-qwen_sb_syn50v15}"
ISLAND_NAME="$RUN_NAMESPACE"
MAX_EVOLUTION_STEPS="${SHADOW_EVOL_MAX_STEPS:-30}"

HARNESS=prime_agent
BUILDER_TIMEOUT="${SHADOW_EVOL_BUILDER_TIMEOUT:-7200}"
if [[ -z "${SHADOW_PRIME_AGENT_BIN:-}" && -x /opt/prime-agent/bin/prime-agent ]]; then
    export SHADOW_PRIME_AGENT_BIN=/opt/prime-agent/bin/prime-agent
elif [[ -z "${SHADOW_PRIME_AGENT_BIN:-}" ]]; then
    export SHADOW_PRIME_AGENT_BIN="$SHARED_ROOT/vm_assets/prime_agent_host/opt/prime-agent/bin/prime-agent"
fi

EVAL_MAX_AGENT_STEPS=300
EVAL_NUM_ENVS="${SHADOW_EVOL_EVAL_NUM_ENVS:-65}"
EVAL_BATCH_TIMEOUT="${SHADOW_EVOL_EVAL_BATCH_TIMEOUT:-10800}"
INFRA_RETRIES="${OSWORLD_INFRA_RETRIES:-2}"
SEED=42
PRIME_TEMPERATURE="${SHADOW_PRIME_AGENT_TEMPERATURE:-1}"
PRIME_IDLE_TIMEOUT="${SHADOW_PRIME_IDLE_TIMEOUT:-720}"

SCHEDULER_SPEC="${SHADOW_EVOL_SCHEDULER_SPEC:-deterministic:0,2,0}"
PARENT_SELECTION="${SHADOW_EVOL_PARENT_SELECTION:-tournament}"
CONTEXT_WINDOW_SOURCE="${SHADOW_PRIME_CONTEXT_WINDOW_SOURCE:-interface}"
# Default: the OpenAI-compatible deployment named by OPENAI_MODEL in .env.
MODEL="${SHADOW_EVOL_MODEL:-}"
if [[ -z "$MODEL" ]]; then
    MODEL="openai/$("$VENV_BIN/dotenv" -f "$ENV_FILE" get OPENAI_MODEL 2>/dev/null || true)"
fi
if [[ "$MODEL" == openai/ ]]; then
    printf 'ERROR: set OPENAI_MODEL in %s or SHADOW_EVOL_MODEL\n' "$ENV_FILE" >&2
    exit 1
fi

DRY_RUN=0
CONFIRMED=0
usage() {
    cat <<EOF
Usage: $0 [--dry-run | --yes] [--namespace NAME]
  --dry-run     Local checks only; print the resolved command (no AWS calls)
  --yes         Launch seed + $MAX_EVOLUTION_STEPS evolution steps
Environment: SHADOW_EVOL_EVAL_TIMEOUT (600), SHADOW_EVOL_MAX_STEPS, SHADOW_EVOL_EVAL_NUM_ENVS (65),
  SHADOW_EVOL_EVAL_BATCH_TIMEOUT (10800), OSWORLD_INFRA_RETRIES (2).
Synthetic training scores are ungraded; only validation scores measure performance.
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes) CONFIRMED=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --namespace) RUN_NAMESPACE="$2"; ISLAND_NAME="$2"; shift 2 ;;
        --namespace=*) RUN_NAMESPACE="${1#*=}"; ISLAND_NAME="$RUN_NAMESPACE"; shift ;;
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
if [[ -z "$SB_AMI" ]]; then
    printf 'ERROR: set SB_AMI_ID to a ScienceBoard AMI in your account (see scienceboard_setup/s4/README.md).\n' >&2
    exit 2
fi

for required in "$ROOT/ScienceBoard_Evol" "$ROOT/ScienceBoard_Base/tasks/VM" \
                "$SHARED_ROOT/seeds/combined" "$SYNTHETIC_ROOT"; do
    [[ -d "$required" ]] || { printf 'ERROR: missing directory %s\n' "$required" >&2; exit 1; }
done
for required in "$ENV_FILE" "$ROOT/eval_sets/$TRAIN_SET" "$ROOT/eval_sets/$VAL_SET" \
                "$ROOT/eval_sets/scienceboard_test.json" "$PROMPT_TEMPLATE"; do
    [[ -f "$required" ]] || { printf 'ERROR: missing file %s\n' "$required" >&2; exit 1; }
done
if [[ ! "$RUN_NAMESPACE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ || ${#RUN_NAMESPACE} -gt 23 ]]; then
    printf 'ERROR: namespace must be <=23 chars of letters, digits, _ or -: %s\n' \
        "$RUN_NAMESPACE" >&2
    exit 2
fi
for name in MAX_EVOLUTION_STEPS BUILDER_TIMEOUT EVAL_BATCH_TIMEOUT EVAL_NUM_ENVS EVAL_TIMEOUT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { printf 'ERROR: %s must be a positive integer\n' "$name" >&2; exit 2; }
done

grep -q "Each task agent has a $((EVAL_TIMEOUT / 60))-minute wall clock" "$PROMPT_TEMPLATE" || {
    printf 'ERROR: %s does not state the %s-minute task wall clock\n' "$PROMPT_TEMPLATE" "$((EVAL_TIMEOUT / 60))" >&2
    exit 1
}

if [[ ! -d "$SYNTHETIC_ROOT" ]]; then
    printf 'ERROR: synthetic tasks not found at %s\n  Download them as described in %s/data/README.md, or set SCIENCEBOARD_SYNTHETIC_ROOT.\n' \
        "$SYNTHETIC_ROOT" "$ROOT" >&2
    exit 1
fi
# Splits: every task must resolve, train/val/test must be disjoint, and val
# must be official ScienceBoard tasks (scored), train the ungraded bank.
SCIENCEBOARD_SYNTHETIC_ROOT="$SYNTHETIC_ROOT" "$VENV_BIN/python" - \
    "$ROOT" "$TRAIN_SET" "$VAL_SET" <<'PY'
import json, sys
from pathlib import Path
from agent_s4.eval.scienceboard_prime import build_task_index
root, train_name, val_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
index = build_task_index(root / "ScienceBoard_Base/tasks/VM")
load = lambda name: json.loads((root / "eval_sets" / name).read_text())
train, val, test = load(train_name), load(val_name), load("scienceboard_test.json")
missing = [t for t in train + val if t not in index]
if missing:
    raise SystemExit(f"ERROR: unknown task ids: {missing}")
if set(train) & set(val) or (set(train) | set(val)) & set(test):
    raise SystemExit("ERROR: train/val/test splits overlap")
if not all(t.startswith("sb_") for t in train) or any(t.startswith("sb_") for t in val):
    raise SystemExit("ERROR: train must be the synthetic bank and val official tasks")
print(f"Splits OK: train={len(train)} synthetic (ungraded), val={len(val)} official, "
      f"test={len(test)} held out; synthetic package hashes verified")
PY

(cd "$ROOT" && "$VENV_BIN/python" -c 'from agent_s4.harness.prime.builder import resolve_prime_agent_binary; print("Prime builder runtime:", resolve_prime_agent_binary())')
(cd "$ROOT" && "$VENV_BIN/python" -c 'from agent_s4.harness.prime.evaluator import _resolve_runtime_cache; from agent_s4.harness.prime.source import validate_prime_agent_runtime_cache; validate_prime_agent_runtime_cache(_resolve_runtime_cache()); print("Prime VM runtime cache validated")')
(cd "$ROOT/ScienceBoard_Evol" && git describe --tags --exact-match >/dev/null) || {
    printf 'ERROR: ScienceBoard_Evol HEAD carries no tag; template provenance needs one\n' >&2
    exit 1
}

if ps -eo args= | grep -q "agent_s4[.]cli run .*[n]ame=$RUN_NAMESPACE,"; then
    printf 'ERROR: an orchestrator for %s is already running.\n' "$RUN_NAMESPACE" >&2
    exit 1
fi

CMD=(
    python -m agent_s4.cli run
    --benchmark scienceboard
    --no-scratch-vm
    --island "name=$ISLAND_NAME,evaluator_harness=$HARNESS,eval_model=$MODEL"
    --seed-dir "$SHARED_ROOT/seeds/combined"
    --prompt-template "$PROMPT_TEMPLATE"
    --builder-model "$MODEL"
    --builder-harness "$HARNESS"
    --builder-timeout "$BUILDER_TIMEOUT"
    --eval-timeout "$EVAL_TIMEOUT"
    --eval-max-steps "$EVAL_MAX_AGENT_STEPS"
    --train-set "$TRAIN_SET"
    --val-set "$VAL_SET"
    --selection-signal val
    --parent-selection "$PARENT_SELECTION"
    --scheduler "$SCHEDULER_SPEC"
    --elimination-count 5
    --max-steps "$MAX_EVOLUTION_STEPS"
    --region "$AWS_REGION_NAME"
    --seed "$SEED"
)
resolved_cmd="$(printf ' %q' "${CMD[@]}")"
resolved_cmd="${resolved_cmd# }"
CONFIG_FILE="$ROOT/runs/${RUN_NAMESPACE}_run_config.json"

printf '\nScienceBoard evolution — Qwen/Prime Builder + task agent, %s min per-task cap\n' "$((EVAL_TIMEOUT / 60))"
printf '  namespace:          %s\n' "$RUN_NAMESPACE"
printf '  state database:     shadow_evolution_scienceboard_%s.db\n' "$RUN_NAMESPACE"
printf '  model:              %s (builder and task agent, Prime Agent)\n' "$MODEL"
printf '  split:              synthetic train=48 (ungraded), official val=15, test=94 held out\n'
printf '  schedule:           %s, parent selection %s, %s steps\n' \
    "$SCHEDULER_SPEC" "$PARENT_SELECTION" "$MAX_EVOLUTION_STEPS"
printf '  agent budget:       %s steps/task, %ss wall clock/task (after setup)\n' \
    "$EVAL_MAX_AGENT_STEPS" "$EVAL_TIMEOUT"
printf '  prime:              temperature %s, idle timeout %ss, context window %s\n' \
    "$PRIME_TEMPERATURE" "$PRIME_IDLE_TIMEOUT" "$CONTEXT_WINDOW_SOURCE"
printf '  VMs:                %s x %s, %s concurrent, %s infra retries, batch watchdog %ss\n' \
    "$SB_AMI" "$SB_INSTANCE" "$EVAL_NUM_ENVS" "$INFRA_RETRIES" "$EVAL_BATCH_TIMEOUT"
printf '  security group:     %s (public IP: %s)\n' "$AWS_SECURITY_GROUP" "$USE_PUBLIC_IP"
printf '  run config:         %s\n' "$CONFIG_FILE"
printf '  command:            %s\n' "$resolved_cmd"

if [[ "$DRY_RUN" == 1 ]]; then
    printf '\nLocal checks passed; live AWS checks run only with --yes.\n'
    exit 0
fi
if [[ "$CONFIRMED" != 1 ]]; then
    printf '\nERROR: live execution requires --yes; use --dry-run to inspect it first.\n' >&2
    exit 2
fi

# The VMs must accept this host on the control daemon (5000) and on the app
# REST port the evaluators read (8000); otherwise every task burns its retries
# in allocation or scores an unreachable app as a zero.
"$VENV_BIN/python" - \
    "$AWS_SECURITY_GROUP" "$AWS_REGION_NAME" "$SB_AMI" <<'SGCHECK'
import ipaddress, sys, urllib.request
import boto3, botocore.exceptions
group_id, region, ami = sys.argv[1:4]
try:
    ec2 = boto3.client("ec2", region_name=region)
    group = ec2.describe_security_groups(GroupIds=[group_id])["SecurityGroups"][0]
    ec2.describe_images(ImageIds=[ami])
except botocore.exceptions.TokenRetrievalError as exc:
    raise SystemExit(f"ERROR: boto3 cannot mint credentials ({exc}); run aws sso login") from None
address = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=15).read().decode().strip()
for port in (5000, 8000):
    ok = any(
        p.get("FromPort") is not None and p["FromPort"] <= port <= p["ToPort"]
        and any(ipaddress.ip_address(address) in ipaddress.ip_network(r["CidrIp"])
                for r in p.get("IpRanges", []))
        for p in group["IpPermissions"]
    )
    if not ok:
        raise SystemExit(f"ERROR: {group_id} does not admit {address} on port {port}")
print(f"Security group admits {address} on 5000 and 8000; AMI {ami} visible")
SGCHECK

python3 - "$CONFIG_FILE" "$resolved_cmd" <<PY
import datetime, hashlib, json, os, shlex, sys
from pathlib import Path
path, orchestrator_command = Path(sys.argv[1]), sys.argv[2]
parts = shlex.split(orchestrator_command)
i = parts.index("--max-steps")
current = {
    "namespace": "$RUN_NAMESPACE",
    "benchmark": "scienceboard",
    "command_without_step_budget": parts[:i] + parts[i + 2:],
    "synthetic_root": "$SYNTHETIC_ROOT",
    "island_name": "$ISLAND_NAME",
    "region": "$AWS_REGION_NAME",
    "aws_profile": "$AWS_PROFILE_NAME",
    "aws_subnet": "$AWS_SUBNET",
    "aws_security_group": "$AWS_SECURITY_GROUP",
    "use_public_ip": "$USE_PUBLIC_IP",
    "sb_ami_id": "$SB_AMI",
    "sb_instance_type": "$SB_INSTANCE",
    "prime_binary": "$SHADOW_PRIME_AGENT_BIN",
    "seed_sha256": {str(p.relative_to(Path("$SHARED_ROOT/seeds/combined"))): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("$SHARED_ROOT/seeds/combined").rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
    "synthetic_score_semantics": "ungraded_constant_zero",
    "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path("$SYNTHETIC_ROOT/PACKAGE_MANIFEST.json"), Path("$PROMPT_TEMPLATE"),
        Path("$ROOT/eval_sets/$TRAIN_SET"), Path("$ROOT/eval_sets/$VAL_SET"),
        Path("$ROOT/../../agent_s4/eval/scienceboard_prime.py"),
    ]},
    "builder_harness": "$HARNESS",
    "builder_model": "$MODEL",
    "builder_timeout": $BUILDER_TIMEOUT,
    "eval_model": "$MODEL",
    "eval_timeout": $EVAL_TIMEOUT,
    "eval_max_steps": $EVAL_MAX_AGENT_STEPS,
    "eval_num_envs": $EVAL_NUM_ENVS,
    "eval_batch_timeout": $EVAL_BATCH_TIMEOUT,
    "scheduler": "$SCHEDULER_SPEC",
    "parent_selection": "$PARENT_SELECTION",
    "max_steps": $MAX_EVOLUTION_STEPS,
    "prime_temperature": "$PRIME_TEMPERATURE",
    "prime_idle_timeout": "$PRIME_IDLE_TIMEOUT",
    "context_window_source": "$CONTEXT_WINDOW_SOURCE",
    "infra_retries": $INFRA_RETRIES,
    "orchestrator_command": orchestrator_command,
}
if path.is_file():
    recorded = json.loads(path.read_text())
    drift = {k: (recorded.get(k), v) for k, v in current.items()
             if recorded.get(k) != v and k not in {"max_steps", "orchestrator_command", "config_history"}}
    if current["max_steps"] < recorded.get("max_steps", 0):
        drift["max_steps"] = (recorded.get("max_steps"), current["max_steps"])
    accepted = {k.strip() for k in os.environ.get("SHADOW_EVOL_ACCEPT_CONFIG_DRIFT", "").split(",") if k.strip()}
    unaccepted = {k: v for k, v in drift.items() if k not in accepted}
    if unaccepted:
        lines = "\n".join(f"  {k}: recorded={a!r} now={b!r}" for k, (a, b) in sorted(unaccepted.items()))
        raise SystemExit("ERROR: configuration drifted from first launch:\n" + lines +
                         "\nName keys in SHADOW_EVOL_ACCEPT_CONFIG_DRIFT to accept and record them.")
    current["config_history"] = recorded.get("config_history", [])
    if drift:
        current["config_history"].append({
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "steps_completed_before_change": recorded.get("max_steps"),
            "accepted_drift": {k: {"was": a, "now": b} for k, (a, b) in sorted(drift.items())},
        })
    print("Configuration matches" if not drift else "Accepted configuration change", path)
else:
    if Path("$ROOT/shadow_evolution_scienceboard_$RUN_NAMESPACE.db").exists():
        raise SystemExit("ERROR: existing namespace DB has no run config; choose a new namespace")
    print(f"Recorded run configuration to {path}")
path.write_text(json.dumps(current, indent=2) + "\n")
PY

cd "$ROOT"
exec env \
    AWS_REGION="$AWS_REGION_NAME" \
    AWS_DEFAULT_REGION="$AWS_REGION_NAME" \
    AWS_SUBNET_ID="$AWS_SUBNET" AWS_SECURITY_GROUP_ID="$AWS_SECURITY_GROUP" \
    SHADOW_EVOL_OWNER_TAG="$OWNER_TAG" \
    SHADOW_EVOL_RUN_NAMESPACE="$RUN_NAMESPACE" \
    SHADOW_EVOL_USE_PUBLIC_IP="$USE_PUBLIC_IP" \
    SHADOW_EVOL_EVAL_NUM_ENVS="$EVAL_NUM_ENVS" \
    SHADOW_EVOL_EVAL_BATCH_TIMEOUT="$EVAL_BATCH_TIMEOUT" \
    SHADOW_EVOL_INFRA_RETRIES_PIN="$INFRA_RETRIES" \
    SHADOW_PRIME_CONTEXT_WINDOW_SOURCE="$CONTEXT_WINDOW_SOURCE" \
    SHADOW_PRIME_TEMPERATURE_PIN="$PRIME_TEMPERATURE" \
    SHADOW_PRIME_IDLE_TIMEOUT_PIN="$PRIME_IDLE_TIMEOUT" \
    SB_AMI_ID="$SB_AMI" SB_INSTANCE_TYPE="$SB_INSTANCE" \
    SCIENCEBOARD_SYNTHETIC_ROOT_PIN="$SYNTHETIC_ROOT" \
    dotenv -f "$ENV_FILE" run -- \
    bash -c '
        # dotenv run overrides caller-exported variables; re-apply the pins
        # after the env file is loaded so they hold.
        export OSWORLD_INFRA_RETRIES="$SHADOW_EVOL_INFRA_RETRIES_PIN"
        export SHADOW_PRIME_AGENT_TEMPERATURE="$SHADOW_PRIME_TEMPERATURE_PIN"
        export SHADOW_PRIME_IDLE_TIMEOUT="$SHADOW_PRIME_IDLE_TIMEOUT_PIN"
        export SCIENCEBOARD_SYNTHETIC_ROOT="$SCIENCEBOARD_SYNTHETIC_ROOT_PIN"
        : "${OPENAI_API_KEY:?OPENAI_API_KEY is missing from .env}"
        : "${OPENAI_BASE_URL:?OPENAI_BASE_URL is missing from .env}"
        if [[ "openai/$OPENAI_MODEL" != "$0" ]]; then
            printf "ERROR: model %s does not match OPENAI_MODEL %s\n" "$0" "$OPENAI_MODEL" >&2
            exit 1
        fi
        exec "$@"
    ' "$MODEL" "${CMD[@]}"
