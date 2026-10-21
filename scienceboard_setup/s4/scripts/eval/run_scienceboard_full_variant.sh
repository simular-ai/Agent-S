#!/usr/bin/env bash
# Full ScienceBoard eval (all 169 official tasks) of one variant of a
# ScienceBoard evolution run, under exactly the evolution's task-agent settings:
# Prime Agent on Qwen3.8-27B, 600 s per-task wall clock after setup,
# 300 agent steps, temperature 1, 720 s idle budget, t3.large, no recording,
# 2 infra retries.
#
#   bash scripts/eval/run_scienceboard_full_variant.sh --variant seed --yes
#   bash scripts/eval/run_scienceboard_full_variant.sh --variant if-014 --yes
#
# Rerun a subset with no per-task wall clock (300 steps / idle budget still apply):
#   bash scripts/eval/run_scienceboard_full_variant.sh --variant seed \
#       --tasks-file eval_sets/scienceboard_full169_seed_trunc.json \
#       --no-time-limit --suffix notimelimit --yes
#
# The variant's interface is materialized from its git tag in the island
# clone, so a later commit to the clone cannot change what is measured.
# Results: eval_cache_scienceboard_<ns>_full/<ns>-<variant>/
set -euo pipefail
# ROOT (scienceboard_setup/s4) is the ScienceBoard workspace; SHARED_ROOT
# (osworld_setup/s4) holds the venv and the Prime runtime.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHARED_ROOT="${SHADOW_EVOL_SHARED_ROOT:-$(cd "$ROOT/../../osworld_setup/s4" && pwd)}"
export SHADOW_EVOL_PROJECT_ROOT="$ROOT" SHADOW_EVOL_SHARED_ROOT="$SHARED_ROOT"
VENV_BIN="$SHARED_ROOT/.venv/bin"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
# agent_s4 imports from the Agent-S repository root, installed or not.
export PYTHONPATH="$(cd "$ROOT/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
NAMESPACE="${NAMESPACE:-qwen_sb_syn50v15}"
VARIANT=""
TASKS_FILE="${TASKS_FILE:-eval_sets/scienceboard_full_169.json}"
NUM_ENVS="${NUM_ENVS:-65}"
CONFIRMED=0
NO_TIME_LIMIT=0
SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --tasks-file) TASKS_FILE="$2"; shift 2 ;;
        --no-time-limit) NO_TIME_LIMIT=1; shift ;;
        --suffix) SUFFIX="-$2"; shift 2 ;;
        --yes) CONFIRMED=1; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$VARIANT" ]] || { printf 'usage: %s --variant seed|if-NNN [--yes]\n' "$0" >&2; exit 2; }

CLONE="$ROOT/ScienceBoard_Evol_$NAMESPACE"
TAG="evol/$NAMESPACE/$VARIANT"
git -C "$CLONE" rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
    || { printf 'ERROR: tag %s not found in %s\n' "$TAG" "$CLONE" >&2; exit 1; }
CONFIG="$ROOT/runs/${NAMESPACE}_run_config.json"
[[ -f "$CONFIG" ]] || { printf 'ERROR: run config %s missing\n' "$CONFIG" >&2; exit 1; }
read -r MODEL AMI INSTANCE SUBNET SG PROFILE TEMP IDLE TIMEOUT STEPS RETRIES < <(
    python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["eval_model"], c["sb_ami_id"], c["sb_instance_type"], c["aws_subnet"], c["aws_security_group"], c["aws_profile"] or "-", c["prime_temperature"], c["prime_idle_timeout"], c["eval_timeout"], c["eval_max_steps"], c["infra_retries"])' "$CONFIG")
# The run config records an empty profile as "-": use boto3's default chain.
if [[ "$PROFILE" != "-" ]]; then export AWS_PROFILE="$PROFILE"; else unset AWS_PROFILE; fi

RESULT_DIR="$ROOT/eval_cache_scienceboard_${NAMESPACE}_full/${NAMESPACE}-${VARIANT}${SUFFIX}"
SHADOW_DIR="$RESULT_DIR/_shadow_interface"
mapfile -t TASK_IDS < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$ROOT/$TASKS_FILE")
RUN_NAME="shadow-evol-sbfull-$(date -u +%Y%m%dT%H%M%SZ)-$VARIANT"

TIME_ARGS=(--timeout "$TIMEOUT")
[[ "$NO_TIME_LIMIT" == 1 ]] && { TIME_ARGS=(--no-time-limit); TIMEOUT="unlimited"; }
printf 'ScienceBoard full eval: %s (%s), %d tasks, %s envs\n' "$VARIANT" "$TAG" "${#TASK_IDS[@]}" "$NUM_ENVS"
printf '  model %s | %ss/task | %s steps | temp %s | idle %ss | %s x %s | retries %s\n' \
    "$MODEL" "$TIMEOUT" "$STEPS" "$TEMP" "$IDLE" "$AMI" "$INSTANCE" "$RETRIES"
printf '  results %s\n  instance Name tag %s\n' "$RESULT_DIR" "$RUN_NAME"
[[ "$CONFIRMED" == 1 ]] || { printf 'Dry run; pass --yes to launch.\n'; exit 0; }

mkdir -p "$SHADOW_DIR"
rm -rf "${SHADOW_DIR:?}"/*
git -C "$CLONE" archive "$TAG" shadow_interface | tar -x --strip-components=1 -C "$SHADOW_DIR"
printf 'Materialized %s (%s)\n' "$TAG" "$(git -C "$CLONE" rev-parse --short "$TAG")"

cleanup() {
    "$VENV_BIN/python" - "$RUN_NAME" <<'PY' || true
import sys, boto3
ec2 = boto3.client("ec2", region_name="us-east-1")
filters = [{"Name": "tag:Name", "Values": [sys.argv[1]]},
           {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}]
ids = [i["InstanceId"] for r in ec2.describe_instances(Filters=filters)["Reservations"] for i in r["Instances"]]
if ids:
    ec2.terminate_instances(InstanceIds=ids)
print(f"cleanup: terminated {len(ids)} leftover instance(s) tagged {sys.argv[1]}")
PY
}
trap cleanup EXIT

cd "$ROOT/ScienceBoard_Evol_$NAMESPACE"
env AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 \
    AWS_SUBNET_ID="$SUBNET" AWS_SECURITY_GROUP_ID="$SG" \
    SHADOW_EVOL_OWNER_TAG="${SHADOW_EVOL_OWNER_TAG:-}" AWS_INSTANCE_NAME="$RUN_NAME" \
    SB_AMI_ID="$AMI" SB_INSTANCE_TYPE="$INSTANCE" \
    PIN_TEMP="$TEMP" PIN_IDLE="$IDLE" PIN_RETRIES="$RETRIES" \
    SHADOW_PRIME_CONTEXT_WINDOW_SOURCE=interface \
    "$VENV_BIN/dotenv" -f "$ENV_FILE" run -- bash -c '
        export SHADOW_PRIME_AGENT_TEMPERATURE="$PIN_TEMP" SHADOW_PRIME_IDLE_TIMEOUT="$PIN_IDLE"
        export OSWORLD_INFRA_RETRIES="$PIN_RETRIES"
        exec "$@"' sbfull \
    "$VENV_BIN/python" -m agent_s4.eval.scienceboard_prime \
        --benchmark scienceboard --harness prime_agent --model "$MODEL" \
        --task_ids "${TASK_IDS[@]}" --num_envs "$NUM_ENVS" \
        --shadow_dir "$SHADOW_DIR" --result_dir "$RESULT_DIR" \
        --test_config_base_dir "$ROOT/ScienceBoard_Base/tasks/VM" \
        --region us-east-1 "${TIME_ARGS[@]}" --max-steps "$STEPS" \
        --infra-retries "$RETRIES" --use-public-ip
