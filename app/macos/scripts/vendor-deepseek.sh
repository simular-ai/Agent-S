#!/bin/sh
# Builds build/deepseek-runtime: the pinned Node.js and the DeepSeek Harness packages that Agent S's
# profile loads, for bundle.sh to sign and ship. Inputs are pinned: Node by SHA-256, dsh and its
# dependencies by DeepSeekRuntime/package-lock.json, installed without running package scripts.
#
#   ./scripts/vendor-deepseek.sh           build (skipped when the inputs haven't changed)
#   ./scripts/vendor-deepseek.sh --check   build, then run the runtime integration test against it
#   ./scripts/vendor-deepseek.sh --trace   regenerate DeepSeekRuntime/packages.txt after a dsh update
set -eu
cd "$(dirname "$0")/.."
NODE_VERSION=24.22.3
NODE_SHA256=ff6e17d4874b76d272212b908cc020d6e815fba5a6c7eaaa06add725840307fd
MODE=${1:-}
CACHE="$PWD/build/cache"
OUT="$PWD/build/deepseek-runtime"
mkdir -p "$CACHE"

archive="$CACHE/node-v$NODE_VERSION-darwin-arm64.tar.gz"
if ! echo "$NODE_SHA256  $archive" | shasum -a 256 -c - >/dev/null 2>&1; then
  curl -fsSL --retry 3 -o "$archive.part" "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-darwin-arm64.tar.gz"
  if ! echo "$NODE_SHA256  $archive.part" | shasum -a 256 -c - >/dev/null; then
    rm -f "$archive.part"
    echo "The Node.js download doesn't match its pinned checksum." >&2
    exit 1
  fi
  mv "$archive.part" "$archive"
fi

stamp=$(cat "$0" DeepSeekRuntime/package.json DeepSeekRuntime/package-lock.json DeepSeekRuntime/packages.txt \
  DeepSeekRuntime/tools/prune.py | shasum -a 256 | cut -c1-16)

install() { # <folder>: the full, unpruned tree
  rm -rf "$1"
  mkdir -p "$1/bin" "$1/node"
  tar -xzf "$archive" -C "$1/node" --strip-components 1
  cp DeepSeekRuntime/package.json DeepSeekRuntime/package-lock.json "$1/"
  (cd "$1" && HOME="$1/.npm-home" npm_config_cache="$CACHE/npm" "$1/node/bin/node" \
    "$1/node/lib/node_modules/npm/bin/npm-cli.js" ci --omit=dev --ignore-scripts --no-audit --no-fund --no-update-notifier --loglevel=error)
  rm -rf "$1/.npm-home"
  mv "$1/node/bin/node" "$1/bin/node"
  mv "$1/node/LICENSE" "$1/NODE-LICENSE"
  rm -rf "$1/node"
}

run_tests() { # <runtime folder> [extra environment...]
  runtime=$1
  shift
  env AGENT_S_DSH_RUNTIME="$runtime" "$@" ./scripts/test.sh --filter DeepSeekSDKRuntimeTests
}

if [ "$MODE" = --trace ]; then
  full="$CACHE/deepseek-full"
  install "$full"
  trace="$CACHE/deepseek-trace.txt"
  rm -f "$trace"
  run_tests "$full" NODE_OPTIONS="--import=$PWD/DeepSeekRuntime/tools/trace.mjs" AGENT_S_TRACE="$trace"
  python3 -I DeepSeekRuntime/tools/packages.py "$full/node_modules" "$trace" > DeepSeekRuntime/packages.txt
  echo "Updated DeepSeekRuntime/packages.txt. Run $0 --check to verify the pruned runtime."
  exit 0
fi

if [ ! -f "$OUT/.stamp" ] || [ "$(cat "$OUT/.stamp")" != "$stamp" ]; then
  staging="$OUT.staging"
  install "$staging"
  python3 -I DeepSeekRuntime/tools/prune.py "$staging/node_modules" DeepSeekRuntime/packages.txt "$staging/mach-o.txt"
  echo "$stamp" > "$staging/.stamp"
  rm -rf "$OUT"
  mv "$staging" "$OUT"
  echo "Built $OUT ($(du -sh "$OUT" | cut -f1))"
fi

if [ "$MODE" = --check ]; then run_tests "$OUT"; fi
