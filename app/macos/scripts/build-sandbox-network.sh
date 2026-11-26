#!/bin/sh
# Bundle the host-routed sandbox network helper and its linked dependencies' notices.
set -eu
cd "$(dirname "$0")/.."
if ! command -v go >/dev/null 2>&1; then
  echo "Install Go 1.26 or later to build the sandbox network helper." >&2
  exit 1
fi

export GOTOOLCHAIN=local CGO_ENABLED=1 GOOS=darwin GOARCH=arm64
export MACOSX_DEPLOYMENT_TARGET="$1"
app="$PWD/$2"
go -C SandboxNetworkRuntime build -mod=readonly -trimpath -o "$app/Contents/Helpers/AgentSSandboxNetwork" .
go -C SandboxNetworkRuntime list -mod=readonly -deps \
  -f '{{with .Module}}{{if not .Main}}{{.Path}}{{"\t"}}{{.Dir}}{{end}}{{end}}' . > build/sandbox-network-modules.txt

python3 -I - build/sandbox-network-modules.txt "$app/Contents/Resources/SandboxNetworkNotices" <<'PYTHON'
import shutil
import sys
from pathlib import Path

module_list, destination = map(Path, sys.argv[1:])
for entry in sorted(set(module_list.read_text().splitlines()) - {""}):
    name, location = entry.split("\t", 1)
    for notice in sorted(Path(location).iterdir()):
        if notice.is_file() and notice.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "PATENTS")):
            target = destination / name / notice.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(notice, target)
PYTHON
