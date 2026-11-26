#!/bin/bash
# Installs the coding-agent harnesses and screen tools into the Private computer, as root.
# Agent S runs this before the first Claude Code, Codex or DeepSeek turn (and after Reset computer).
# The tool files are already in /usr/local/lib/sai/new; packages install once per TOOLS_VERSION.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
TOOLS_VERSION=1  # Bump to reinstall packages; the tool files above are refreshed on every run.
DSH_VERSION="${1:?dsh version}"
PLAYWRIGHT_VERSION=1.65.1
lib=/usr/local/lib/sai

install -d -m 0755 "$lib"
install -m 0644 "$lib/new/agent-s-env.sh" "$lib/agent-s-env.sh"
install -m 0644 "$lib/new/web.mjs" "$lib/web.mjs"
install -m 0644 "$lib/new/ui.ts" "$lib/ui.ts"
install -m 0644 "$lib/new/agent-s-x11-socket.py" "$lib/agent-s-x11-socket.py"
install -m 0755 "$lib/new/web" /usr/local/bin/web
install -m 0755 "$lib/new/ui" /usr/local/bin/ui
install -m 0755 "$lib/new/agent-s-ollama-relay.py" /usr/local/bin/agent-s-ollama-relay
rm -rf "$lib/new"

if [ "$(cat "$lib/.tools-version" 2>/dev/null)" = "$TOOLS_VERSION" ] &&
   [ "$(dsh --version 2>/dev/null)" = "$DSH_VERSION" ]; then
  echo "Harnesses are installed."
  exit 0
fi

# Stdout carries the progress lines Agent S shows; tool output goes to stderr (kept for error messages).
echo "Installing system packages…"
apt-get update -q >&2
apt-get install -y -q --no-install-recommends curl ca-certificates gnupg git at-spi2-core python3 >&2

# simulang needs Node >= 22.18; Debian 13 ships 20.
if ! node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=18)?0:1)' 2>/dev/null; then
  echo "Installing Node.js 22…"
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >&2
  apt-get install -y -q nodejs >&2
fi

echo "Installing Claude Code, Codex, DeepSeek Harness and simulang…"
npm install -g --no-audit --no-fund --loglevel=error \
  @anthropic-ai/claude-code @openai/codex @simular-ai/simulang "@deepseek-ai/dsh@$DSH_VERSION" >&2

echo "Installing the browser driver…"
(cd "$lib" && npm install --no-audit --no-fund --loglevel=error "playwright-core@$PLAYWRIGHT_VERSION" >&2)

apt-get clean >&2
install -d -o agent -g agent /home/agent/work
echo "$TOOLS_VERSION" > "$lib/.tools-version"
echo "Harnesses are installed."
