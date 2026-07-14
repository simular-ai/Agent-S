# Isolated Agent S observer VM

These scripts build and operate a disposable Ubuntu 22.04 GNOME/Xorg VM. The VM
has one 1920x1080 display, no host folders or clipboard integration, a non-sudo
runtime user, localhost-only SSH/VNC forwards, and a restricted SSH key that can
launch only the observer MCP server.

## Build and run

```bash
./scripts/agent_s_vm/build_base.sh
./scripts/agent_s_vm/start.sh
./scripts/agent_s_vm/status.sh
python ./scripts/agent_s_vm/smoke_test.py --screenshot /tmp/agent-s-observer.png
```

Start the VM before starting or restarting Codex. Codex launches the MCP bridge
while loading its tool catalog; the bridge does not start a VM implicitly. The
20-second MCP startup timeout is intentionally shorter than the VM's maximum
180-second boot window.

The build downloads the official Ubuntu cloud image, verifies its signed
checksum, installs the pinned `requirements-observer.lock`, copies only the
observer package and guest runtime assets into `/opt/agent-s`, and removes
runtime sudo access before sealing the base image. It does not copy the legacy
desktop executor, repository metadata, or model credentials into the image.

Connect Remmina to `127.0.0.1:5905` to inspect the synthetic safety-test page.
There is no clipboard or file sharing between the viewer and guest.

Create a short-lived UI-TARS-1.5-7B Hugging Face Inference Endpoint, then place
only its URL in the host file below. Keep the API token in the normal Hugging
Face token store.

```text
~/.config/agent-s-lab/endpoint.env
HF_ENDPOINT_URL=https://example.endpoints.huggingface.cloud/v1/
```

The endpoint is lazy configuration: `status`, `observe`, `start_task`, and
`reset_task` work without it. Only `propose_next` requires the OpenAI key, the
Hugging Face token, and this endpoint URL.

Register `codex-mcp-bridge.sh` as a stdio MCP server and apply the exact tool
allowlist and `propose_next` approval rule in `codex-config.toml.example`. Start
a fresh Codex thread after registration. If the provider host cached its MCP
catalog before registration, fully restart that host rather than opening only
another conversation.

## Status and recovery

Run the cheap status card first:

```bash
./scripts/agent_s_vm/status.sh
```

The integration values have these meanings:

- `guest_unreachable`: the QEMU PID is absent, or the localhost SSH transport is
  unavailable or blocked in the current execution sandbox.
- `model_unconfigured`: the VM transport is reachable but no endpoint URL is
  configured. Observation can still work.
- `transport_ready`: the process, transport, and endpoint checks pass. Run the
  deep canary before relying on the MCP.

The deep status performs MCP initialization, exact tool-catalog validation, and
a 1920x1080 observation without starting a task:

```bash
./scripts/agent_s_vm/status.sh --deep
```

The full smoke test additionally starts and resets one in-memory task. It never
changes the desktop:

```bash
python3 ./scripts/agent_s_vm/smoke_test.py
```

If the QEMU process exists but SSH or the MCP canary stalls, recycle the
disposable runtime:

```bash
./scripts/agent_s_vm/stop.sh
./scripts/agent_s_vm/start.sh
./scripts/agent_s_vm/status.sh --deep
```

The stop step removes the current overlay. That is intentional: no runtime VM
state is durable. After a passing canary, restart Codex and confirm that the
five allowlisted tools are present.

Stop the VM and remove its disposable overlay with:

```bash
./scripts/agent_s_vm/stop.sh
```

Pause the paid Hugging Face endpoint separately before considering the test
complete.
