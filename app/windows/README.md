# Agent S for Windows — native first increment

Agent S for Windows is a native WinUI 3 app for running agents on a private Linux computer. New conversations default to **Private**. Existing Sai conversations keep their cloud target. Select **Cloud · Sai computer** to create another one. Windows CI publishes the development app and runs credential-free startup checks. The Private VM still needs hands-on verification on a Hyper-V-capable Windows computer.

## Framework choice

Use **WinUI 3 with the Windows App SDK**, Microsoft's recommended native Windows desktop UI framework. The app uses native XAML controls, Credential Locker and Windows URI launching. C# keeps the UI, API calls and lifecycle in one language; there is no Electron/Chromium host or separate Node service. The Mac UI and Apple's Virtualization framework are platform-specific and stay unchanged.

- [WinUI documentation](https://learn.microsoft.com/en-us/windows/apps/winui)
- [Windows App SDK releases](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads)

WPF is a reasonable mature alternative, but there is no existing .NET UI to preserve here. Avalonia is useful when sharing the UI across platforms; Agent S already has a native SwiftUI app and this request is for native Windows. This PR uses the current stable Windows App SDK 2.7.0 and .NET 10, pinned in the project and SDK selection respectively.

## Set up the Private computer

The prototype uses Windows 11 x64 Pro or Enterprise with Hyper-V, the Windows OpenSSH Client, and Ollama on the Windows host. Use the same Windows account for setup and for Agent S. The account needs access to manage Hyper-V VMs. Set `OLLAMA_NO_CLOUD=1` for Ollama, restart it, and pull a local model with tool calling such as `nemotron-3-nano:4b` (the default) or `gemma4`. Keep Ollama listening at `127.0.0.1:11434`. The host runs the model, so its memory is separate from the VM.

Provide a fresh, trusted Debian 13 amd64 generic cloud disk in VHDX format. Verify the original image against Debian’s signed checksums. To convert the Debian QCOW2 disk, run `qemu-img convert -f qcow2 -O vhdx input.qcow2 debian.vhdx`. The setup script uses an app-owned copy of that disk. Run setup from an elevated Windows PowerShell:

```powershell
.\app\windows\scripts\setup-private-vm.ps1 -VhdxPath C:\path\to\debian.vhdx
```

Add `-Dev` to set up the computer for a source build (Agent S Dev) instead; see [Agent S Dev and Agent S](#agent-s-dev-and-agent-s).

Setup installs the guest packages over the Hyper-V Default Switch, then removes that temporary adapter. The running guest uses the dedicated internal network `192.168.247.0/30` without a default gateway. Agent S connects to a pinned SSH host key for screenshots and a small set of desktop actions. Host drives and clipboard are outside the guest. The VM has two vCPUs and 2, 4 or 6 GiB of memory, as on macOS: `-MemoryGB 4` picks it at setup (2 by default), and **Settings › Computer › Memory** changes it later. The app applies the choice each time the VM starts, including after **Reset computer**, so stop the computer to apply a change now.

Create a **Private** agent and enter the installed Ollama model tag. To exercise the local path, disconnect the computer from the internet and send “Open the demo, enter Sample in the name field, and submit.” Agent S boots the guest and shows screenshots as it works. **Stop** stops the turn, **Stop computer** shuts down the guest, and **Reset computer** restores the clean checkpoint while keeping Windows chat history. Use **Stop computer** to release the guest’s memory before closing the app.

- [Debian cloud images](https://cloud.debian.org/images/cloud/)
- [Hyper-V requirements](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v)
- [Ollama on Windows](https://docs.ollama.com/windows)
- [Disable Ollama cloud features](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)

## Coding-agent harnesses

As on macOS, a Private agent can run **Claude Code**, **Codex** or **DeepSeek Harness** inside the Private computer, or the app's built-in **Agent S** loop. Choose the harness in **New agent › This PC**, or switch an existing agent from the model menu above the chat. New Private agents default to Claude Code; agents created before harness support keep the built-in loop.

- The harnesses use the macOS app's commands: `claude -p --output-format stream-json`, `codex exec --json --oss`, and `dsh --profile headless --json` with the same pinned version and privacy overlay. Their JSON events become chat messages and activity rows, and each harness's session ID is kept per agent so the next message continues the session.
- They drive the screen with the same `web` (Playwright over Chromium's loopback debugging port) and `ui` (simulang accessibility tree) shell commands as the macOS guest. The screen panel refreshes while a harness works.
- Before the first harness turn, Agent S installs Node.js 22, the three CLIs, simulang and `playwright-core` into the guest as root. This needs internet access and takes a few minutes; later turns skip it. **Reset computer** restores the clean image, so the next harness turn installs them again and earlier harness sessions start over.
- The model is the host's Ollama. A WSL guest can't reach the host's loopback, so the app starts a relay: `agent-s-ollama-relay` listens on the guest's `127.0.0.1:11434` and multiplexes connections over its `wsl.exe` stdio to the app, which connects each one to Ollama on `127.0.0.1:11434`. It needs no firewall rule, administrator rights or `.wslconfig` change, and nothing else on the network can use it.
- Ollama's OpenAI-compatible endpoint, which Codex and DeepSeek use, applies the server's default context. That is 4K tokens on PCs without a large GPU, smaller than a harness's first request. Before a harness turn, Agent S creates `agent-s/<model>:32k` in Ollama with `num_ctx` 32768. The variant shares the model's weights, so nothing is downloaded, and it is hidden from the model pickers. Your Ollama settings are not changed.
- WSLg keeps `/tmp/.X11-unix` read-only, so the desktop's X server listens only on an abstract socket, and simulang only looks for the socket file. `ui` runs simulang in a user and mount namespace with a private `/tmp/.X11-unix` and a socket proxied to the display (`agent-s-x11-socket.py`). It needs no root and doesn't change the desktop.
- Harnesses run with per-tool approval disabled, as in the macOS Private mode. The guest is the sandbox: it has no Windows drives and no Windows interop. **Stop** ends the turn's whole process group in the guest.
- Only the WSL computer supports harnesses. A Hyper-V VM from `setup-private-vm.ps1` allows only the built-in loop's bounded desktop actions.

## This PC

Like **This Mac** on macOS, a **This PC** agent works on the PC itself: its apps, shell and files. Choose **This PC · your apps and files** in **New agent**. These agents use the Claude Code CLI installed on the PC, with Ollama or OpenRouter as the model provider.

- **Requirements:** [Claude Code](https://code.claude.com/docs/en/setup) and Git for Windows, whose Git Bash Claude Code uses for its Bash tool. New agent checks for both. Agent S runs Claude Code with its own profile (`%LOCALAPPDATA%\Simular\Agent S\claude-this-pc`), so your `~/.claude` settings, hooks and credentials are not used.
- **Approvals:** every tool use waits for you. Claude Code's `PreToolUse` hook (`approve.ps1`) asks the app over a loopback port with a per-run token. The chat shows the exact command or file change with **Allow**, **Allow for this task** (the rest of the current turn) and **Deny**. Only looking runs without asking: `pc apps`, `pc tree` and `pc screenshot` as plain commands, and reading the screenshot `pc` just saved. Steps still pending when a turn ends or stops are denied.
- **Using apps:** the agent uses `pc`, a PowerShell tool built on Windows UI Automation and the Windows counterpart of the Mac's `mac` tool. It needs nothing installed. Its commands are `pc apps`, `pc open`, `pc focus`, `pc tree` (the front window with refs), `pc click`, `pc type`, `pc key` and `pc screenshot`. The screen panel shows the agent's latest screenshot.
- **OpenRouter:** add a key in **Settings** or in New agent. It is stored in Windows Credential Locker. A loopback proxy adds it to requests to `https://openrouter.ai/api/v1` and requires a per-launch token, so the agent never holds the key. Prompts and what the agent reads go to OpenRouter and the model's provider. Pick any model in the model browser or type an id from openrouter.ai/models.
- **Zero Data Retention:** as on macOS, **Request Zero Data Retention (ZDR)** is on by default and applies to every OpenRouter agent. With it on, the proxy forwards only model requests and adds `"provider": {"zdr": true}` to each, so OpenRouter routes them only to ZDR endpoints. Turn it off in **Settings** or next to the OpenRouter model field.
- **OpenRouter on the Private computer:** every Private harness can use OpenRouter, the built-in Agent S loop included (OpenAI chat completions through the proxy). The relay forwards a second guest port, `127.0.0.1:11435`, to the same proxy. The guest gets only the proxy's per-launch token, never the key. With OpenRouter, prompts and what the agent reads leave the PC, so only choose it when that's acceptable.

## Matching the macOS app

- **First launch:** "Set up your first agent" (harness, provider and model; DeepSeek and Nemotron 3 Nano 4B by default), then "Get your agent ready", where the Private computer sets up while the model downloads. Existing installs skip it.
- **Models:** one browser for every model choice. The Ollama tab lists the whole ollama.com library (each local variant's download size and inputs, installed models, and "May not fit in memory"); the OpenRouter tab matches macOS's "OpenRouter models" sheet: every model with its input, ID, text token price ("$3.00 in · $15.00 out / 1M tokens"), ZDR availability (with a warning for Safety Retention only and no public ZDR endpoint) and context, a check on the current model, and **Refresh ZDR availability** when it couldn't be checked. Every OpenRouter model field has **Browse OpenRouter models…** above it and shows the chosen model's text token price, input and Zero Data Retention below it. Each model shows the agents that use it. Before a download, Agent S checks the Ollama version a model needs and offers to update Ollama.
- **Chat:** long prompts can be expanded and copied in full; This PC has a floating transcript with shared approvals and pauses new actions while its composer has focus; tool steps fold into one row per stretch with plain titles ("Open a webpage") and expandable Tool, Input and Output; code blocks have **Copy code**; a ↓ button jumps to the latest message; new Private chats offer sample tasks; Sai approvals add **Approve for this task**.
- **Composer:** stays usable while a turn runs. Messages become follow-ups (**Stop & send**, **Send next**, ×); each agent keeps its own draft; a bar above it says when the computer is off, needs setup, or is missing a key.
- **Agents:** right-click for Agent Settings…, Rename…, Color, New conversation, Move up/down and Delete; drag to reorder within a section; a badge shows local, hosted or unconfirmed inference. Empty sections link to creating an agent there.
- **Computer panel:** **Take over** (WSL Private computer) sends your mouse and keyboard to the guest's display over a WSL channel and streams the screen at about five frames a second; **Maximize** fills the window with the computer; the split is resizable. The Private desktop has a dock (tint2) with a Chromium launcher.
- **Context:** Action items (approvals to review, Open work to run or delete), Memory (notes agents save and recall for Claude Code, Codex and DeepSeek turns; add, edit, pin, resolve, share or forget them), and Session and instructions. Memory lives beside the conversations, one file per agent.
- **Computer memory:** 2, 4 or 6 GB, as on macOS, chosen in "Set up your first agent" and in **Settings › Computer**. WSL has one memory limit for every distro (`.wslconfig`, half the PC's memory by default), which Agent S never edits. So every guest process starts as root in a cgroup v2 group, `agent-s-<distro>`, whose `memory.max` is the chosen size, then drops to `agent`. Each start recreates the group and applies the size, so a change applies to the next task. Setups from before the choice keep WSL's limit. The cap can't exceed WSL's limit, which Settings shows. The Hyper-V computer applies the choice when it next starts.
- **Settings:** Computer (memory, as above), saved-key OpenRouter spend (daily, weekly, monthly and total), Accounts (with "Get a key" and key-prefix checks), ZDR, Ollama status, and **Check for Updates…** (opens the newest GitHub release).
- **Look:** black/white accent instead of the system blue, and the macOS chat, composer and button sizes.

`AGENT_S_DATA_DIR` points conversations and settings at another folder (used by the smoke test for a fresh install); the Private computer, CLI profiles and keys are unaffected.

## Sai cloud

- Create named local agent conversations with a Sai computer and allowed model; resume their Sai session after restarting.
- Stream chat and activity using the same `/v1/agents` contracts as the macOS app.
- Show approval text, explicitly approve/deny, or open HTTPS link-only approvals. No automatic permission grants. Approval controls and URLs are not restored as actionable state from disk.
- Stop through the server abort endpoint. Before a new turn returns its session ID, a stop waits for that ID instead of accidentally aborting the preceding turn. Failed aborts remain visible and retryable.
- Discover computer availability and request a wake. **Open computer** opens Sai's live page in the default browser; there is no embedded screen panel in this increment. Live links stay in memory, not the saved transcript.
- Store the Sai API key in Windows Credential Locker, and conversations under `%LOCALAPPDATA%\Simular\Agent S\conversations.json`. Removing a key clears saved session associations and screen-link caches; it does not delete conversations. Deleting an agent removes its local conversation only.

Chat history contains the text you send and the agent's responses. It is ordinary local JSON under the Windows user profile, not an encrypted database. Do not share that directory. Cloud prompts/tool results are processed by Sai and its configured providers. No analytics or new backend endpoints are added.

## Simular Box

Choose **Cloud · Simular Box** in **New agent**. Install the Claude Code CLI on Windows and add a [Simular Box key](https://box.simular.ai/access) in **Settings**. Leave the box name blank to use your account's default box, or enter a name. Choose a local Ollama model or an OpenRouter model.

Claude Code runs on Windows with only the Box MCP tools. Browser, desktop, shell, and file actions run on the selected Box. The app saves the Box key in Windows Credential Locker and passes it to the CLI through its environment. With OpenRouter, prompts and Box tool results can be sent to the hosted model provider.

The computer panel shows screenshots and offers **Open computer**, **Wake box**, and **Refresh screen**. **Stop** ends the local Claude Code process. A Box action already sent may still finish.

## Build

Use Windows 11 x64, Visual Studio with the **WinUI application development** workload and the **.NET 10 SDK**. Windows App SDK/WebView/native build tooling must be available on the machine. From the repository root:

```powershell
dotnet run --project app/windows/AgentS.Core.Tests -c Release
dotnet build app/windows/AgentS.Windows -c Release -p:Platform=x64 -r win-x64
```

The PR’s **Windows app** Actions run provides an `agent-s-windows-x64` artifact after the startup smoke test passes. Download and extract the full archive, then run `AgentS.Windows.exe`. This is an unsigned development build, not a release installer.

### Agent S Dev and Agent S

Local and CI builds are **Agent S Dev**: the window, taskbar and exe say "Agent S Dev", the icon is black, and it keeps its own data in `%LOCALAPPDATA%\Simular\Agent S Dev` (conversations, settings, CLI profiles, the WSL computer `AgentS-Dev-Private`) so it never touches an installed Agent S. The release workflow builds **Agent S** with the indigo macOS icon using `-p:AgentSChannel=Release`. API keys in Credential Locker are shared. For a Hyper-V Private computer for Agent S Dev, run `setup-private-vm.ps1 -Dev`; it gets its own VM, switch and network (`192.168.248.0/30`). `scripts/make-release-icon.ps1` regenerates the Windows release icon and logo from the macOS artwork.

When building locally, run the resulting `AgentS.Windows.exe` from its build output directory. Keep its adjacent self-contained runtime files together. The project runs unelevated and unpackaged; signing, installer/update distribution and clean-machine installation are separate release work. ARM64 is declared as a future build target but not yet validated by the x64 CI job.

To use Sai, add a key from [Sai API keys](https://platform.simular.ai/api-keys) in **Settings**, then create an agent with **Cloud · Sai computer** selected. This uses computers in your existing Sai account.

Publishing must preserve `resources.pri`. The project invokes the SDK’s `GenerateProjectPriFile` target before resolving publish output and names the application index `resources.pri`; without it, compilation can pass while startup fails to locate `MainWindow.xaml`. CI checks the published index and launches the actual executable to catch this regression. Startup diagnostics are opt-in via `AGENT_S_STARTUP_DIAGNOSTICS` for the credential-free smoke run; they do not handle or hide the failure.

The dependency-free core contract checks also run on Linux/macOS with .NET 10. They use a fake HTTP transport, not a production account, and do not establish live API or UI correctness.

## Route to feature parity

| Mac mode / feature | This PR | Next work |
| --- | --- | --- |
| Sai cloud chat, named conversations, approvals | Implemented, pending Windows/live verification | Streaming/recovery and approval UX validation |
| Computer display and context | Live screen, Take over and Maximize (Private), Action items, Open work, Memory, Session and instructions | Take over on a Box, saved/resumed sandbox state |
| Composer | Follow-up queue with Stop & send | Live steering of a running DeepSeek turn; several agents working at once |
| This Mac → This PC | Claude Code with Ollama or OpenRouter, per-action approvals, `pc` UI Automation tool | Codex with a Codex account, Claude Code account sign-in and DeepSeek on the host |
| Claude Code / Codex / DeepSeek on the Private computer | Implemented for the WSL computer with local Ollama, using the macOS commands, stream contracts and `web`/`ui` tools | Verify on a clean Windows host; Hyper-V VMs keep the built-in loop |
| Simular Box | Claude Code with Ollama or OpenRouter, screenshot and live view | Verify a live Box turn, session resume, wake and stop; Codex, DeepSeek and account sign-in remain follow-ups |
| Private Linux VM + Ollama | Hyper-V prototype with local Ollama, guest screen, stop and reset | Verify offline tasks on a clean Windows host, measure memory and disk, and evaluate Windows Home support |
| Model downloads and catalogs | Full Ollama library and OpenRouter list with sizes, ZDR and per-agent labels; downloads with progress | — |
| 1Password, guest file sharing | Not included | Platform-specific integrations and explicit data/consent handling |

Conversations store their target. Private turns use the installed Ollama model and the local guest. Sai turns use the existing API and retain their server session IDs. Closing during a remote turn offers server abort when its session ID is known. Losing the stream leaves completion unconfirmed.

## Verification and release gates

Local Linux verification is limited to core compilation and API/stream contract checks. Windows CI covers native compilation/publishing, opening and fitting the window to the desktop without credentials, disabled Send/Stop, opening the Private agent form, opening/closing Settings, the masked key field and clean exit. Its separate `windows-smoke-evidence` artifact contains a machine-readable report and real-window captures for review. This test never enters an API key or starts a remote task. Credential save/read/remove and authenticated flows still require a Windows test account. Do not treat these as verified by the portable tests.

Hosted runners do not exercise the Hyper-V guest. Before marking the draft ready, verify on a Windows 11 x64 Hyper-V host:

1. Provision the guest and complete a demo task with networking disconnected. Verify that the model runs on the Windows host and the guest has only the dedicated internal adapter.
2. Watch screenshots update, stop a turn, restart the VM, and reset it to the clean checkpoint. Confirm that Windows chat history remains available.
3. Record cold boot time, idle and peak guest memory, host model memory, and actual disk usage.

For Sai, use a Windows test account and record:

1. Build/run on a fresh Windows 11 x64 machine; verify Credential Locker save/read/remove and restart without printing keys.
2. Create two named agents and switch/restart them. Confirm computer/session/history isolation; change credentials and confirm old sessions/live links are not reused.
3. Send a task, watch streaming text/activity, continue its session, deny and approve actions, and open a link-only approval. Fail an approval request and confirm it remains retryable.
4. Stop before and after `data-session`, close during a turn, and disconnect midstream. Never show a confirmed stop/completion when the server has not confirmed it.
5. Test 401, offline/timeout, invalid SSE, corrupt history, disk-save failure, missing computer and wake rejection. Existing local history must not be silently overwritten when loading fails.
6. Verify screen link reuse/expiry, no live token in local JSON, and no URI opening without a user click. Verify browser launch failure is visible.
7. Capture native screenshots at normal/narrow widths and 150%/200% scaling, light/dark/high-contrast themes, keyboard focus and Narrator. Check transcript reading position during updates and agent changes. Use the CI startup/Settings captures as limited startup evidence; they do not verify chat, approval, live-screen or scaling behavior.
