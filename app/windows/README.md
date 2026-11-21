# Agent S for Windows — native first increment

A C# / WinUI 3 desktop client for Agent S, alongside the existing SwiftUI Mac app. This first increment supports **Sai cloud computers only**. It is not yet a Windows release or full feature parity with the Mac app. The Windows CI job publishes the native project and runs a no-credential startup/Accounts smoke test; manual Windows verification is required before merge/release.

## Framework choice

Use **WinUI 3 with the Windows App SDK**, Microsoft's recommended native Windows desktop UI framework. The app uses native XAML controls, Credential Locker and Windows URI launching. C# keeps the UI, API calls and lifecycle in one language; there is no Electron/Chromium host or separate Node service. The Mac UI and Apple's Virtualization framework are platform-specific and stay unchanged.

- [WinUI documentation](https://learn.microsoft.com/en-us/windows/apps/winui)
- [Windows App SDK releases](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads)

WPF is a reasonable mature alternative, but there is no existing .NET UI to preserve here. Avalonia is useful when sharing the UI across platforms; Agent S already has a native SwiftUI app and this request is for native Windows. This PR uses the current stable Windows App SDK 2.7.0 and .NET 10, pinned in the project and SDK selection respectively.

## Included

- Create named local agent conversations with a Sai computer and allowed model; resume their Sai session after restarting.
- Stream chat and activity using the same `/v1/agents` contracts as `app/Sources/AgentS/Models/SaiAPI.swift`.
- Show approval text, explicitly approve/deny, or open HTTPS link-only approvals. No automatic permission grants. Approval controls and URLs are not restored as actionable state from disk.
- Stop through the server abort endpoint. Before a new turn returns its session ID, a stop waits for that ID instead of accidentally aborting the preceding turn. Failed aborts remain visible and retryable.
- Discover computer availability and request a wake. **Open computer** opens Sai's live page in the default browser; there is no embedded screen panel in this increment. Live links stay in memory, not the saved transcript.
- Store the Sai API key in Windows Credential Locker, and conversations under `%LOCALAPPDATA%\Simular\Agent S\conversations.json`. Removing a key clears saved session associations and screen-link caches; it does not delete conversations. Deleting an agent removes its local conversation only.

Chat history contains the text you send and the agent's responses. It is ordinary local JSON under the Windows user profile, not an encrypted database. Do not share that directory. Cloud prompts/tool results are processed by Sai and its configured providers. No analytics or new backend endpoints are added.

## Build

Use Windows 11 x64, Visual Studio with the **WinUI application development** workload and the **.NET 10 SDK**. Windows App SDK/WebView/native build tooling must be available on the machine. From the repository root:

```powershell
dotnet run --project app/windows/AgentS.Core.Tests -c Release
dotnet build app/windows/AgentS.Windows -c Release -p:Platform=x64 -r win-x64
```

The PR’s **Windows app** Actions run provides an `agent-s-windows-x64` artifact after the startup smoke test passes. Download and extract the full archive, then run `AgentS.Windows.exe`. This is an unsigned development build, not a release installer.

When building locally, run the resulting `AgentS.Windows.exe` from its build output directory. Keep its adjacent self-contained runtime files together. The project runs unelevated and unpackaged; signing, installer/update distribution and clean-machine installation are separate release work. ARM64 is declared as a future build target but not yet validated by the x64 CI job.

Add a key from [Sai API keys](https://platform.simular.ai/api-keys) in **Accounts**, refresh computers, then create an agent. It uses your existing Sai account/computers; it does not provision a free computer or enroll the local Windows machine.

Publishing must preserve `resources.pri`. `EnableMsixTooling` generates the XAML resource index even for this unpackaged app; without it, compilation can pass while startup fails to locate `MainWindow.xaml`. CI checks the published index and launches the actual executable to catch this regression. Startup diagnostics are opt-in via `AGENT_S_STARTUP_DIAGNOSTICS` for the credential-free smoke run; they do not handle or hide the failure.

The dependency-free core contract checks also run on Linux/macOS with .NET 10. They use a fake HTTP transport, not a production account, and do not establish live API or UI correctness.

## Route to feature parity

| Mac mode / feature | This PR | Next work |
| --- | --- | --- |
| Sai cloud chat, named conversations, approvals | Implemented, pending Windows/live verification | Streaming/recovery and approval UX validation |
| Computer display and context | Browser screen link, native status | Embedded screen panel, context/usage parity |
| This Mac → This PC | Not included | Windows simulang adapter, accessibility/screenshot/input boundary and per-action approval bridge; reuse CLI protocols rather than invent a new agent loop |
| Claude Code / Codex / DeepSeek and Box | Not included | Windows process lifecycle and account-profile adapters; port/test the existing stream contracts and provider matrix |
| Private Linux VM + Ollama | Not included | Prototype Hyper-V versus WSL2 against display/input isolation, shared files, resource limits, hardware requirements and Windows editions before choosing. WSLg windows alone do not establish an isolated private desktop |
| 1Password, guest file sharing, model downloads | Not included | Platform-specific integrations and explicit data/consent handling |

Keep the provider/agent behavior and Sai API unchanged as Windows UI lands incrementally. There is no background agent process in this first client. Closing during a remote turn offers server abort when its session ID is known; closing without it warns that remote work may continue. Losing the stream is not treated as completion.

## Verification and release gates

Local Linux verification is limited to core compilation and API/stream contract checks. Windows CI covers native compilation/publishing, opening the window without credentials, disabled Send/Stop/New agent controls, opening/cancelling Accounts, the masked key field and clean exit. Its separate `windows-smoke-evidence` artifact contains a machine-readable report and real-window captures for review. This test never enters an API key or starts a remote task. Credential save/read/remove and authenticated flows still require a Windows test account. Do not treat these as verified by the portable tests.

Before marking ready, use a test Sai account on Windows and record:

1. Build/run on a fresh Windows 11 x64 machine; verify Credential Locker save/read/remove and restart without printing keys.
2. Create two named agents and switch/restart them. Confirm computer/session/history isolation; change credentials and confirm old sessions/live links are not reused.
3. Send a task, watch streaming text/activity, continue its session, deny and approve actions, and open a link-only approval. Fail an approval request and confirm it remains retryable.
4. Stop before and after `data-session`, close during a turn, and disconnect midstream. Never show a confirmed stop/completion when the server has not confirmed it.
5. Test 401, offline/timeout, invalid SSE, corrupt history, disk-save failure, missing computer and wake rejection. Existing local history must not be silently overwritten when loading fails.
6. Verify screen link reuse/expiry, no live token in local JSON, and no URI opening without a user click. Verify browser launch failure is visible.
7. Capture native screenshots at normal/narrow widths and 150%/200% scaling, light/dark/high-contrast themes, keyboard focus and Narrator. Check transcript reading position during updates and agent changes. Use the CI startup/Accounts captures as limited startup evidence; they do not verify chat, approval, live-screen or scaling behavior.
