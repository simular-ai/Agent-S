# Agent S for Mac

Agent S is a native macOS app for computer-use agents. Choose a Linux VM on your Mac,
let an agent work in your Mac apps with approvals, or connect to a Sai computer or
Simular Box. Each agent has its own conversation and can pick up where it left off.


![Agent S showing a demo conversation, the Sai computer, and session context](screenshots/agent-s-cloud.png)

*Demo conversation with the cloud computer and session context beside the chat. The computer panel shows an illustrative desktop.*

<details>
<summary>See an approval on This Mac</summary>

![Agent S asking for approval to open Notes on this Mac](screenshots/agent-s-mac-approvals.png)

</details>

## Build and run

Agent S requires an Apple Silicon Mac with macOS 14 or later. To build from source, install
Xcode with Swift 6.2 or later. Build and open the app from the repository root.

```sh
cd app/macos
./scripts/bundle.sh
open "build/Agent S.app"
```

The build script signs the app for local development by default. Run `./scripts/test.sh`
from `app/macos/` to run the Swift test suite.

Closing the last window keeps Agent S, active turns, and the VM running. Click Agent S in
the Dock to reopen it. Use **Quit Agent S** to stop active work and shut down the VM.

## Choose how an agent runs

Click **+** in the sidebar, name the agent, and select **Runs**. For **Cloud**, choose a
computer first. Choose the harness and **Provider** when prompted. To change an existing
agent's configuration, right-click it and choose **Agent Settings…**. Each **Linux VM**
agent keeps the provider selected when it was created. To use a different provider,
create a new agent.

| Runs on | Harness | Model provider |
| --- | --- | --- |
| **Linux VM** · On your Mac | Claude Code, Codex, DeepSeek | Ollama, OpenRouter |
| **This Mac** · your Mac apps and files | Claude Code | Ollama, OpenRouter, Claude Code account |
| **This Mac** · your Mac apps and files | Codex | Codex account |
| **Cloud** · Sai computer | Sai | Sai API, with models available on your Sai account |
| **Cloud** · Simular Box | Claude Code | Ollama, OpenRouter, Claude Code account |
| **Cloud** · Simular Box | Codex | Ollama, OpenRouter, Codex account |
| **Cloud** · Simular Box | DeepSeek | Ollama, OpenRouter |

The display button beside the chat opens the computer panel, where you can watch the
Linux VM or cloud computer and see the agent's session context.

### Linux VM

1. Open **Settings → Computer** and click **Set up the computer**. This downloads a
   Debian 13 ARM64 image and creates a sparse 32 GB disk that grows with use.
2. When creating the agent, set **Runs** to **Linux VM** and select a **Provider**.
   For **Ollama**, start [Ollama](https://ollama.com/) 0.14 or later on your Mac and
   select or download a local model in **Agent Settings… → Model**. The default is
   `gemma4`. For **OpenRouter**, add your API key and choose a hosted model.
3. Turn on the VM from the chat or computer panel. The first boot needs internet access
   to install the desktop and tools in the VM. Wait for setup to finish, then send a message.
   One agent can use the shared Linux desktop at a time.

In **Settings → Computer → Memory**, choose **2 GB**, **4 GB**, or **6 GB** for the next
VM start. The default is **6 GB**. This setting allocates memory to the VM. Ollama runs
separately, so budget host memory for the model too. In the computer panel, use **Take over**
to send mouse and keyboard input to the VM.

### This Mac

1. Install the CLI you chose for the agent. [Claude Code setup](https://code.claude.com/docs/en/setup) · [Codex CLI setup](https://learn.chatgpt.com/docs/cli)
2. Install simulang with `npm install -g @simular-ai/simulang`.
3. In **Settings → Mac Permissions**, allow **Accessibility**. Enable **Screen Recording**
   if you want the agent to take screenshots. These permissions apply to Agent S and are
   shared across its agents. Screen Recording may require reopening the app.
4. For on-device inference, start Ollama on your Mac. Set **Runs** to **This Mac**,
   select the **Claude Code** harness and the **Ollama** provider, and choose a local model.
5. Send a message and review proposed actions in the chat or in the floating **Agent Activity**
   panel while another app is in front. Read-only observations can run automatically.
   **Approve for this task** approves the remaining actions in the current turn.
   You can stop a turn from either window.

### Sai computer

Create a **Cloud → Sai computer** agent. Add your Sai API key in **Settings → Accounts**
or in the new-agent form, then choose a computer and model from your Sai account.
Sending a message can wake a sleeping Sai cloud computer. You can also use **Wake computer**
in the computer panel when Sai makes that action available. For a computer you enrolled,
make sure its Sai runtime is online.
[Sai API keys](https://platform.simular.ai/api-keys) · [Sai computer lifecycle](https://platform.simular.ai/documentation/concepts/computers)

### Simular Box

Create a **Cloud → Simular Box** agent. Add your personal Box key in **Settings → Accounts**
and leave the **Box** field empty for your default Box, or enter the name of a specific Box.
The selected CLI runs on your Mac and uses the Box's tools.
[Box access keys](https://box.simular.ai/access)

Install Claude Code or Codex using the links above. For DeepSeek on Box, use Node.js 24.9
or later and install the version expected by Agent S.

```sh
npm install -g @deepseek-ai/dsh@0.2.2-rc.5
```

### Model setup

- **Ollama** runs on the Mac and is needed when you select it as the provider. You can
  check the connection in **Settings → Ollama** and download a model in **Agent Settings…**.
- **OpenRouter** requires an API key in **Settings → Accounts** and a model in the agent's
  settings. [OpenRouter keys](https://openrouter.ai/keys)
- **Claude Code account** and **Codex account** use automatic model selection. After
  creating the agent, choose **Sign in with Claude Code…** or **Sign in with ChatGPT…**.
  The installed CLI handles sign-in in an Agent S profile.

## Permissions and data

- With **Ollama**, choose a local model for on-device inference in **Linux VM** or in
  **This Mac** with **Claude Code**.
- **Linux VM** agents share the guest, browser, and files. The guest can reach the internet,
  shares the clipboard, and can read and write the app's shared folder at
  `~/Library/Application Support/Agent S/share` (`/mnt/sai` in the VM).
  Harness commands run in the VM with per-tool approval disabled.
- **This Mac** uses the Mac's apps, Terminal, and files with approvals for actions.
- Hosted providers receive prompts and tool results sent to the model, which can include
  web pages, file contents, and screenshots. Agent S stores the OpenRouter key in the
  Mac's Keychain and forwards guest model requests through the app's proxy.
- **Sai** receives tasks and runs them on the selected computer. **Simular Box**
  receives tool calls and their inputs even when the agent uses Ollama. A hosted
  model provider can also receive Box tool results used as model context. Sai API
  requests and Agent S–managed Box sessions identify the app to the service.
- The app keeps its chat history and agent settings in
  `~/Library/Application Support/Agent S`. Sai, Box, and OpenRouter API keys are
  stored in the macOS Keychain.

## Source layout

- [Sources/AgentS](Sources/AgentS) contains the SwiftUI app, agent sessions, provider
  integrations, and Virtualization framework runtime.
- [Sources/AgentSCore](Sources/AgentSCore) contains mode and provider rules, guest
  setup, harness commands, and stream parsers.
- [Tests/AgentSCoreTests](Tests/AgentSCoreTests) contains the Swift tests.

## Publishing

The bundle identifier is `ai.simular.agent-s`. The full macOS app is intended for
direct distribution as a Developer ID–signed, notarized DMG. A Mac App Store build
would need to meet Apple's sandbox requirements for its automation features.
[Apple's macOS distribution options](https://developer.apple.com/macos/distribution/)

1. Build with the organization's **Developer ID Application** identity by setting
   `SIGN_IDENTITY` when running `./scripts/bundle.sh release`. Verify that the app's signature
   includes the hardened runtime and a secure timestamp.
2. Put the app in a DMG, sign the DMG, and submit it with `xcrun notarytool submit`
   using the organization's notary profile. After Apple accepts it, staple and validate
   the ticket with `xcrun stapler`. Verify Gatekeeper on a clean Mac.
   [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
3. Test a fresh install and upgrade, including macOS permissions, VM startup, and Sai
   and Box sign-in. Pin and verify the guest downloads, and publish a privacy policy
   that covers the modes shipped in the release.
4. Verify the account integrations for the intended distribution. OpenAI documents
   **Sign in with ChatGPT** for commercial or hosted integrations in place of the
   current Codex App Server authentication. Anthropic's terms govern the use of
   Claude Code in a product. [Codex authentication](https://learn.chatgpt.com/docs/app-server#auth-endpoints) · [Claude Code terms](https://code.claude.com/docs/en/legal-and-compliance)

[Repository license](../../LICENSE)
