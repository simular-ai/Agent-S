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
Xcode with Swift 6.2 or later and Go 1.26 or later. Build and open the app from the repository root.

```sh
cd app/macos
./scripts/bundle.sh
open "build/Agent S.app"
```

The build script signs the app for local development by default. It also bundles the
DeepSeek Harness runtime: on the first build, `scripts/vendor-deepseek.sh` downloads the pinned
Node.js release and installs the packages in `DeepSeekRuntime/package-lock.json`, so that build
needs network access. Run `./scripts/test.sh` from `app/macos/` to run the Swift test suite, and
`./scripts/vendor-deepseek.sh --check` to test the bundled runtime with synthetic model and tool
servers. After updating DeepSeek Harness in `DeepSeekRuntime/package.json`, regenerate the lock
file and run `./scripts/vendor-deepseek.sh --trace` to update the list of packages Agent S ships.

Closing the last window keeps Agent S, active turns, and the VM running. Click Agent S in
the Dock to reopen the window, or use **Quit Agent S** to exit the app.

The distributed app checks for updates automatically. Choose **Agent S → Check for Updates…** to
check manually. To upgrade from v0.3.3, quit Agent S and use the new DMG to replace the app
in Applications. Your agents and conversations stay in Application Support.

### Sandbox networking

**Settings → Sandbox Network** defaults to **Follow this Mac’s network**. The sandbox’s IPv4
connections use sockets on this Mac, and its DNS uses the macOS resolver. Connected VPNs apply
their policies to those Mac connections. Use fully qualified names for resources that rely on
VPN DNS search domains. Choose **Standard sandbox network** to use Apple’s VM NAT. Restart the
sandbox to apply a change.

Saved sandbox state uses the network mode it was created with. Choose that mode to resume it,
or use **Discard saved state and start…** to boot in the selected mode. Files saved to the
sandbox disk carry over.

The Computer panel shows pending requests and open work first. Check items under
**Open work** to run them together in one chat turn or delete them from memory.

## Choose how an agent runs

On first launch, pick a harness and model for **Sandbox on This Mac**, then click
**Set up agent** to download and start the sandbox. **Nemotron 3 Nano 4B** is
the default Ollama model. The model browser shows input support and estimated
download sizes for local models. Onboarding warns if the selected model is
likely to need more memory than your Mac can spare. The setup screen can open
the Ollama installer and download your selected model. Open chat when the
sandbox and model are ready, or use **Set up a cloud agent…** to add one.
The new chat offers sample tasks. Click one to fill the composer, where you can
edit it before sending.

Click **+** in the sidebar, name the agent, and select **Runs**. For **Cloud Computer**, choose a
computer first. Choose the harness and **Provider** when prompted. To change an existing
agent's configuration, right-click it and choose **Agent Settings…**. Each **Sandbox on This Mac**
agent keeps the provider selected when it was created. To use a different provider,
create a new agent.

| Runs on | Harness | Model provider |
| --- | --- | --- |
| **Sandbox on This Mac** · Linux VM | Claude Code | Ollama, OpenRouter, Claude Code account |
| **Sandbox on This Mac** · Linux VM | Codex | Ollama, OpenRouter, Codex account |
| **Sandbox on This Mac** · Linux VM | DeepSeek | Ollama, OpenRouter |
| **This Mac** · your Mac apps and files | Claude Code | Ollama, OpenRouter, Claude Code account |
| **This Mac** · your Mac apps and files | Codex | Codex account |
| **This Mac** · your Mac apps and files | DeepSeek | Ollama, OpenRouter |
| **Cloud Computer** · Sai computer | Sai | Sai API, with models available on your Sai account |
| **Cloud Computer** · Simular Box | Claude Code | Ollama, OpenRouter, Claude Code account |
| **Cloud Computer** · Simular Box | Codex | Ollama, OpenRouter, Codex account |
| **Cloud Computer** · Simular Box | DeepSeek | Ollama, OpenRouter |

The display button beside the chat opens the computer panel, where you can watch the
Linux VM or cloud computer and see the agent's session context.

### Sandbox on This Mac

1. If the sandbox hasn’t been set up, click **Set up** in the chat or
   **Settings → Computer → Set up the computer**. Setup downloads a Debian 13 ARM64
   image and creates a sparse 32 GB disk that grows with use.
2. When creating the agent, set **Runs** to **Sandbox on This Mac** and select a **Provider**.
   For **Ollama**, start [Ollama](https://ollama.com/) 0.14 or later on your Mac and
   select or download a local model in **Agent Settings… → Model**. The default is
   `gemma4`. For **OpenRouter**, add your API key and choose a hosted model.
3. Turn on the VM from the chat or computer panel. The first boot needs internet access
   to install the desktop and tools in the VM. Wait for setup to finish, then send a message.
   One agent can use the shared Linux desktop at a time.

Choose **2 GB**, **4 GB**, or **6 GB** during onboarding or in **Settings → Computer → Memory**.
The default is **2 GB**. Changes apply when you start a powered-off sandbox. Ollama uses
memory on the Mac separately, so budget host memory for the model too. In the computer panel, use **Take over**
to send mouse and keyboard input to the VM.

On Apple silicon, quitting Agent S saves the running sandbox when the Mac and VM
configuration support it. Reopen Agent S to resume open apps and windows. **Turn off**
shuts down the sandbox, and settings such as a new memory allocation apply on the next
full start. The sandbox disk keeps files and sign-ins across either action.

Chromium stays open between tasks while the sandbox is running. To use a subscription,
select **Use Claude Code account** or **Use Codex account** as the provider. Start the
sandbox and choose **Sign in** above the composer.

- **Codex** shows a one-time code in Agent S and opens your Mac browser. Enter the code
  there. Agent S detects when sign-in completes, and Codex stores its credentials in
  the sandbox. An older Codex version may show the code in the Computer panel.
  [Enable device-code sign-in for Codex](https://learn.chatgpt.com/docs/auth#login-on-headless-devices)
- **Claude Code** opens its sign-in page in your Mac’s browser. Finish signing in
  there and return to Agent S. Claude Code stores its credentials in the sandbox.

### This Mac

1. Install the CLI you chose for the agent. [Claude Code setup](https://code.claude.com/docs/en/setup) · [Codex CLI setup](https://learn.chatgpt.com/docs/cli). Agent S includes [DeepSeek Harness](#deepseek-harness).
2. Install simulang with `npm install -g @simular-ai/simulang`.
3. In **Settings → Mac Permissions**, allow **Accessibility**. Enable **Screen Recording**
   if you want the agent to take screenshots. These permissions apply to Agent S and are
   shared across its agents. Screen Recording may require reopening the app.
4. For on-device inference, start Ollama on your Mac. Set **Runs** to **This Mac**,
   select **Claude Code** or **DeepSeek** with **Ollama**, and choose a local model.
5. Send a message and review proposed actions in the chat or in the floating **Agent Activity**
   panel while another app is in front. Read-only observations can run automatically.
   **Approve for this task** approves the remaining actions in the current turn.
   You can stop a turn from either window.

### Sai computer

Create a **Cloud Computer → Sai computer** agent. Add your Sai API key in **Settings → Accounts**
or in the new-agent form, then choose a computer and model from your Sai account.
Sending a message can wake a sleeping Sai cloud computer. You can also use **Wake computer**
in the computer panel when Sai makes that action available. For a computer you enrolled,
make sure its Sai runtime is online.
[Sai API keys](https://platform.simular.ai/api-keys) · [Sai computer lifecycle](https://platform.simular.ai/documentation/concepts/computers)

### Simular Box

Create a **Cloud Computer → Simular Box** agent. Add your personal Box key in **Settings → Accounts**
and leave the **Box** field empty for your default Box, or enter the name of a specific Box.
The selected CLI runs on your Mac and uses the Box's tools.
[Box access keys](https://box.simular.ai/access)

Install Claude Code or Codex using the links above. DeepSeek Harness is included.

### DeepSeek Harness

For **This Mac** and **Simular Box**, Agent S runs the DeepSeek Harness release it was built with
(0.2.2-rc.5) on its own signed copy of Node.js. Each agent keeps one harness process for its
conversation, so follow-up messages start without relaunching it. **Stop** ends the process, and
the next message continues the same conversation. A harness with no messages for 15 minutes
exits until the next message. Development builds without the bundled runtime use
`AGENT_S_DSH_RUNTIME` (a folder built by `scripts/vendor-deepseek.sh`) or an installed `dsh` of
the same version.

### Model setup

- **Ollama** runs on the Mac and is needed when you select it as the provider. You can
  check the connection in **Settings → Ollama** and download a model in **Agent Settings…**.
- **OpenRouter** requires an API key in **Settings → Accounts**. Search by model or provider
  in the OpenRouter model settings or enter a model ID. **Request Zero Data Retention (ZDR)**
  applies to all OpenRouter agents. [OpenRouter keys](https://openrouter.ai/keys)
- **Claude Code account** and **Codex account** use automatic model selection. After
  creating the agent, choose **Sign in with Claude Code…** or **Sign in with ChatGPT…**.
  The installed CLI handles sign-in in an app-specific profile.

## Permissions and data

- Choose **Ollama** and a local model for on-device inference. This works with
  **Sandbox on This Mac** and with **This Mac** when the harness is **Claude Code** or **DeepSeek**.
- **Sandbox on This Mac** agents share the guest, browser, and files. The guest can reach the internet,
  shares the clipboard, and can read and write the app's shared folder at
  `~/Library/Application Support/Agent S/share` (`/mnt/sai` in the VM).
  Sending a message starts the task in the VM. Harness commands run there without
  per-tool approval. Sandbox CLI sign-ins live in the guest and are accessible to
  other agents using that sandbox.
- **This Mac** uses the Mac's apps, Terminal, and files with approvals for actions.
- Hosted providers receive prompts and tool results sent to the model, which can include
  web pages, file contents, and screenshots. Agent S stores the OpenRouter key in the
  Mac's Keychain and forwards guest model requests through the app's proxy.
- **Sai** receives tasks and runs them on the selected computer. **Simular Box**
  receives tool calls and their inputs even when the agent uses Ollama. A hosted
  model provider can also receive Box tool results used as model context. The app
  identifies itself when connecting to Sai and Simular Box.
- The app keeps its chat history and agent settings in
  `~/Library/Application Support/Agent S`. Sai, Box, and OpenRouter API keys are
  stored in the macOS Keychain.

## Source layout

- [Sources/AgentS](Sources/AgentS) contains the SwiftUI app, agent sessions, provider
  integrations, and Virtualization framework runtime.
- [Sources/AgentSCore](Sources/AgentSCore) contains mode and provider rules, guest
  setup, harness commands, and stream parsers.
- [Tests/AgentSCoreTests](Tests/AgentSCoreTests) contains the Swift tests.
- [DeepSeekRuntime](DeepSeekRuntime) pins the bundled DeepSeek Harness packages and holds the
  tools that select and prune them.

### Regression check for approvals

Commit `357d798` added a task-start prompt to the shared local VM send path.
A direct **Send** in **Sandbox on This Mac** authorizes that turn for every harness.
Keep confirmation for Simular Box task starts and approval for This Mac actions
in their respective paths.

When changing approvals, send a message to a sandbox agent using Codex, DeepSeek,
and Claude Code. Each should start without an approval card, including agents with
a saved **Ask first** preference. Check that **Ask first** on Simular Box still asks
before starting a task and **Ask for approval** on This Mac still asks before an
action such as opening an app.

## Publishing

The bundle identifier is `ai.simular.agent-s`. The full macOS app is intended for
direct distribution as a Developer ID–signed, notarized DMG. A Mac App Store build
would need to meet Apple's sandbox requirements for its automation features.
[Apple's macOS distribution options](https://developer.apple.com/macos/distribution/)

### Set up the release Mac

Install the organization's **Developer ID Application** signing identity, store
notarization credentials, and generate the Sparkle update key once. Run these commands
from `app/macos`. The Sparkle tools are under the SwiftPM scratch directory if you use
`--scratch-path`. [Sparkle setup](https://sparkle-project.org/documentation/)

```sh
swift package resolve
sparkle_bin="$PWD/.build/artifacts/sparkle/Sparkle/bin"
xcrun notarytool store-credentials agent-s-notary
"$sparkle_bin/generate_keys" --account agent-s
```

Keep the Sparkle private key in the release Mac's Keychain and reuse its public key for
each release. The release script embeds that public key and configures
`https://agent-s.ai/appcast.xml` as the default update feed.

### Build and verify a release

Set `CFBundleShortVersionString` to the release version and increment `CFBundleVersion`
in `Resources/Info.plist`. Sparkle compares the build number to find newer releases.

```sh
sparkle_public_key=$("$sparkle_bin/generate_keys" --account agent-s -p)
SIGN_IDENTITY='Developer ID Application: Your Organization (TEAMID)' \
  NOTARY_KEYCHAIN_PROFILE=agent-s-notary \
  SPARKLE_PUBLIC_ED_KEY="$sparkle_public_key" ./scripts/release.sh
```

The script builds and signs the app and disk image, submits the DMG to Apple, and staples
the approved ticket. It writes the versioned DMG and SHA-256 checksum to `build/`.
[Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)

Verify Gatekeeper on a clean Apple Silicon Mac. Test a fresh install and an upgrade from
v0.3.3, including macOS permissions, sandbox startup, and Sai and Box sign-in. The app
verifies the pinned Debian archive before unpacking. Publish a privacy policy that
covers the modes shipped in the release.

For an in-app update test, build two increasing versions against the same staging HTTPS feed
using `SPARKLE_FEED_URL`. Publish the newer test DMG and its signed feed to staging,
then use **Check for Updates…** in the older build. Confirm that the app installs the newer
build and reopens with its existing agents, chats, and sandbox.

Verify the account integrations for the intended distribution. OpenAI documents
**Sign in with ChatGPT** for commercial or hosted integrations in place of the current
Codex App Server authentication. Anthropic's terms govern the use of Claude Code in a product.
[Codex authentication](https://learn.chatgpt.com/docs/app-server#auth-endpoints) · [Claude Code terms](https://code.claude.com/docs/en/legal-and-compliance)

### Publish the DMG and update feed

Attach the verified DMG and checksum to a GitHub Release for the tested commit. Then
stage the final stapled DMG and generate the feed. Use the exact GitHub Release tag
in `tag`. To embed release notes, add an HTML or Markdown file with the DMG's base name
to the staging directory before running `generate_appcast`.

```sh
version=$(plutil -extract CFBundleShortVersionString raw -o - Resources/Info.plist)
tag="v$version"
staging=$(mktemp -d)
cp "build/Agent-S-macOS-${version}-arm64.dmg" "$staging/"
"$sparkle_bin/generate_appcast" --account agent-s --maximum-deltas 0 \
  --download-url-prefix "https://github.com/simular-ai/Agent-S/releases/download/$tag/" \
  --embed-release-notes -o "$staging/appcast.xml" "$staging"
```

Publish the generated `appcast.xml` verbatim as `website/appcast.xml`. The Pages workflow
deploys it when the change reaches `main`. Sparkle signs both the archive and the feed, so
regenerate the feed if its XML changes. Verify the deployed HTTPS feed and the release
download before announcing the update. [Sparkle publishing guide](https://sparkle-project.org/documentation/publishing/)

[Repository license](../../LICENSE)
