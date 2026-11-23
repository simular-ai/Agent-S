import Testing
@testable import AgentSCore

@Suite("Claude Code's stream")
struct ClaudeStreamTests {
    @Test func startsASession() {
        let line = #"{"type":"system","subtype":"init","session_id":"s-1","model":"qwen3-coder","tools":[]}"#
        #expect(ClaudeStream.parse(line: line) == [.started(sessionId: "s-1", model: "qwen3-coder")])
    }

    @Test func splitsAnAnswerIntoWordsAndToolCalls() {
        let line = #"{"type":"assistant","message":{"content":[{"type":"text","text":"Opening the site."},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"simulang run open.ts\nmore","description":"Open the site"}}]}}"#
        #expect(ClaudeStream.parse(line: line) == [
            .text("Opening the site."),
            .toolCall(id: "t1", name: "Bash", summary: "Open the site", detail: "simulang run open.ts\nmore"),
        ])
    }

    @Test func streamsTextBeforeEachCompletedBlock() {
        var stream = ClaudeStream()
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"message_start","message":{"id":"m1"}}}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hidden"}}}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"First"}}}"#) == [.textDelta(id: "m1/1", text: "First")])
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" "}}}"#) == [.textDelta(id: "m1/1", text: " ")])
        #expect(stream.consume(line: #"{"type":"assistant","message":{"id":"m1","content":[{"type":"text","text":"First block."}]}}"#) == [.textSnapshot(id: "m1/1", text: "First block.")])
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"content_block_stop","index":1}}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"stream_event","api_message_id":"m1","event":{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Second"}}}"#) == [.textDelta(id: "m1/2", text: "Second")])
        #expect(stream.consume(line: #"{"type":"assistant","message":{"id":"m1","content":[{"type":"text","text":"Second block."}]}}"#) == [.textSnapshot(id: "m1/2", text: "Second block.")])
        #expect(stream.consume(line: #"{"type":"assistant","message":{"id":"m2","content":[{"type":"text","text":"Completed only."}]}}"#) == [.text("Completed only.")])
    }

    @Test(arguments: [true, false])
    func discardsInterruptedPartialTextOnRetry(_ streamsRetry: Bool) {
        var stream = ClaudeStream()
        _ = stream.consume(line: #"{"type":"stream_event","event":{"type":"message_start","message":{"id":"m1"}}}"#)
        _ = stream.consume(line: #"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Interrupted"}}}"#)
        #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"message_stop"}}"#).isEmpty)
        let remove = ClaudeEvent.textSnapshot(id: "m1/0", text: "")
        let reply = #"{"type":"assistant","message":{"id":"m2","content":[{"type":"text","text":"Retried."}]}}"#
        if streamsRetry {
            #expect(stream.consume(line: #"{"type":"stream_event","event":{"type":"message_start","message":{"id":"m2"}}}"#) == [remove])
            #expect(stream.consume(line: reply) == [.text("Retried.")])
        } else {
            #expect(stream.consume(line: reply) == [remove, .text("Retried.")])
        }
    }

    @Test func readsToolResultsAndClipsThem() {
        let long = String(repeating: "x", count: ClaudeStream.maxToolOutput + 10)
        let line = #"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"\#(long)"}],"is_error":true}]}}"#
        guard case let .toolResult(id, output, isError)? = ClaudeStream.parse(line: line).first else {
            Issue.record("no tool result"); return
        }
        #expect(id == "t1")
        #expect(isError)
        #expect(output.hasSuffix("…"))
        #expect(output.count == ClaudeStream.maxToolOutput + 2)
    }

    @Test func endsTheTurn() {
        let line = #"{"type":"result","subtype":"success","is_error":false,"result":"Done.","total_cost_usd":0}"#
        #expect(ClaudeStream.parse(line: line) == [.finished(result: "Done.", isError: false, costUSD: 0)])
    }

    @Test func skipsWhatItDoesNotKnow() {
        #expect(ClaudeStream.parse(line: "not json").isEmpty)
        #expect(ClaudeStream.parse(line: #"{"type":"rate_limit_event"}"#).isEmpty)
        #expect(ClaudeStream.parse(line: #"{"type":"assistant","message":{"content":[{"type":"text","text":"  "}]}}"#).isEmpty)
    }

    @Test func summarizesByTheMostTellingField() {
        #expect(ClaudeStream.summarize(tool: "Read", input: ["file_path": "/home/sai/work/a.ts"]) == "/home/sai/work/a.ts")
        #expect(ClaudeStream.summarize(tool: "Glob", input: [:]) == "Glob")
    }
}

@Suite("Finding the guest")
struct GuestNetworkTests {
    let leases = """
    {
    \tname=saige
    \tip_address=192.168.64.7
    \thw_address=1,2:4b:0:aa:1:f
    \tlease=0x6740a1b2
    }
    {
    \tname=other
    \tip_address=192.168.64.3
    \thw_address=1,de:ad:be:ef:0:1
    }
    """

    @Test func findsTheLeaseByMACWhateverTheLeadingZeros() {
        #expect(GuestNetwork.address(for: "02:4b:00:aa:01:0f", inLeases: leases) == "192.168.64.7")
        #expect(GuestNetwork.address(for: "02:4b:00:aa:01:10", inLeases: leases) == nil)
    }

    @Test func findsTheGuestByHostnameWhenItsLeaseUsesADUID() {
        let leases = """
        {
        \tname=saige
        \tip_address=192.168.64.2
        \thw_address=ff,f1:f5:dd:7f:0:2:0:0:ab:11
        \tidentifier=ff,f1:f5:dd:7f:0:2:0:0:ab:11
        }
        {
        \tname=other
        \tip_address=192.168.64.3
        \thw_address=1,72:c4:8c:ae:92:4
        }
        """
        #expect(GuestNetwork.address(for: "72:c4:8c:ae:92:03", inLeases: leases) == nil)
        #expect(GuestNetwork.address(for: "72:c4:8c:ae:92:03", hostname: "saige", inLeases: leases) == "192.168.64.2")
        // A lease under the MAC wins over one under the hostname.
        #expect(GuestNetwork.address(for: "72:c4:8c:ae:92:04", hostname: "saige", inLeases: leases) == "192.168.64.3")
    }

    @Test func theHostIsTheGateway() {
        #expect(GuestNetwork.hostAddress(forGuest: "192.168.64.7") == "192.168.64.1")
        #expect(GuestNetwork.hostAddress(forGuest: "nope") == nil)
    }
}

@Suite("The guest's first boot")
struct GuestSetupTests {
    @Test func givesTheAppItsKeyAndInstallsTheTools() {
        let data = GuestSetup.userData(sshPublicKey: "ssh-ed25519 AAAA test\n")
        #expect(data.hasPrefix("#cloud-config\n"))
        #expect(data.contains("      - ssh-ed25519 AAAA test\n"))
        #expect(data.contains("@simular-ai/simulang @anthropic-ai/claude-code @openai/codex"))
        #expect(data.contains("[sai, /mnt/sai, virtiofs"))
        #expect(data.contains("touch /var/lib/sai-ready"))
    }

    @Test func firstBootInstallsSupportedNodeBeforeHarnesses() {
        let data = GuestSetup.userData(sshPublicKey: "ssh-ed25519 AAAA test")
        #expect(data.contains("runcmd:\n  - |\n    (\n"))
        #expect(data.contains("https://deb.nodesource.com/setup_24.x"))
        #expect(data.contains("env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs &&"))
        #expect(data.contains(") && (command -v simulang"))
        #expect(data.contains("\n  - touch \(GuestSetup.readyMarker)"))
        #expect(!data.contains("sudo npm install"))
    }
}

@Suite("Running Claude Code in the guest")
struct ClaudeCommandTests {
    @Test func pointsClaudeCodeAtOllamaAndStreams() {
        let shell = ClaudeCommand(prompt: "hi", resume: nil, model: "qwen3-coder",
                                  ollamaURL: "http://192.168.64.1:11434").remoteShell
        #expect(shell.hasPrefix("cd ~/work && rm -f AGENTS.md && DISPLAY=:0 "))
        #expect(shell.contains("ANTHROPIC_BASE_URL='http://192.168.64.1:11434'"))
        #expect(shell.contains("ANTHROPIC_AUTH_TOKEN='ollama'"))
        #expect(shell.contains("--output-format stream-json --verbose --include-partial-messages"))
        #expect(!shell.contains("--resume"))
        // Replaces Claude Code's large default prompt and tool set to keep each call small.
        #expect(shell.contains("--system-prompt '"))
        #expect(!shell.contains("--append-system-prompt"))
        #expect(shell.contains("--tools Bash,Read,Write,Edit --disable-slash-commands --strict-mcp-config"))
        #expect(shell.contains("CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096"))
    }

    @Test func usesHostedLimitsAndAPlaceholderKeyForOpenRouter() {
        let shell = ClaudeCommand(prompt: "hi", resume: nil, model: "anthropic/claude-sonnet-5.5",
                                  ollamaURL: "http://127.0.0.1:11434", provider: .openRouter).remoteShell
        #expect(shell.contains("ANTHROPIC_AUTH_TOKEN='agent-s-proxy'"))
        #expect(shell.contains("ANTHROPIC_MODEL='anthropic/claude-sonnet-5.5'"))
        #expect(shell.contains("CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000"))
        #expect(shell.contains("CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384"))
    }

    @Test func quotesWhatTheUserTyped() {
        let command = ClaudeCommand(prompt: "it's $HOME; rm -rf /", resume: "s-1", model: "m",
                                    ollamaURL: "http://h:11434")
        #expect(command.remoteShell.contains(#"-p 'it'\''s $HOME; rm -rf /'"#))
        #expect(command.remoteShell.contains("--resume 's-1'"))
    }
}

@Suite("Running Codex in the guest")
struct CodexTests {
    @Test func runsCodexOnOllamaWithInstructionsInAgentsFile() {
        let shell = CodexCommand(prompt: "hi", resume: nil, model: "gemma4").remoteShell
        #expect(shell.hasPrefix("cd ~/work && printf '%s\\n' 'You are Sai."))
        #expect(shell.contains("> AGENTS.md && DISPLAY=:0 AGENT_S_PROXY_KEY='ollama' codex exec --json --oss --local-provider ollama -m 'gemma4'"))
        #expect(shell.contains("view_image"))
        #expect(shell.contains(#"exec_command(cmd: "web snapshot")"#))
        #expect(shell.contains(#"-c 'web_search="disabled"'"#))
        #expect(shell.hasSuffix("'hi' </dev/null"))
    }

    @Test func reachesOpenRouterThroughTheTunnelWithoutAKey() {
        let shell = CodexCommand(prompt: "hi", resume: nil, model: "anthropic/claude-sonnet-5.5", provider: .openRouter).remoteShell
        #expect(!shell.contains("--oss"))
        #expect(shell.contains(#"-c 'model_provider="saige"'"#))
        #expect(shell.contains(#"base_url="http://127.0.0.1:11434/v1""#))
        #expect(shell.contains(#"wire_api="responses""#))
        #expect(shell.contains("AGENT_S_PROXY_KEY='agent-s-proxy'"))
        // The Mac proxy's per-launch token, when given, replaces the placeholder.
        let withToken = CodexCommand(prompt: "hi", resume: nil, model: "m", provider: .openRouter, apiKey: "agent-s-abc").remoteShell
        #expect(withToken.contains("AGENT_S_PROXY_KEY='agent-s-abc'"))
        let claude = ClaudeCommand(prompt: "hi", resume: nil, model: "m", ollamaURL: "u", provider: .openRouter, apiKey: "agent-s-abc").remoteShell
        #expect(claude.contains("ANTHROPIC_AUTH_TOKEN='agent-s-abc'"))
    }

    @Test func resumesTheThread() {
        let shell = CodexCommand(prompt: "next", resume: "t-1", model: "m").remoteShell
        #expect(shell.contains("codex exec --json --oss"))
        #expect(shell.contains("--skip-git-repo-check resume 't-1' 'next'"))
    }

    @Test func parsesCodexEvents() {
        #expect(CodexStream.parse(line: #"{"type":"thread.started","thread_id":"t-1"}"#) == [.started(sessionId: "t-1", model: nil)])
        #expect(CodexStream.parse(line: #"{"type":"item.started","item":{"id":"item_2","type":"command_execution","command":"/bin/bash -lc 'echo hi\necho again'","aggregated_output":"","exit_code":null,"status":"in_progress"}}"#)
                == [.toolCall(id: "item_2", name: "Shell", summary: "echo hi", detail: "/bin/bash -lc 'echo hi\necho again'")])
        #expect(CodexStream.parse(line: #"{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"x","aggregated_output":"hi\n","exit_code":1,"status":"completed"}}"#)
                == [.toolResult(id: "item_2", output: "hi\n", isError: true)])
        #expect(CodexStream.parse(line: #"{"type":"item.completed","item":{"id":"item_4","type":"agent_message","text":"Done."}}"#) == [.text("Done.")])
        #expect(CodexStream.parse(line: #"{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Model metadata for `gemma4` not found."}}"#).isEmpty)
        #expect(CodexStream.parse(line: #"{"type":"turn.completed","usage":{}}"#) == [.finished(result: nil, isError: false, costUSD: nil)])
        #expect(CodexStream.parse(line: #"{"type":"turn.completed","usage":{"input_tokens":17153,"cached_input_tokens":8507,"output_tokens":128}}"#)
                == [.usage(input: 17153, output: 128), .finished(result: nil, isError: false, costUSD: nil)])
    }
}

@Suite("The guest's screen tools")
struct GuestToolsTests {
    @Test func installsUiAndItsScript() {
        let line = GuestTools.installCommand
        #expect(line.contains("sudo tee /usr/local/bin/ui >/dev/null && sudo chmod 755 /usr/local/bin/ui"))
        #expect(line.contains("sudo mkdir -p /usr/local/lib/sai"))
        #expect(line.contains("sudo tee /usr/local/lib/sai/ui.ts"))
        #expect(line.contains("command -v xterm >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q xterm"))
        // The clipboard agent must start in the desktop session, never from this SSH command.
        #expect(line.contains("pgrep -x spice-vdagent >/dev/null || sudo systemctl restart getty@tty1"))
        #expect(!line.contains("DISPLAY=:0 $(tr"))
    }

    @Test func existingGuestChecksAndRepairsOnlyTheDeepSeekRuntime() {
        let ready = GuestTools.deepSeekReadyCommand
        let install = GuestTools.installDeepSeekCommand
        #expect(ready.contains("/usr/bin/timeout --kill-after=2s 8s"))
        #expect(ready.contains("exit \(GuestTools.deepSeekUpdateRequiredStatus)"))
        #expect(ready.contains("major < 24 || (major === 24 && minor < 9)"))
        #expect(ready.contains("npm --version"))
        #expect(ready.contains("dsh_version=$(dsh --version 2>/dev/null) &&"))
        #expect(ready.contains("= '\(DeepSeekCommand.version)'"))
        #expect(install.contains("https://deb.nodesource.com/setup_24.x"))
        #expect(install.contains("sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs"))
        #expect(install.contains("sudo npm install -g --engine-strict @deepseek-ai/dsh@\(DeepSeekCommand.version) &&"))
        #expect(!install.contains("systemctl"))
        #expect(GuestTools.installCommand.hasPrefix(GuestSetup.ensureNodeCommand(asRoot: false)))
        #expect(GuestTools.installCommand.contains(GuestSetup.ensureBaseHarnessesCommand(asRoot: false)))
        #expect(GuestTools.installCommand.contains("sudo npm install -g --engine-strict @deepseek-ai/dsh@\(DeepSeekCommand.version)"))
    }

    @Test func installsDesktopLauncherInTheScriptsOnlyRefresh() {
        let refresh = GuestTools.installScriptsCommand
        #expect(GuestTools.ui.contains(#"exec /usr/local/lib/sai/launch-desktop-app "$@""#))
        #expect(GuestTools.uiScript.contains("ui launch <app>"))
        #expect(refresh.contains("sudo tee /usr/local/lib/sai/launch-desktop-app"))
        #expect(refresh.contains("systemctl --user daemon-reload"))
        #expect(!refresh.contains("apt-get install"))
        #expect(!refresh.contains("systemctl restart"))
        #expect(GuestTools.installCommand.contains("command -v gio"))
        #expect(GuestTools.installCommand.contains("libglib2.0-bin"))
    }

    @Test func launchesDesktopAppsOutsideTheTurnScope() {
        let launcher = GuestTools.desktopLauncher
        #expect(launcher.contains("pgrep -u \"$uid\" -xo openbox"))
        #expect(launcher.contains("env -u DBUS_SESSION_BUS_ADDRESS"))
        #expect(launcher.contains("--setenv=\"DBUS_SESSION_BUS_ADDRESS=$desktop_bus\""))
        #expect(launcher.contains("--service-type=exec"))
        #expect(launcher.contains("--property=ExitType=cgroup"))
        #expect(launcher.contains("--expand-environment=no"))
        #expect(launcher.contains("--collect"))
        #expect(launcher.contains(#""${options[@]}" -- "${command[@]}""#))
        #expect(!launcher.contains("--scope"))
        #expect(!launcher.contains("--wait"))
        #expect(!launcher.contains("--pipe"))
    }

    @Test func webDrivesChromiumOverLocalDebuggingPort() {
        #expect(GuestTools.browserLauncher.contains("--remote-debugging-port=9222 --remote-debugging-address=127.0.0.1"))
        #expect(GuestTools.webScript.contains("ariaSnapshot({ mode: 'ai' })"))
        #expect(GuestTools.installCommand.contains("playwright-core@1.65.1"))
        #expect(GuestTools.ui.contains("exec web open"))
    }

    @Test func chromiumRunsInTheGuestUserServiceWithDesktopAccessibility() {
        #expect(GuestTools.installCommand.contains("sudo tee /etc/systemd/user/agent-s-browser.service"))
        #expect(GuestTools.installCommand.contains("systemctl --user daemon-reload"))
        #expect(GuestTools.browserService.contains("ExecStart=/usr/local/lib/sai/start-browser"))
        #expect(GuestTools.browserLauncher.contains("DBUS_SESSION_BUS_ADDRESS"))
        #expect(GuestTools.browserLauncher.contains("--force-renderer-accessibility"))
        #expect(!GuestTools.browserLauncher.contains("--user-data-dir"))
        #expect(GuestTools.uiScript.contains("AccessibilitySnapshot.fromWindow"))
    }

    @Test func webWaitsForTheBrowserWithoutKillingOrRestartingIt() {
        #expect(GuestTools.web.contains("systemctl --user start agent-s-browser.service"))
        #expect(GuestTools.web.contains("if ! wait_for_browser"))
        #expect(!GuestTools.web.contains("pkill"))
        #expect(!GuestTools.web.contains("--user restart"))
    }
}

@Suite("Privacy modes")
struct PrivacyModeTests {
    @Test func fullyPrivateKeepsEverythingLocal() {
        #expect(PrivacyMode.fullyPrivate.providers == [.ollama])
        #expect(Harness.allowed(in: .fullyPrivate) == [.claudeCode, .codex, .deepSeek])
    }

    @Test func remoteAllowsOpenRouterAndSai() {
        #expect(PrivacyMode.remote.providers.contains(.openRouter))
        #expect(Harness.allowed(in: .remote).contains(.sai))
        #expect(!Harness.sai.runsInComputer)
        // Cloud agents use Sai computers or route their CLI harness's tools to a Box.
        #expect(Harness.allowed(in: .remote, computer: .sai) == [.sai])
        #expect(Harness.allowed(in: .remote, computer: .box) == [.claudeCode, .codex, .deepSeek])
        #expect(Harness.allowed(in: .fullyPrivate, computer: nil) == [.claudeCode, .codex, .deepSeek])
    }

    @Test func hostedAccountsAreLimitedToCompatibleHarnesses() {
        #expect(ModelProvider.allowed(in: .thisMac, harness: .claudeCode, computer: nil) == [.ollama, .openRouter, .claudeAccount])
        #expect(ModelProvider.allowed(in: .thisMac, harness: .codex, computer: nil) == [.codexAccount])
        #expect(ModelProvider.allowed(in: .remote, harness: .claudeCode, computer: .box).contains(.claudeAccount))
        #expect(ModelProvider.allowed(in: .remote, harness: .codex, computer: .box) == [.ollama, .openRouter, .codexAccount])
        #expect(ModelProvider.allowed(in: .remote, harness: .deepSeek, computer: .box) == [.ollama, .openRouter])
        #expect(ModelProvider.fitting(.claudeAccount, in: .remote, harness: .codex, computer: .box) == .codexAccount)
        #expect(ModelProvider.fitting(.openRouter, in: .remote, harness: .codex, computer: .box) == .openRouter)
        #expect(ModelProvider.allowed(in: .remote, harness: .sai, computer: .sai).isEmpty)
        for harness in Harness.allowed(in: .fullyPrivate) {
            #expect(ModelProvider.allowed(in: .fullyPrivate, harness: harness, computer: nil) == [.ollama])
        }
    }

    @Test func thisMacSupportsDeepSeekWithOllamaAndOpenRouter() {
        #expect(Harness.allowed(in: .thisMac) == [.claudeCode, .codex, .deepSeek])
        #expect(Harness.allowed(in: .thisMac, computer: nil) == [.claudeCode, .codex, .deepSeek])
        #expect(ModelProvider.allowed(at: .thisMac, harness: .deepSeek, computer: nil) == [.ollama, .openRouter])
        #expect(ModelProvider.allowed(in: .thisMac, harness: .deepSeek, computer: nil) == [.ollama, .openRouter])
        #expect(ModelProvider.fitting(.claudeAccount, in: .thisMac, harness: .deepSeek, computer: nil) == .ollama)
        #expect(ModelProvider.fitting(.codexAccount, in: .thisMac, harness: .deepSeek, computer: nil) == .ollama)
    }

    @Test func boxToolBridgeBindsToolsToTheConfiguredBox() {
        #expect(Box.allowsAgentTool("shell"))
        #expect(Box.allowsAgentTool("browser_navigate"))
        #expect(!Box.allowsAgentTool("box_delete"))
        #expect(!Box.allowsAgentTool("exec_command"))
        let arguments = ["command": "pwd", "box": "different"]
        let namedBox = Box.scopedAgentArguments(arguments, acceptsBox: true, box: "work")
        #expect(namedBox?["box"] as? String == "work")
        #expect(namedBox?["command"] as? String == "pwd")
        #expect(Box.scopedAgentArguments(arguments, acceptsBox: false, box: "work") == nil)
        #expect(Box.scopedAgentArguments(arguments, acceptsBox: true, box: nil)?["box"] == nil)
    }

    @Test func incompatibleAccountsNeverProduceGuestCommands() {
        for (harness, provider) in [(Harness.claudeCode, ModelProvider.codexAccount), (.codex, .claudeAccount),
                                    (.deepSeek, .claudeAccount), (.deepSeek, .codexAccount), (.sai, .codexAccount)] {
            #expect(harness.remoteShell(prompt: "Hello", resume: nil, model: "Automatic",
                                       ollamaURL: "http://127.0.0.1:11434", provider: provider).isEmpty)
        }
    }
}

@Suite("Running DeepSeek Harness in the guest")
struct DeepSeekTests {
    @Test func runsHeadlessWithJSONAndALocalOnlyOverlay() {
        let command = DeepSeekCommand(prompt: "hi", resume: nil, model: "gemma4")
        let shell = command.remoteShell
        #expect(shell.contains("dsh --profile headless --patch /home/sai/.dsh/agent-s.patch.yml --json 'hi' </dev/null"))
        #expect(shell.contains("> AGENTS.md"))
        #expect(shell.contains(#"bash(command: "web snapshot", description: "web snapshot")"#))
        #expect(shell.contains("DSH_TELEMETRY_MODE=DISABLED AGENT_S_PROXY_KEY='ollama'"))
        // Nothing goes to DeepSeek: no session-log upload, telemetry or DeepSeek web search.
        for plugin in ["session-log-deepseek", "session-telemetry-otel", "web-search-deepseek"] {
            #expect(command.patch.contains("- id: \(plugin)\n  disabled: true"))
        }
        #expect(command.patch.contains("baseURL: http://127.0.0.1:11434/v1"))
        #expect(command.patch.contains(#"model: "gemma4""#))
        #expect(command.patch.contains("@deepseek-ai/dsh-bash-local"))
    }

    @Test func modelIdCannotInjectConfig() {
        let patch = DeepSeekCommand(prompt: "hi", resume: nil, model: "m\"\n- id: x\n  config: !!js process.exit(1)").patch
        #expect(patch.contains(#"model: "m-id:xconfig:jsprocess.exit1""#))
        #expect(!patch.contains("!!js"))
    }

    @Test func resumesTheSession() {
        let shell = DeepSeekCommand(prompt: "next", resume: "session-1", model: "m").remoteShell
        #expect(shell.contains("--json --session-id 'session-1' 'next'"))
    }

    @Test func parsesDshEvents() {
        #expect(DeepSeekStream.parse(line: #"{"type":"session","sessionId":"session-1","cwd":"/home/sai/work"}"#)
                == [.started(sessionId: "session-1", model: nil)])
        #expect(DeepSeekStream.parse(line: #"{"type":"tool_call","callId":"c1","tool":"bash","input":{"command":"web open https://x.com","description":"Open X"}}"#)
                == [.toolCall(id: "c1", name: "bash", summary: "web open https://x.com", detail: "web open https://x.com")])
        #expect(DeepSeekStream.parse(line: #"{"type":"tool_result","callId":"c1","status":"error","result":"Error: nope"}"#)
                == [.toolResult(id: "c1", output: "Error: nope", isError: true)])
        #expect(DeepSeekStream.parse(line: #"{"type":"text","text":"Done."}"#) == [.text("Done.")])
        #expect(DeepSeekStream.parse(line: #"{"type":"status","phase":"step_end","usage":{"inputTokens":10,"outputTokens":5,"cacheReadTokens":90}}"#)
                == [.usage(input: 100, output: 5)])
        #expect(DeepSeekStream.parse(line: #"{"type":"final","reason":{"kind":"completed"},"text":"Done."}"#)
                == [.finished(result: nil, isError: false, costUSD: nil)])
    }
}

@Suite("Box agents")
struct BoxTests {
    @Test func claudeCodeInTheAppHasOnlyBoxTools() {
        let command = BoxClaudeCommand(prompt: "hi", resume: nil, model: "anthropic/claude-sonnet-5.5", provider: .openRouter,
                                       apiKey: "agent-s-token", boxName: nil, baseURL: "http://127.0.0.1:11435")
        let args = command.arguments(mcpConfigPath: "/tmp/mcp.json")
        let tools = args.firstIndex(of: "--tools")!
        #expect(args[tools + 1] == "")
        #expect(args.contains("--strict-mcp-config"))
        #expect(args[args.firstIndex(of: "--setting-sources")! + 1] == "")
        let env = command.environment(home: "/Users/x", configDir: "/cfg", boxToken: "cb_x")
        #expect(env["CLAUDE_CONFIG_DIR"] == "/cfg")
        #expect(env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11435")
        #expect(Box.mcpConfig.contains("${BOX_TOKEN}") && !Box.mcpConfig.contains("cb_"))
        #expect(args[args.firstIndex(of: "--disallowedTools")! + 1].contains("mcp__box__box_delete"))
        #expect(Box.onlyBoxTools(["mcp__box__shell", "mcp__box__browser_click"]))
        #expect(!Box.onlyBoxTools(["mcp__box__shell", "Bash"]))
    }
}
