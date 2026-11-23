import Testing
@testable import AgentSCore

@Suite("Local VM model routing")
struct LocalVMProviderTests {
    @Test func choosesTheComputerSeparatelyFromTheProvider() {
        #expect(AgentLocation.allCases == [.linuxVM, .thisMac, .cloud])
        #expect(AgentLocation.cloud.displayName == "Cloud Computer")
        #expect(AgentLocation.cloud.rawValue == "cloud")
        #expect(AgentLocation.linuxVM.mode(for: .ollama) == .fullyPrivate)
        #expect(AgentLocation.linuxVM.mode(for: .openRouter) == .localVM)
        #expect(AgentLocation.linuxVM.mode(for: .claudeAccount) == .localVM)
        #expect(AgentLocation.linuxVM.mode(for: .codexAccount) == .localVM)
        #expect(AgentLocation.thisMac.mode(for: .ollama) == .thisMac)
        #expect(PrivacyMode.fullyPrivate.location == PrivacyMode.localVM.location)
        #expect(ModelProvider.allowed(at: .linuxVM, harness: .claudeCode, computer: nil) == [.ollama, .openRouter, .claudeAccount])
        #expect(ModelProvider.allowed(at: .linuxVM, harness: .codex, computer: nil) == [.ollama, .openRouter, .codexAccount])
        #expect(ModelProvider.allowed(at: .linuxVM, harness: .deepSeek, computer: nil) == [.ollama, .openRouter])
        #expect(ModelProvider.fitting(.ollama, at: .thisMac, harness: .claudeCode, computer: nil) == .ollama)
    }

    @Test func savedPrivateModeStillUsesOnlyOllama() {
        #expect(PrivacyMode(rawValue: "private") == .fullyPrivate)
        #expect(PrivacyMode.fullyPrivate.usesLocalComputer)
        #expect(PrivacyMode.localVM.usesLocalComputer)
        #expect(!PrivacyMode.thisMac.usesLocalComputer)
        #expect(!PrivacyMode.remote.usesLocalComputer)
        #expect(ModelProvider.fitting(.openRouter, in: .fullyPrivate, harness: .claudeCode, computer: nil) == .ollama)
        #expect(ModelProvider.fitting(.claudeAccount, in: .fullyPrivate, harness: .claudeCode, computer: nil) == .ollama)
        #expect(ModelProvider.fitting(.codexAccount, in: .fullyPrivate, harness: .codex, computer: nil) == .ollama)
    }

    @Test(arguments: [Harness.claudeCode, .codex, .deepSeek])
    func vmHarnessesUseTheOpenRouterTunnel(_ harness: Harness) {
        #expect(Harness.allowed(in: .localVM).contains(harness))
        #expect(ModelProvider.allowed(in: .localVM, harness: harness, computer: nil).contains(.openRouter))

        let baseURL = "http://127.0.0.1:11435"
        let shell = harness.remoteShell(prompt: "Hello", resume: nil, model: "test/model", ollamaURL: baseURL,
                                        provider: .openRouter, apiKey: "proxy-test-token")
        #expect(shell.contains(harness == .claudeCode ? baseURL : baseURL + "/v1"))
        #expect(shell.contains("proxy-test-token"))
        #expect(!shell.contains("http://127.0.0.1:11434"))
        if harness == .claudeCode { #expect(shell.contains("ANTHROPIC_API_KEY=''")) }
    }

    @Test func deepSeekKeepsProviderPatchesSeparate() {
        let privateShell = DeepSeekCommand(prompt: "Hello", resume: nil, model: "test").remoteShell
        let hosted = DeepSeekCommand(prompt: "Hello", resume: nil, model: "test/model",
                                     provider: .openRouter, guestPort: 11435)
        #expect(privateShell.contains("/home/sai/.dsh/agent-s.patch.yml"))
        #expect(hosted.remoteShell.contains("/home/sai/.dsh/agent-s-openrouter.patch.yml"))
    }

    @Test(arguments: [false, true])
    func nativeClaudeCodeUsesTheSandboxLogin(_ resumes: Bool) {
        let environment = SandboxAccount.profileEnvironment(for: .claudeAccount)
        let shell = Harness.claudeCode.remoteShell(prompt: "--help", resume: resumes ? "session-id" : nil,
                                                  model: "unused-model", ollamaURL: "http://127.0.0.1:11435",
                                                  provider: .claudeAccount, apiKey: "unused-token")
        #expect(!environment.isEmpty)
        #expect(shell.contains(environment + "DISPLAY=:0 ENABLE_TOOL_SEARCH=false claude -p"))
        #expect(shell.contains("cd ~/work && rm -f AGENTS.md"))
        #expect(shell.contains("--output-format stream-json --verbose --include-partial-messages"))
        #expect(shell.contains("--tools Bash,Read,Write,Edit"))
        #expect(shell.contains("--strict-mcp-config --no-chrome"))
        #expect(!shell.contains("ANTHROPIC_BASE_URL="))
        #expect(!shell.contains("ANTHROPIC_MODEL="))
        #expect(!shell.contains("unused-model"))
        #expect(!shell.contains("unused-token"))
        #expect(shell.hasSuffix(resumes ? "--resume 'session-id' -- '--help'" : "-- '--help'"))
    }

    @Test(arguments: [false, true])
    func nativeCodexUsesTheSandboxLogin(_ resumes: Bool) {
        let environment = SandboxAccount.profileEnvironment(for: .codexAccount)
        let shell = Harness.codex.remoteShell(prompt: "--help", resume: resumes ? "session-id" : nil,
                                             model: "unused-model", ollamaURL: "http://127.0.0.1:11435",
                                             provider: .codexAccount, apiKey: "unused-token")
        #expect(!environment.isEmpty)
        #expect(shell.contains("> AGENTS.md && " + environment + "DISPLAY=:0 codex exec --json"))
        #expect(shell.contains(#"-c 'model_provider="openai"' -c 'forced_login_method="chatgpt"'"#))
        #expect(shell.contains(#"-c 'cli_auth_credentials_store="file"'"#))
        #expect(!shell.contains("--oss"))
        #expect(!shell.contains("AGENT_S_PROXY_KEY="))
        #expect(!shell.contains("unused-model"))
        #expect(!shell.contains("unused-token"))
        #expect(shell.hasSuffix(resumes ? "resume 'session-id' -- '--help' </dev/null" : "-- '--help' </dev/null"))
    }
}
