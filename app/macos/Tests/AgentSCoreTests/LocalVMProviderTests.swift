import Testing
@testable import AgentSCore

@Suite("Local VM model routing")
struct LocalVMProviderTests {
    @Test func choosesTheComputerSeparatelyFromTheProvider() {
        #expect(AgentLocation.allCases == [.linuxVM, .thisMac, .cloud])
        #expect(AgentLocation.linuxVM.mode(for: .ollama) == .fullyPrivate)
        #expect(AgentLocation.linuxVM.mode(for: .openRouter) == .localVM)
        #expect(AgentLocation.thisMac.mode(for: .ollama) == .thisMac)
        #expect(PrivacyMode.fullyPrivate.location == PrivacyMode.localVM.location)
        #expect(ModelProvider.allowed(at: .linuxVM, harness: .claudeCode, computer: nil) == [.ollama, .openRouter])
        #expect(ModelProvider.fitting(.ollama, at: .thisMac, harness: .claudeCode, computer: nil) == .ollama)
    }

    @Test func savedPrivateModeStillUsesOnlyOllama() {
        #expect(PrivacyMode(rawValue: "private") == .fullyPrivate)
        #expect(PrivacyMode.fullyPrivate.usesLocalComputer)
        #expect(PrivacyMode.localVM.usesLocalComputer)
        #expect(!PrivacyMode.thisMac.usesLocalComputer)
        #expect(!PrivacyMode.remote.usesLocalComputer)
        #expect(ModelProvider.fitting(.openRouter, in: .fullyPrivate, harness: .claudeCode, computer: nil) == .ollama)
    }

    @Test(arguments: [Harness.claudeCode, .codex, .deepSeek])
    func vmHarnessesUseTheOpenRouterTunnel(_ harness: Harness) {
        #expect(Harness.allowed(in: .localVM).contains(harness))
        #expect(ModelProvider.allowed(in: .localVM, harness: harness, computer: nil) == [.openRouter])

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
}
