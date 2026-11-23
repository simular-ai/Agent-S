import Testing
@testable import AgentSCore

@Suite("Sandbox account commands")
struct SandboxAccountTests {
    @Test func usesSeparatePersistentGuestProfiles() {
        let claude = SandboxAccount.profileEnvironment(for: .claudeAccount)
        let codex = SandboxAccount.profileEnvironment(for: .codexAccount)
        #expect(claude.contains(#"export CLAUDE_CONFIG_DIR="$HOME/.agent-s/claude-account""#))
        #expect(codex.contains(#"export CODEX_HOME="$HOME/.agent-s/codex-account""#))
        #expect(claude.hasSuffix(" && "))
        #expect(codex.hasSuffix(" && "))
        #expect(SandboxAccount.profileEnvironment(for: .ollama).isEmpty)
        #expect(SandboxAccount.statusCommand(for: .openRouter).isEmpty)
        #expect(SandboxAccount.signInCommand(for: .ollama).isEmpty)
    }

    @Test func onlyConnectedAccountRoutesProducePlanLabels() {
        #expect(SandboxAccount.plan(for: .claudeAccount, output: #"{"signedIn":true,"plan":"max"}"#) == .subscription)
        #expect(SandboxAccount.plan(for: .claudeAccount, output: #"{"signedIn":true}"#) == .account)
        #expect(SandboxAccount.plan(for: .codexAccount, output: #"{"signedIn":true}"#) == .account)
        #expect(SandboxAccount.plan(for: .codexAccount, output: #"{"signedIn":false}"#) == nil)
        #expect(SandboxAccount.plan(for: .claudeAccount, output: "CLI blew up") == nil)
        #expect(SandboxAccount.plan(for: .openRouter, output: #"{"signedIn":true}"#) == nil)
    }

    @Test func launchesVendorSignInWithGuestTerminal() {
        let claude = SandboxAccount.signInCommand(for: .claudeAccount)
        let codex = SandboxAccount.signInCommand(for: .codexAccount)
        #expect(claude.contains("claude auth login --claudeai"))
        #expect(codex.contains("login --device-auth"))
        for command in [claude, codex] {
            #expect(command.contains("systemctl --user start agent-s-browser.service"))
            #expect(command.contains("/usr/bin/setsid -f /usr/bin/xterm"))
            #expect(command.contains("XTerm*selectToClipboard: true"))
        }
    }
}
