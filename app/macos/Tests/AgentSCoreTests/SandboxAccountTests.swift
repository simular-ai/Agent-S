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

    @Test func runsClaudeSignInThroughHostTerminal() {
        let command = SandboxAccount.claudeTerminalSignInCommand
        #expect(command.hasPrefix(SandboxAccount.profileEnvironment(for: .claudeAccount)))
        #expect(command.contains("unset DISPLAY WAYLAND_DISPLAY"))
        #expect(command.contains("CLAUDE_CODE_OAUTH_REFRESH_TOKEN CLAUDE_CODE_OAUTH_SCOPES"))
        #expect(command.contains("export BROWSER=/usr/bin/false"))
        #expect(command.contains(#"flock -n -E 75 "$CLAUDE_CONFIG_DIR/terminal-sign-in.lock" claude auth login --claudeai"#))
        #expect(command.hasSuffix("exec claude auth login --claudeai"))
        #expect(!command.contains("/usr/bin/xterm"))
        #expect(!command.contains("</dev/null"))
        #expect(SandboxAccount.signInCommand(for: .claudeAccount).isEmpty)
    }

    @Test func launchesCodexFallbackSignInWithGuestTerminal() {
        let command = SandboxAccount.signInCommand(for: .codexAccount)
        #expect(command.contains("login --device-auth"))
        #expect(command.contains("systemctl --user start agent-s-browser.service"))
        #expect(command.contains("/usr/bin/setsid -f /usr/bin/xterm"))
        #expect(command.contains("XTerm*selectToClipboard: true"))
    }

    @Test func startsDeviceCodeAppServerWithTheGuestCodexProfile() {
        let command = SandboxAccount.codexDeviceSignInCommand
        #expect(command.hasPrefix(SandboxAccount.profileEnvironment(for: .codexAccount)))
        #expect(command.contains(#"cli_auth_credentials_store="file""#))
        #expect(command.contains(#"forced_login_method="chatgpt""#))
        #expect(command.hasSuffix("app-server --listen stdio://"))
        #expect(!command.contains("/usr/bin/xterm"))
        #expect(!command.contains("</dev/null"))
    }
}
