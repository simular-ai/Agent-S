import Foundation
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

    @Test func routesClaudeBrowserOpeningThroughOnePrivateAnnouncement() {
        let command = SandboxAccount.claudeBrowserSignInCommand
        #expect(command.hasPrefix(SandboxAccount.profileEnvironment(for: .claudeAccount)))
        #expect(command.contains("unset DISPLAY WAYLAND_DISPLAY CLAUDE_CODE_OAUTH_REFRESH_TOKEN CLAUDE_CODE_OAUTH_SCOPES"))
        #expect(command.contains(#"mktemp -d "$CLAUDE_CONFIG_DIR/browser-sign-in.XXXXXXXX""#))
        #expect(command.contains(#"export BROWSER="$browser_dir/open-browser""#))
        #expect(command.contains(#"/usr/bin/timeout --kill-after=2s 300s flock -n -E 75 -F "$CLAUDE_CONFIG_DIR/terminal-sign-in.lock""#))
        #expect(command.contains("claude auth login --claudeai </dev/null >/dev/null 2>&1"))
        #expect(command.contains("while IFS= read -r _; do :; done"))
        #expect(command.contains("'AGENT_S_CLAUDE_BROWSER '"))
        #expect(command.contains("trap cleanup EXIT"))
        #expect(SandboxAccount.signInCommand(for: .claudeAccount).isEmpty)
    }

    @Test func acceptsClaudeBrowserCallbackOnLoopback() throws {
        let current = try #require(ClaudeBrowserSignIn.parseAnnouncement(announcement()))
        #expect(current.authorizationURL.host == "claude.com")
        #expect(current.callbackPort == 49152)
        #expect(current.callbackHost == "localhost")
        let legacy = try #require(ClaudeBrowserSignIn.parseAnnouncement(announcement(
            destination: "https://claude.ai/oauth/authorize", redirect: "http://127.0.0.1:65535/callback"
        )))
        #expect(legacy.callbackPort == 65535)
        #expect(legacy.callbackHost == "127.0.0.1")
    }

    @Test(arguments: [
        "http://localhost:0/callback", "http://localhost:22/callback", "http://localhost:65536/callback",
        "http://localhost:49152/wrong", "http://localhost:49152/callback?destination=evil",
        "http://localhost:49152/callback#fragment", "https://localhost:49152/callback",
        "http://abc@localhost:49152/callback", "http://localhost.evil.example:49152/callback",
        "http://0.0.0.0:49152/callback", "https://platform.claude.com/oauth/code/callback",
    ])
    func rejectsUnrelatedBrowserRedirects(_ redirect: String) {
        #expect(ClaudeBrowserSignIn.parseAnnouncement(announcement(redirect: redirect)) == nil)
    }

    @Test(arguments: [
        "https://claude.com.bad.example/cai/oauth/authorize", "https://bad.example@claude.com/cai/oauth/authorize",
        "http://claude.com/cai/oauth/authorize", "https://claude.com:65000/cai/oauth/authorize",
        "https://claude.com/help", "https://platform.claude.com/oauth/authorize",
    ])
    func rejectsOtherAuthorizationOrigins(_ destination: String) {
        #expect(ClaudeBrowserSignIn.parseAnnouncement(announcement(destination: destination)) == nil)
    }

    @Test func rejectsRawAndMalformedBrowserAnnouncements() {
        let valid = announcement()
        #expect(ClaudeBrowserSignIn.parseAnnouncement(String(valid.split(separator: " ")[1])) == nil)
        #expect(ClaudeBrowserSignIn.parseAnnouncement(valid + "\nextra") == nil)
        #expect(ClaudeBrowserSignIn.parseAnnouncement(valid + "&redirect_uri=http://localhost:49152/callback") == nil)
        #expect(ClaudeBrowserSignIn.parseAnnouncement(valid + "&return_to=https://bad.example") == nil)
        #expect(ClaudeBrowserSignIn.parseAnnouncement(valid.replacingOccurrences(of: "S256", with: "plain")) == nil)
        #expect(ClaudeBrowserSignIn.parseAnnouncement(valid.replacingOccurrences(of: "9d1c", with: "339c")) == nil)
        #expect(ClaudeBrowserSignIn.parseAnnouncement(valid.replacingOccurrences(of: "&state=", with: "&unknown=")) == nil)
    }

    private func announcement(
        destination: String = "https://claude.com/cai/oauth/authorize",
        redirect: String = "http://localhost:49152/callback"
    ) -> String {
        var url = URLComponents(string: destination)!
        url.queryItems = [
            .init(name: "code", value: "true"),
            .init(name: "client_id", value: "9d1c250a-e61b-44d9-88ed-5944d1962f5e"),
            .init(name: "response_type", value: "code"),
            .init(name: "redirect_uri", value: redirect),
            .init(name: "scope", value: "user:profile user:inference"),
            .init(name: "code_challenge", value: String(repeating: "c", count: 43)),
            .init(name: "code_challenge_method", value: "S256"),
            .init(name: "state", value: String(repeating: "s", count: 43)),
        ]
        return "AGENT_S_CLAUDE_BROWSER " + url.string!
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
