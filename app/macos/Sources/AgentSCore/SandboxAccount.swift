import Foundation

/// Guest-native sign-in. Each CLI owns its credentials in a persistent guest profile.
public enum SandboxAccount {
    /// The shell prelude shared by sign-in, status, and turns. Includes the trailing ` && `.
    public static func profileEnvironment(for provider: ModelProvider) -> String {
        let variable: String
        let inheritedOverrides: String
        switch provider {
        case .claudeAccount:
            variable = "CLAUDE_CONFIG_DIR"
            inheritedOverrides = "ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL CLAUDE_CODE_OAUTH_TOKEN "
                + "ANTHROPIC_PROFILE ANTHROPIC_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL "
                + "ANTHROPIC_DEFAULT_OPUS_MODEL CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY"
        case .codexAccount:
            variable = "CODEX_HOME"
            inheritedOverrides = "OPENAI_API_KEY CODEX_API_KEY CODEX_ACCESS_TOKEN OPENAI_BASE_URL CHATGPT_BASE_URL"
        case .ollama, .openRouter:
            return ""
        }
        let directory = "\"$HOME/.agent-s/\(provider.rawValue)\""
        return "umask 077 && mkdir -p \(directory) && unset \(inheritedOverrides) && export \(variable)=\(directory) && "
    }

    /// Return only a sign-in bit and, where available, a recognized plan name.
    public static func statusCommand(for provider: ModelProvider) -> String {
        switch provider {
        case .claudeAccount:
            return profileEnvironment(for: provider) + requireCLI("claude") + #"""
            status=0; result=$(claude auth status --json 2>/dev/null) || status=$?;
            case "$status" in
                0) printf '%s' "$result" | node -e '
                    let output = { signedIn: false };
                    try {
                        const value = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
                        if (value.authMethod === "claude.ai") {
                            output.signedIn = true;
                            const plan = typeof value.subscriptionType === "string" ? value.subscriptionType.toLowerCase() : "";
                            if (plan === "pro" || plan === "max") output.plan = plan;
                        }
                    } catch {}
                    console.log(JSON.stringify(output));' ;;
                1) printf '%s\n' '{"signedIn":false}' ;;
                *) exit "$status" ;;
            esac
            """#
        case .codexAccount:
            return profileEnvironment(for: provider) + requireCLI("codex") + """
            status=0; result=$(\(codex) login status 2>&1) || status=$?;
            """ + "\n" + #"""
            case "$status" in
                0) if printf '%s\n' "$result" | grep -Fqx 'Logged in using ChatGPT'; then
                       printf '%s\n' '{"signedIn":true}'
                   else
                       printf '%s\n' '{"signedIn":false}'
                   fi ;;
                1) printf '%s\n' '{"signedIn":false}' ;;
                *) exit "$status" ;;
            esac
            """#
        case .ollama, .openRouter:
            return ""
        }
    }

    public static func plan(for provider: ModelProvider, output: String) -> HarnessAccountPlan? {
        guard provider.usesAccount,
              let status = try? JSONDecoder().decode(Status.self, from: Data(output.utf8)),
              status.signedIn else { return nil }
        return HarnessAccountPlan(provider: provider, reportedPlan: status.plan)
    }

    /// Managed device authorization over SSH stdio. The Codex process writes its own guest profile.
    public static var codexDeviceSignInCommand: String {
        profileEnvironment(for: .codexAccount) + requireCLI("codex")
            + "exec \(codex) app-server --listen stdio://"
    }

    /// Open the unmodified CLI's sign-in on the guest desktop and release the SSH connection.
    public static func signInCommand(for provider: ModelProvider) -> String {
        let executable: String
        let login: String
        let title: String
        switch provider {
        case .claudeAccount:
            executable = "claude"
            login = "claude auth login --claudeai"
            title = "Sign in to Claude Code"
        case .codexAccount:
            executable = "codex"
            login = "\(codex) login --device-auth"
            title = "Sign in to ChatGPT"
        case .ollama, .openRouter:
            return ""
        }
        return profileEnvironment(for: provider) + requireCLI(executable) + #"""
        if [ ! -x /usr/bin/xterm ]; then printf '%s\n' 'The sandbox sign-in terminal is unavailable.' >&2; exit 1; fi;
        export DISPLAY=:0 XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}";
        if ! /usr/bin/xset q >/dev/null 2>&1; then printf '%s\n' 'The sandbox desktop is unavailable.' >&2; exit 1; fi;
        XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" systemctl --user start agent-s-browser.service >/dev/null 2>&1 || :
        """# + "\n"
            + "exec /usr/bin/setsid -f /usr/bin/xterm -hold -xrm 'XTerm*selectToClipboard: true' -T '\(title)' -e \(login) </dev/null >/dev/null 2>&1"
    }

    private struct Status: Decodable {
        let signedIn: Bool
        let plan: String?
    }

    private static let codex = #"codex -c 'cli_auth_credentials_store="file"' -c 'forced_login_method="chatgpt"'"#

    private static func requireCLI(_ executable: String) -> String {
        "if ! command -v \(executable) >/dev/null 2>&1; "
            + "then printf '%s\\n' 'The sandbox \(executable) CLI is unavailable.' >&2; exit 127; fi;\n"
    }
}
