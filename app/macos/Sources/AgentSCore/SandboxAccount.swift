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
            return profileEnvironment(for: provider) + requireCLI("claude") + claudeStatus
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

    /// Reconcile an earlier attempt only after the native login has released its lock.
    public static var claudeCompletedSignInStatusCommand: String {
        profileEnvironment(for: .claudeAccount) + requireCLI("claude") + requireCLI("flock") + #"""
        exec 9>> "$CLAUDE_CONFIG_DIR/terminal-sign-in.lock" || exit 1
        flock -s -n -E 75 9 || exit "$?"
        """# + "\n" + claudeStatus
    }

    private static let claudeStatus = #"""
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

    /// Runs native OAuth and relays the browser URL. Keep SSH stdin open for the lifetime of the sign-in.
    public static var claudeBrowserSignInCommand: String {
        profileEnvironment(for: .claudeAccount) + requireCLI("claude") + requireCLI("flock") + #"""
        unset DISPLAY WAYLAND_DISPLAY CLAUDE_CODE_OAUTH_REFRESH_TOKEN CLAUDE_CODE_OAUTH_SCOPES;
        browser_dir=$(mktemp -d "$CLAUDE_CONFIG_DIR/browser-sign-in.XXXXXXXX") || exit 1
        login_pid=
        watch_pid=
        output_pid=
        cleanup() {
            if [ -n "$watch_pid" ]; then
                kill -TERM "$watch_pid" 2>/dev/null || :
                wait "$watch_pid" 2>/dev/null || :
            fi
            if [ -n "$login_pid" ]; then
                kill -TERM "$login_pid" 2>/dev/null || :
                wait "$login_pid" 2>/dev/null || :
            fi
            if [ -n "$output_pid" ]; then
                kill -TERM "$output_pid" 2>/dev/null || :
                wait "$output_pid" 2>/dev/null || :
            fi
            rm -rf -- "$browser_dir"
        }
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' HUP TERM
        export AGENT_S_CLAUDE_BROWSER_DIR="$browser_dir"
        cat > "$browser_dir/open-browser" <<'AGENT_S_CLAUDE_BROWSER_HOOK' || exit 1
        #!/bin/sh
        [ "$#" -eq 1 ] && [ "${#1}" -le 4096 ] || exit 1
        printf '%s\n' "$1" > "$AGENT_S_CLAUDE_BROWSER_DIR/url.tmp" &&
            mv -- "$AGENT_S_CLAUDE_BROWSER_DIR/url.tmp" "$AGENT_S_CLAUDE_BROWSER_DIR/url"
        AGENT_S_CLAUDE_BROWSER_HOOK
        chmod 700 "$browser_dir/open-browser" || exit 1
        export BROWSER="$browser_dir/open-browser"

        # Drain native output in the guest and forward one fixed completion signal.
        mkfifo "$browser_dir/login-output" || exit 1
        (
            announced=0
            while IFS= read -r line; do
                case "$line" in
                    'Login successful.'|'Paste code here if prompted > Login successful.')
                        if [ "$announced" -eq 0 ]; then
                            printf '%s\n' '\#(ClaudeBrowserSignIn.completionAnnouncement)' || exit 1
                            announced=1
                        fi ;;
                esac
            done
        ) < "$browser_dir/login-output" &
        output_pid=$!
        /usr/bin/timeout --kill-after=2s 300s flock -n -E 75 -F "$CLAUDE_CONFIG_DIR/terminal-sign-in.lock" \
            claude auth login --claudeai </dev/null > "$browser_dir/login-output" 2>/dev/null &
        login_pid=$!
        # Losing a non-PTY SSH session must also release the guest's profile lock.
        session_pid=$$
        exec 3<&0
        (
            while IFS= read -r _; do :; done
            kill -TERM "$session_pid" 2>/dev/null || :
        ) <&3 3<&- &
        watch_pid=$!
        exec 3<&-
        while [ ! -f "$browser_dir/url" ] && kill -0 "$login_pid" 2>/dev/null; do
            sleep 0.1
        done
        if [ -f "$browser_dir/url" ]; then
            IFS= read -r authorization_url < "$browser_dir/url" || exit 1
            printf '%s%s\n' '\#(ClaudeBrowserSignIn.announcementPrefix)' "$authorization_url" || exit 1
        fi
        status=0
        wait "$login_pid" || status=$?
        login_pid=
        wait "$output_pid" || :
        output_pid=
        exit "$status"
        """#
    }

    /// Older Codex versions use the guest desktop terminal for device sign-in.
    public static func signInCommand(for provider: ModelProvider) -> String {
        guard provider == .codexAccount else { return "" }
        return profileEnvironment(for: provider) + requireCLI("codex") + #"""
        if [ ! -x /usr/bin/xterm ]; then printf '%s\n' 'The sandbox sign-in terminal is unavailable.' >&2; exit 1; fi;
        export DISPLAY=:0 XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}";
        if ! /usr/bin/xset q >/dev/null 2>&1; then printf '%s\n' 'The sandbox desktop is unavailable.' >&2; exit 1; fi;
        XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" systemctl --user start agent-s-browser.service >/dev/null 2>&1 || :
        """# + "\n"
            + "exec /usr/bin/setsid -f /usr/bin/xterm -hold -xrm 'XTerm*selectToClipboard: true' -T 'Sign in to ChatGPT' -e \(codex) login --device-auth </dev/null >/dev/null 2>&1"
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
