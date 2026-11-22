import Foundation

/// The command that runs one Claude Code turn inside the guest, over SSH.
///
/// Claude Code talks to Ollama on the host through Ollama's Anthropic-compatible endpoint
/// (`ANTHROPIC_BASE_URL`, Ollama >= 0.14). This is supported by Ollama, not by Anthropic.
/// Permissions are bypassed: the guest is a throwaway VM with only the shared folder and the web,
/// which is the isolation that makes that acceptable.
public struct ClaudeCommand: Equatable, Sendable {
    public var prompt: String
    /// The session to continue; nil starts one.
    public var resume: String?
    /// The Ollama model Claude Code runs on (also used for its background work).
    public var model: String
    /// Ollama as the guest reaches it: `http://127.0.0.1:11435`, tunnelled back over the turn's SSH.
    public var ollamaURL: String
    public var provider: ModelProvider
    public var contextTokens: Int { provider.contextTokens }
    public var maxOutputTokens: Int { provider.maxOutputTokens }

    /// Sent as the API key; for OpenRouter, the Mac proxy's per-launch token (not the real key).
    public var apiKey: String

    public init(prompt: String, resume: String?, model: String, ollamaURL: String, provider: ModelProvider = .ollama,
                apiKey: String? = nil) {
        self.prompt = prompt
        self.resume = resume
        self.model = model
        self.ollamaURL = ollamaURL
        self.provider = provider
        self.apiKey = apiKey ?? provider.placeholderKey
    }

    /// What the agent is told about where it is: the screen it drives and how to drive it.
    public static var systemPrompt: String { AgentInstructions.text(for: .claudeCode) }

    /// The longest answer a turn may write on the local model (see `ModelProvider`).
    public static let maxOutputTokens = ModelProvider.ollama.maxOutputTokens

    /// The built-in tools the agent gets: enough to write and run simulang scripts.
    public static let tools = "Bash,Read,Write,Edit"

    /// The remote shell line, every value single-quoted for the guest's shell.
    public var remoteShell: String {
        let env = [
            "DISPLAY=:0",
            "ANTHROPIC_BASE_URL=\(quote(ollamaURL))",
            "ANTHROPIC_AUTH_TOKEN=\(quote(apiKey))",
            "ANTHROPIC_MODEL=\(quote(model))",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL=\(quote(model))",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS=\(contextTokens)",
            // Claude Code keeps room for a full answer out of the window. Its default (~32K) uses up
            // the whole local window, so it compacted every few steps ("autocompact is thrashing").
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS=\(maxOutputTokens)",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1",
        ]
        var args = [
            "claude", "-p", quote(prompt),
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", "bypassPermissions",
            // A small request: Claude Code's default prompt, its full tool set, skills and MCP add
            // up to ~17.6K tokens a call, which a local model re-reads on every step (~10 s each
            // on an M5). Our own prompt and four tools come to ~2K.
            "--system-prompt", quote(Self.systemPrompt),
            "--tools", Self.tools,
            "--disable-slash-commands",
            "--strict-mcp-config",
        ]
        if let resume { args += ["--resume", quote(resume)] }
        // Codex's AGENTS.md (from a Codex turn) would add stale instructions; this prompt is the whole brief.
        return "cd ~/work && rm -f AGENTS.md && " + (env + args).joined(separator: " ")
    }

    /// POSIX single quotes: everything literal, an embedded `'` closed, escaped and reopened.
    public func quote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
