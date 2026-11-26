import Foundation

/// The command that runs one DeepSeek Harness (`dsh`) turn inside the guest: headless, JSON
/// events, the model reached through the turn's tunnel (Ollama, or the Mac's OpenRouter proxy).
/// dsh reads AGENTS.md, so the turn writes Agent S’s instructions there, and a patch overlay that
/// sets the model and keeps everything else local: no session-log upload to DeepSeek, no
/// telemetry, no DeepSeek web search. Its shell runs unconfined (the computer is the sandbox),
/// which also drops the sandbox-escalation arguments small models confuse with the required ones.
public struct DeepSeekCommand: Equatable, Sendable {
    public var prompt: String
    public var resume: String?
    public var model: String
    public var provider: ModelProvider
    public var apiKey: String
    private let guestPort: UInt16
    private let allowsSteering: Bool

    /// Pinned: dsh is a developer preview with breaking changes between releases.
    public static let version = "0.2.2-rc.5"
    static let patchPath = "/home/sai/.dsh/agent-s.patch.yml"

    public init(prompt: String, resume: String?, model: String, provider: ModelProvider = .ollama,
                apiKey: String? = nil, guestPort: UInt16 = 11434, allowsSteering: Bool = false) {
        self.prompt = prompt
        self.resume = resume
        self.model = model
        self.provider = provider
        self.apiKey = apiKey ?? provider.placeholderKey
        self.guestPort = guestPort
        self.allowsSteering = allowsSteering
    }

    /// The overlay applied on top of dsh's headless profile.
    public var patch: String {
        // dsh's YAML can run JavaScript (`!!js`): keep the model id to plain id characters so it
        // can't break out of its quoted scalar.
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/@-")
        let id = String(model.filter { allowed.contains($0) })
        return """
        - id: llm-pi-ai
          config:
            providers:
              saige:
                apiKeyEnv: AGENT_S_PROXY_KEY
                api: openai-completions
                baseURL: http://127.0.0.1:\(guestPort)/v1
                models:
                  - id: "\(id)"
        - id: agent-default-model
          config:
            provider: saige
            model: "\(id)"
        - id: session-log-deepseek
          disabled: true
        - id: session-telemetry-otel
          disabled: true
        - id: web-search-deepseek
          disabled: true
        - id: bash-sandbox
          disabled: true
        - id: permission
          disabled: true
        - insert:
            - id: agent-s-live-text
              name: ./\(DeepSeekStreamingPlugin.filename)
            - id: bash-local
              name: "@deepseek-ai/dsh-bash-local"
        """
    }

    public var remoteShell: String {
        let patchPath = provider == .openRouter ? "/home/sai/.dsh/agent-s-openrouter.patch.yml" : Self.patchPath
        var dsh = ["dsh", "--profile", "headless", "--patch", patchPath, "--json"]
        if let resume { dsh += ["--session-id", quote(resume)] }
        dsh.append(quote(prompt))
        return "mkdir -p ~/work ~/.dsh && cd ~/work"
            + " && printf '%s\\n' \(quote(AgentInstructions.text(for: .deepSeek))) > AGENTS.md"
            + " && printf '%s\\n' \(quote(patch)) > \(patchPath)"
            + " && printf '%s\\n' \(quote(DeepSeekStreamingPlugin.source)) > ~/.dsh/\(DeepSeekStreamingPlugin.filename)"
            + " && DISPLAY=:0 DSH_TELEMETRY_MODE=DISABLED AGENT_S_PROXY_KEY=\(quote(apiKey)) "
            + (allowsSteering ? "AGENT_S_STEERING=1 " : "")
            + dsh.joined(separator: " ") + (allowsSteering ? "" : " </dev/null")
    }

    public func quote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

/// The Box profile runs the pinned SDK runtime with only the remote Box bridge rather than the
/// SDK's local filesystem, shell, or working-directory tools.
public enum DeepSeekBoxProfile {
    public static let patch = DeepSeekSDKProfile.patch(server: "box", environmentPrefix: "AGENT_S_BOX")
}

/// The Mac profile uses the same restricted runtime with the app's approval-gated Mac bridge.
public enum DeepSeekMacProfile {
    public static let patch = DeepSeekSDKProfile.patch(server: "agent_s", environmentPrefix: "AGENT_S_MAC",
                                                      toolCallTimeoutMs: 1_800_000)
}

/// DeepSeek Harness's SDK runtime: one long-lived `dsh --profile sdk-minimal` process per agent,
/// driven over newline-delimited JSON-RPC on stdio. Every turn is a `session/prompt` on it.
public enum DeepSeekSDKProfile {
    /// The model route the patch registers. `initialize` selects it with the agent's model.
    public static let provider = "agent-s"

    public static func arguments(entry: String?, patchPath: String) -> [String] {
        (entry.map { [$0] } ?? []) + ["--profile", "sdk-minimal", "--patch", patchPath]
    }

    /// A new client-chosen session identity in the form dsh's persistence uses.
    public static func newSessionID() -> String { "session-" + UUID().uuidString.lowercased() }

    /// Whether `id` is a session identity this app created or dsh reported.
    public static func isSessionID(_ id: String) -> Bool {
        id.hasPrefix("session-") && UUID(uuidString: String(id.dropFirst("session-".count))) != nil
    }

    static func patch(server: String, environmentPrefix: String, toolCallTimeoutMs: Int? = nil) -> String {
        let timeout = toolCallTimeoutMs.map { "\n        toolCallTimeoutMs: \($0)" } ?? ""
        return """
        - id: llm-deepseek
          disabled: true
        - id: session-log-deepseek
          disabled: true
        - id: plugin-package-inventory-deepseek
          disabled: true
        - id: deepseek-llm-api-extensions
          disabled: true
        - id: tool-fs
          disabled: true
        - id: tool-working-directory
          disabled: true
        - id: persistent-bash
          disabled: true
        - id: persistent-pwsh
          disabled: true
        - id: terminal-bash
          disabled: true
        - id: terminal-pwsh
          disabled: true
        - id: pty
          disabled: true
        - id: subprocess
          disabled: true
        - id: mcp-resources
          disabled: true
        - id: sessions
          config:
            root: !!js process.env.AGENT_S_DSH_SESSIONS
            compression: none
        - insert:
            - id: agent-s-live-text
              name: ./\(DeepSeekStreamingPlugin.filename)
            - id: llm-pi-ai
              name: '@deepseek-ai/dsh-llm-pi-ai'
              config:
                providers:
                  \(provider):
                    api: openai-completions
                    apiKeyEnv: AGENT_S_MODEL_TOKEN
                    baseURL: !!js process.env.AGENT_S_MODEL_URL
                    models:
                      - id: !!js process.env.AGENT_S_MODEL
                        contextWindow: !!js Number(process.env.AGENT_S_MODEL_CONTEXT)
                        maxTokens: !!js Number(process.env.AGENT_S_MODEL_OUTPUT)
            - id: mcp-\(server)
              name: '@deepseek-ai/dsh-mcp-client'
              config:
                serverName: \(server)
                transport: streamable-http
                url: !!js process.env.\(environmentPrefix)_URL
                headers:
                  Authorization: !!js process.env.\(environmentPrefix)_AUTHORIZATION
                startupTimeoutMs: 10000\(timeout)
                reconnect:
                  enabled: false
                failOnStartupError: true
        """
    }
}

/// Parses one DeepSeek Harness process's output and reconciles live text with committed messages.
public struct DeepSeekStream {
    private var liveAttempt: String?

    public init() {}

    public static func parse(line: String) -> [ClaudeEvent] {
        var stream = Self()
        return stream.consume(line: line)
    }

    public mutating func consume(line: String) -> [ClaudeEvent] {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [] }
        return consume(object: object)
    }

    /// One event in the headless line format, from either harness mode.
    public mutating func consume(object: [String: Any]) -> [ClaudeEvent] {
        guard let type = object["type"] as? String else { return [] }
        switch type {
        case "agent_s_steer_ready":
            return [.steeringReady]
        case "agent_s_steer_result":
            guard let id = object["id"] as? String, !id.isEmpty,
                  let accepted = object["accepted"] as? Bool else { return [] }
            return [.steeringResult(id: id, accepted: accepted)]
        case "agent_s_text_stream":
            guard let id = object["id"] as? String, !id.isEmpty else { return [] }
            liveAttempt = id
            return []
        case "agent_s_text_delta", "agent_s_text_snapshot":
            guard let id = object["id"] as? String, id == liveAttempt,
                  let text = object["text"] as? String else { return [] }
            if type == "agent_s_text_snapshot" {
                liveAttempt = nil
                return [.textSnapshot(id: id, text: text)]
            }
            return text.isEmpty ? [] : [.textDelta(id: id, text: text)]
        case "session":
            guard let id = object["sessionId"] as? String else { return [] }
            return [.started(sessionId: id, model: nil)]
        case "tool_call":
            let input = object["input"] as? [String: Any] ?? [:]
            let tool = object["tool"] as? String ?? "tool"
            let summary = (input["command"] as? String).map { ClaudeStream.summarize(tool: tool, input: ["command": $0]) }
                ?? ClaudeStream.summarize(tool: tool, input: input)
            return [.toolCall(id: object["callId"] as? String ?? "", name: tool, summary: summary,
                              detail: ToolActivity.detail(input: input))]
        case "tool_result":
            let result = object["result"].map { $0 as? String ?? String(describing: $0) } ?? ""
            return [.toolResult(id: object["callId"] as? String ?? "", output: ClaudeStream.clip(result),
                                isError: (object["status"] as? String) != "completed")]
        case "text":
            // The bundled plugin emits canonical snapshots of these same committed messages.
            guard liveAttempt == nil, let text = object["text"] as? String,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
            return [.text(text)]
        case "status":
            guard object["phase"] as? String == "step_end",
                  let usage = object["usage"] as? [String: Any] else { return [] }
            let input = ((usage["inputTokens"] as? Int) ?? 0) + ((usage["cacheReadTokens"] as? Int) ?? 0)
            let output = (usage["outputTokens"] as? Int) ?? 0
            return input + output > 0 ? [.usage(input: input, output: output)] : []
        case "final":
            // The answer already arrived as text events; this only says how the turn ended.
            let kind = (object["reason"] as? [String: Any])?["kind"] as? String
            return [.finished(result: kind == "completed" ? nil : "DeepSeek Harness stopped: \(kind ?? "unknown")",
                              isError: kind != "completed", costUSD: nil)]
        default:
            return []
        }
    }
}
