import Foundation

/// The agent program that runs a turn inside the guest. Both talk to the same local model through
/// Ollama and drive the same screen; they differ in their tools, prompts and output format.
public enum Harness: String, CaseIterable, Identifiable, Codable, Sendable {
    case claudeCode = "claude-code"
    case codex
    /// DeepSeek Harness (`dsh`, open source): runs in the computer like Claude Code and Codex.
    case deepSeek = "deepseek"
    /// Simular's Sai agent through the Sai API: runs on one of the user's Sai computers, called
    /// from the Mac (no shell in the local computer). Cloud mode only.
    case sai

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .claudeCode: "Claude Code"
        case .codex: "Codex"
        case .deepSeek: "DeepSeek"
        case .sai: "Sai"
        }
    }

    /// Runs inside Agent S’s local computer (over SSH), rather than through a remote API.
    public var runsInComputer: Bool { self != .sai }

    /// Every harness a privacy mode can use, across its computers.
    public static func allowed(in mode: PrivacyMode) -> [Harness] {
        switch mode {
        case .fullyPrivate, .localVM: [.claudeCode, .codex, .deepSeek]
        case .thisMac: [.claudeCode, .codex]
        case .remote: CloudComputer.box.harnesses + CloudComputer.sai.harnesses
        }
    }

    /// The remote shell line for one turn.
    public func remoteShell(prompt: String, resume: String?, model: String, ollamaURL: String,
                            provider: ModelProvider = .ollama, apiKey: String? = nil, allowsSteering: Bool = false) -> String {
        let mode = AgentLocation.linuxVM.mode(for: provider)
        guard ModelProvider.allowed(in: mode, harness: self, computer: nil).contains(provider) else { return "" }
        let guestPort = URLComponents(string: ollamaURL)?.port.flatMap { UInt16(exactly: $0) } ?? 11434
        return switch self {
        case .claudeCode:
            ClaudeCommand(prompt: prompt, resume: resume, model: model, ollamaURL: ollamaURL, provider: provider,
                          apiKey: apiKey).remoteShell
        case .codex:
            CodexCommand(prompt: prompt, resume: resume, model: model, provider: provider,
                         apiKey: apiKey, guestPort: guestPort).remoteShell
        case .deepSeek:
            DeepSeekCommand(prompt: prompt, resume: resume, model: model, provider: provider,
                            apiKey: apiKey, guestPort: guestPort, allowsSteering: allowsSteering).remoteShell
        case .sai:
            "" // not a shell: AgentSession calls the Sai API instead
        }
    }

    /// Events from one line of the harness's JSON output.
    public func parse(line: String) -> [ClaudeEvent] {
        switch self {
        case .claudeCode: ClaudeStream.parse(line: line)
        case .codex: CodexStream.parse(line: line)
        case .deepSeek: DeepSeekStream.parse(line: line)
        case .sai: []
        }
    }
}

/// What the agent is told about where it is and how to use the screen, for either harness.
public enum AgentInstructions {
    static func shellToolName(_ harness: Harness) -> String {
        switch harness {
        case .claudeCode: "the Bash tool"
        case .codex: "the exec_command tool"
        case .deepSeek: "the bash tool (always with both a command and a short description)"
        case .sai: "your shell"
        }
    }

    public static func text(for harness: Harness) -> String {
        let look = switch harness {
        case .claudeCode: "open that file with the Read tool (not Bash)"
        case .codex: "open that file with the view_image tool"
        case .deepSeek: "open that file with the read_image tool"
        case .sai: "open that file"
        }
        // Small models need the tool named the way their harness names it, and each command shown
        // as a call to it: gemma4 otherwise calls "ui" as if it were a tool of its own.
        let run: (String) -> String = { command in
            switch harness {
            case .claudeCode: "Bash(command: \"\(command)\")"
            // Codex's shell tool; gemma4 otherwise calls "web open …" as a function of its own.
            case .codex: "exec_command(cmd: \"\(command)\")"
            // dsh's bash tool requires a short description beside the command.
            case .deepSeek: "bash(command: \"\(command)\", description: \"\(command.split(separator: " ").prefix(2).joined(separator: " "))\")"
            case .sai: command
            }
        }
        return """
        You are Sai. You control a real Linux computer with a screen (1440x900, Chromium browser). \
        You CAN see and use this screen: do not say you are command-line only. Do the user's task on \
        this computer. There are no "web" or "ui" tools: they are shell commands, so run them \
        through \(shellToolName(harness)), one command per call.

        Websites (use these for anything in the browser):
        - \(run("web open https://example.com"))   go to a website
        - \(run("web snapshot"))   the page, one element per line with [ref=eN] (links, buttons, \
        text fields, text)
        - \(run("web click e12"))   click the element with ref=e12
        - \(run("web type e21 'hello' --enter"))   fill the text field with ref=e21, then press \
        Enter (leave out --enter to just type)
        - \(run("web read"))   the page's visible text, to read or summarize it
        - \(run("web scroll down"))   see more of the page; also web back, web key Enter, web tabs
        On a follow-up, inspect the current tab with web snapshot before navigating. Leave the browser \
        and its tabs open unless the user asks you to close them.

        Anything outside the browser (other apps, dialogs, the desktop):
        - \(run("ui launch xterm"))   launch an installed desktop app by its desktop ID or executable. \
        Always use ui launch to start desktop apps so they stay open when the turn ends or is stopped. \
        It accepts an absolute executable or .desktop file path if the short name is unavailable.
        - \(run("ui tree"))   the focused window, one element per line with [ref=N]
        - \(run("ui click 12")), \(run("ui type 21 'hello' --enter")), \(run("ui key ctrl+l"))
        - \(run("ui screenshot"))   a small screenshot; \(look) to see it. Only when nothing else \
        shows what you need.
        On a follow-up, check the current window first and continue in the app if it is already open. \
        Leave open apps, windows, and sign-in prompts for the next turn unless the user asks you to close them.

        Example: to check something on a website, run web open https://example.com, then web \
        snapshot (or web read), then web click a link's ref and web snapshot again. Always call a \
        tool right away; do not only plan. Use the [Screen now: ...] note as a hint and verify the \
        current app with ui tree or the current webpage with web snapshot.

        Work step by step: after each action take a new ui tree or web snapshot, because refs change \
        with the window or page. If a command prints FAILED or an error, it did not happen: say so, or try \
        another way, and never tell the user it worked. Before saying something was sent, posted or \
        saved, check with web snapshot that it really was. Only report what you actually saw; never \
        guess what a page says. When done, \
        say briefly what you did and what you found. Files the user shares are in /mnt/sai.

        Never ask the user to type a password or username in the chat. When a sign-in form needs \
        credentials, click into its first field, then tell the user: "Click the key button in the \
        Computer panel to fill it from 1Password, then tell me to continue." and stop.
        """
    }
}

/// The command that runs one Codex turn inside the guest, over SSH. Codex reads the instructions
/// from AGENTS.md in its working directory, so the turn writes them first.
public struct CodexCommand: Equatable, Sendable {
    public var prompt: String
    public var resume: String?
    public var model: String
    public var provider: ModelProvider
    /// For OpenRouter, the Mac proxy's per-launch token (not the real key).
    public var apiKey: String
    public var guestPort: UInt16

    public init(prompt: String, resume: String?, model: String, provider: ModelProvider = .ollama,
                apiKey: String? = nil, guestPort: UInt16 = 11434) {
        self.prompt = prompt
        self.resume = resume
        self.model = model
        self.provider = provider
        self.apiKey = apiKey ?? provider.placeholderKey
        self.guestPort = guestPort
    }

    /// Ollama through Codex's built-in --oss provider; OpenRouter as a custom Responses-API
    /// provider pointed at the tunnel (the Mac's proxy adds the key, so the guest holds none).
    var providerOptions: [String] {
        switch provider {
        case .ollama:
            ["--oss", "--local-provider", "ollama"]
        case .openRouter:
            ["-c", quote("model_provider=\"saige\""),
             "-c", quote(#"model_providers.saige={name="OpenRouter via Agent S",base_url="http://127.0.0.1:\#(guestPort)/v1",env_key="AGENT_S_PROXY_KEY",wire_api="responses"}"#)]
        case .codexAccount:
            ["-c", quote("model_provider=\"openai\""), "-c", quote("forced_login_method=\"chatgpt\""),
             "-c", quote("cli_auth_credentials_store=\"file\"")]
        case .claudeAccount:
            []
        }
    }

    public var remoteShell: String {
        guard provider != .claudeAccount else { return "" }
        let usesAccount = provider == .codexAccount
        var options = ["--json"] + providerOptions
        if !usesAccount { options += ["-m", quote(model)] }
        options += [
            // Codex's web search goes to Ollama's cloud in --oss mode (needs an Ollama account) and
            // tempts the model to answer from search instead of the screen.
            "-c", quote("web_search=\"disabled\""),
            // The guest is the sandbox; Codex's own sandbox would also block the X display.
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        // Options go before `resume`: `codex exec resume` rejects them after it.
        var codex = ["codex", "exec"] + options
        if let resume { codex += ["resume", quote(resume)] }
        if usesAccount { codex.append("--") }
        codex.append(quote(prompt))
        let environment = usesAccount ? SandboxAccount.profileEnvironment(for: provider) + "DISPLAY=:0 "
            : "DISPLAY=:0 AGENT_S_PROXY_KEY=\(quote(apiKey)) "
        return "cd ~/work && printf '%s\\n' \(quote(AgentInstructions.text(for: .codex))) > AGENTS.md && "
            + environment + codex.joined(separator: " ") + " </dev/null"
    }

    public func quote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

/// Parses `codex exec --json` output (one event per line) into the chat's events.
public enum CodexStream {
    public static func parse(line: String) -> [ClaudeEvent] {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = object["type"] as? String
        else { return [] }

        switch type {
        case "thread.started":
            guard let id = object["thread_id"] as? String else { return [] }
            return [.started(sessionId: id, model: nil)]

        case "item.started", "item.completed":
            guard let item = object["item"] as? [String: Any],
                  let kind = item["type"] as? String else { return [] }
            let id = item["id"] as? String ?? ""
            switch (type, kind) {
            case ("item.started", "command_execution"):
                let command = item["command"] as? String
                return [.toolCall(id: id, name: "Shell", summary: summary(of: command ?? ""), detail: command)]
            case ("item.completed", "command_execution"):
                let exit = item["exit_code"] as? Int ?? 0
                return [.toolResult(id: id, output: ClaudeStream.clip(item["aggregated_output"] as? String ?? ""),
                                    isError: exit != 0)]
            case ("item.completed", "agent_message"):
                guard let text = item["text"] as? String,
                      !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
                return [.text(text)]
            default:
                // Reasoning, file edits and warnings (such as missing model metadata) stay out of the chat.
                return []
            }

        case "turn.completed":
            // Codex's input count is summed over the turn's requests (its cached share is part of it).
            var usage = object["usage"] as? [String: Any] ?? [:]
            usage["cache_read_input_tokens"] = nil
            return [ClaudeEvent.usage(from: usage), .finished(result: nil, isError: false, costUSD: nil)].compactMap { $0 }

        case "turn.failed", "error":
            let message = ((object["error"] as? [String: Any])?["message"] as? String) ?? object["message"] as? String
            return [.finished(result: message, isError: true, costUSD: nil)]

        default:
            return []
        }
    }

    /// `/bin/bash -lc 'echo hi'` reads as `echo hi`.
    static func summary(of command: String) -> String {
        var line = command
        for prefix in ["/bin/bash -lc ", "bash -lc ", "/bin/sh -c "] where line.hasPrefix(prefix) {
            line = String(line.dropFirst(prefix.count))
            if line.count >= 2, line.first == "'", line.last == "'" { line = String(line.dropFirst().dropLast()) }
        }
        return ClaudeStream.summarize(tool: "Shell", input: ["command": line])
    }
}
