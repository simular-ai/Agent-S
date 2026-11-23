import Foundation

/// The computer a cloud agent works on. Cloud agents never use the local VM.
public enum CloudComputer: String, CaseIterable, Identifiable, Codable, Sendable {
    /// A computer on the user's Sai account, driven by Sai's own agent through the Sai API.
    case sai
    /// The user's box (box.simular.ai), driven over box's MCP server by a harness that runs in
    /// the app with no local tools.
    case box

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .sai: "Sai computer"
        case .box: "Simular Box"
        }
    }

    /// The harnesses that can drive this computer.
    public var harnesses: [Harness] {
        switch self {
        case .sai: [.sai]
        case .box: [.claudeCode, .codex, .deepSeek]
        }
    }
}

public extension Harness {
    /// The harnesses available for the selected computer.
    static func allowed(in mode: PrivacyMode, computer: CloudComputer?) -> [Harness] {
        guard mode == .remote else { return allowed(in: mode) }
        return (computer ?? .box).harnesses
    }
}

/// box's MCP server (production) and how agents are told to use it.
public enum Box {
    public static let mcpURL = "https://mcp.box.simular.ai/mcp"
    /// Box tools a box agent may not use: they change the user's boxes or plan, not the work.
    public static let blockedTools = ["box_delete", "box_upgrade", "billing", "box_create", "box_restore_paths"]
        .map { "mcp__box__\($0)" }
    public static let accountToolRule = "mcp__box__*"

    /// Whether a tool list (from Claude Code's init event) is only box's tools.
    public static func onlyBoxTools(_ tools: [String]) -> Bool {
        tools.allSatisfy { $0.hasPrefix("mcp__box__") }
    }

    public static func onlyBoxAccountTools(_ tools: [String]) -> Bool {
        tools.allSatisfy { $0 == "EndConversation" || ($0.hasPrefix("mcp__box__") && !blockedTools.contains($0)) }
    }

    private static let workTools: Set<String> = [
        "shell", "read_file", "write_file", "edit_file", "list_dir", "desk", "gui_plan", "script_run",
        "computer_act", "computer_state", "screenshot", "box_status", "box_wake", "box_hibernate", "box_save",
    ]

    /// The app's Box tool bridge exposes work on the remote desktop and its files.
    public static func allowsAgentTool(_ name: String) -> Bool {
        if workTools.contains(name) { return true }
        guard name.hasPrefix("browser_"), name.count <= 64 else { return false }
        return name.utf8.allSatisfy { (97...122).contains($0) || (48...57).contains($0) || $0 == 95 }
    }

    public static func scopedAgentArguments(_ arguments: [String: Any], acceptsBox: Bool, box: String?) -> [String: Any]? {
        if let box, !box.isEmpty, !acceptsBox { return nil }
        var scoped = arguments
        scoped["box"] = acceptsBox ? box.flatMap { $0.isEmpty ? nil : $0 } : nil
        return scoped
    }
    public static let portal = URL(string: "https://box.simular.ai/box")!

    /// The MCP config Claude Code loads: box only, the key read from the environment (never
    /// written to disk or shown to the model).
    public static let mcpConfig = #"{"mcpServers":{"box":{"type":"http","url":"\#(mcpURL)","headers":{"Authorization":"Bearer ${BOX_TOKEN}"}}}}"#

    public static func instructions(boxName: String?) -> String {
        instructions(boxName: boxName, appRoutesTools: false)
    }

    public static func agentInstructions(boxName: String?) -> String {
        instructions(boxName: boxName, appRoutesTools: true)
    }

    private static func instructions(boxName: String?, appRoutesTools: Bool) -> String {
        let target: String
        if let boxName, !boxName.isEmpty {
            target = "the box named \"\(boxName)\"" + (appRoutesTools ? "" : " (pass it as the box argument)")
        } else {
            target = "the user's default Simular Box"
        }
        let toolSource = appRoutesTools ? "the Box tools provided by Agent S" : "the box tools (mcp__box__…)"
        return """
        You are Sai. You work on a cloud computer, a Simular Box: \(target), an Ubuntu desktop with Chrome and a shell, \
        reached only through \(toolSource). You have no other tools and no local files.

        Use the box tools for everything:
        - Websites: browser_navigate to open a page (it returns a snapshot with refs), browser_read for \
        a page's text, browser_click / browser_fill / browser_type with a ref, browser_fetch to read a \
        page without showing it.
        - Desktop apps: gui_plan with the whole plan in one call; computer_state or screenshot to look.
        - Commands and files: shell, read_file, write_file.
        - If a step needs the person (a sign-in, 2FA, a captcha, a payment), ask them to open the box's \
        live view and do it, then continue.

        Work step by step and check each result. If a tool fails, say so; never claim something worked \
        without seeing it. When done, say briefly what you did and what you found.
        """
    }
}

/// Claude Code run in the app for a box agent with the box MCP server as its only tool source.
public struct BoxClaudeCommand: Equatable, Sendable {
    public var prompt: String
    public var resume: String?
    public var model: String
    public var provider: ModelProvider
    public var apiKey: String
    public var boxName: String?
    /// The loopback endpoint when Agent S supplies the model provider.
    public var baseURL: String

    public init(prompt: String, resume: String?, model: String, provider: ModelProvider, apiKey: String,
                boxName: String?, baseURL: String) {
        self.prompt = prompt
        self.resume = resume
        self.model = model
        self.provider = provider
        self.apiKey = apiKey
        self.boxName = boxName
        self.baseURL = baseURL
    }

    public func arguments(mcpConfigPath: String) -> [String] {
        let usesAccount = provider == .claudeAccount
        var args = usesAccount ? ["-p"] : ["-p", prompt]
        args += [
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            // No shell, files or web tools on the Mac: only box's MCP tools.
            "--tools", "",
            "--strict-mcp-config", "--mcp-config", mcpConfigPath,
            "--setting-sources", "",
            "--disable-slash-commands",
            "--permission-mode", usesAccount ? "dontAsk" : "bypassPermissions",
            "--disallowedTools", Box.blockedTools.joined(separator: ","),
            "--system-prompt", Box.instructions(boxName: boxName),
        ]
        if usesAccount {
            args += ["--no-chrome", "--settings", ModelProvider.claudeAccountSettings,
                     "--allowedTools", Box.accountToolRule]
        }
        if let resume { args += ["--resume", resume] }
        if usesAccount { args += ["--", prompt] }
        return args
    }

    /// The full process environment, with Claude Code's login supplied by the chosen profile.
    public func environment(home: String, configDir: String, boxToken: String) -> [String: String] {
        var environment = [
            "HOME": home,
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "CLAUDE_CONFIG_DIR": configDir,
            "BOX_TOKEN": boxToken,
        ]
        environment.merge(provider.claudeEnvironment(baseURL: baseURL, apiKey: apiKey, model: model)) { _, value in value }
        if provider == .claudeAccount { environment["ENABLE_TOOL_SEARCH"] = "false" }
        return environment
    }
}
