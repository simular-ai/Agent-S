import Foundation
import AgentSCore

/// What an agent is: where it runs (private or cloud), its harness, and its model. Chosen when the
/// agent is created; each agent has its own conversation and harness context.
struct AgentConfig: Codable, Identifiable, Equatable {
    /// A private agent never uses a cloud provider or harness, whatever a saved file says.
    var enforcingPrivacy: AgentConfig {
        var config = self
        if privacy != .remote { config.cloudComputer = nil }
        let allowed = Harness.allowed(in: privacy, computer: config.computer)
        if !allowed.contains(config.harness) { config.harness = allowed[0] }
        if let provider = ModelProvider.fitting(config.provider, in: privacy, harness: config.harness, computer: config.computer) {
            config.provider = provider
        }
        return config
    }

    /// The cloud computer a cloud agent works on (older cloud agents: Sai if they used Sai).
    var computer: CloudComputer? {
        guard privacy == .remote else { return nil }
        return cloudComputer ?? (harness == .sai ? .sai : .box)
    }

    /// Whether the agent works on the local VM (private agents only).
    var usesLocalComputer: Bool { privacy == .fullyPrivate }

    var id = UUID()
    var name: String
    var privacy: PrivacyMode
    var harness: Harness
    var provider: ModelProvider
    var ollamaModel: String
    var openRouterModel: String
    var saiMachine: String = ""
    var saiModel: String = "auto"
    /// The headshot: a picture the user chose (a file in the data folder's `avatars`), or else
    /// the first letter of its name on one of the palette's colors. Optional so older saved agents still load.
    var avatarImage: String?
    var avatarColor: Int?
    /// Cloud agents: a Sai computer or box, and which box (empty: the default one).
    var cloudComputer: CloudComputer?
    var boxName: String?
}

/// An agent as saved on disk: its definition, its chat, and its harness sessions (so it resumes
/// with its context after a restart).
struct SavedAgent: Codable {
    var config: AgentConfig
    var items: [ChatItem]
    var sessions: [String: String]
}

/// One item in the chat: what the user asked, what the agent said, a step it took.
struct ChatItem: Identifiable, Equatable, Codable {
    enum Kind: Equatable, Codable {
        case user
        case agent
        case step(name: String, done: Bool, failed: Bool, output: String)
        case notice(isError: Bool)
        /// Sai paused for the user: a yes/no card, or a link to open (sign-ins, phone checks).
        case approval(id: String, linkOnly: Bool, url: URL?, answer: String?)
    }
    let id: String
    var kind: Kind
    var text: String
}

/// One agent: its definition, its conversation, and its harness context. A turn is one SSH process
/// (or one Sai API call), continued with the harness's resume so the agent keeps its context.
/// Each harness keeps its own session; switching harness mid-chat starts a fresh one for it.
@MainActor
final class AgentSession: ObservableObject, Identifiable {
    @Published var config: AgentConfig { didSet { onChange?() } }
    @Published private(set) var items: [ChatItem] = []
    @Published private(set) var isRunning = false
    @Published private(set) var sessions: [Harness: String] = [:]
    private var accountSessions: [ModelProvider: String] = [:]
    private var boxSessions: [String: String] = [:]
    /// Called when something worth saving changed (the definition, or a finished turn).
    var onChange: (() -> Void)?

    nonisolated var id: UUID { MainActor.assumeIsolated { config.id } }

    init(config: AgentConfig, items: [ChatItem] = [], sessions: [Harness: String] = [:]) {
        self.config = config
        self.items = items
        self.sessions = sessions
    }

    convenience init(saved: SavedAgent) {
        let sessions = Dictionary(uniqueKeysWithValues: saved.sessions.compactMap { key, value in
            Harness(rawValue: key).map { ($0, value) }
        })
        // A turn that was running when the app quit isn't running now.
        self.init(config: saved.config.enforcingPrivacy, items: saved.items, sessions: sessions)
        for provider in ModelProvider.allCases where provider.usesAccount {
            accountSessions[provider] = saved.sessions[provider.rawValue]
        }
        boxSessions = saved.sessions.filter { $0.key.hasPrefix("box/") }
    }

    var saved: SavedAgent {
        var savedSessions = Dictionary(uniqueKeysWithValues: sessions.map { ($0.key.rawValue, $0.value) })
        for (provider, id) in accountSessions { savedSessions[provider.rawValue] = id }
        for (key, id) in boxSessions { savedSessions[key] = id }
        return SavedAgent(config: config, items: items, sessions: savedSessions)
    }

    var currentSessionID: String? {
        if let key = boxSessionKey { return boxSessions[key] }
        if config.harness != .sai, config.provider.usesAccount { return accountSessions[config.provider] }
        return sessions[config.harness]
    }

    private var boxSessionKey: String? {
        guard config.computer == .box, config.harness == .codex || config.harness == .deepSeek else { return nil }
        return "box/\(config.harness.rawValue)/\(config.provider.rawValue)/\(config.boxName ?? "")"
    }

    /// The last thing said, for the sidebar.
    var lastLine: String? {
        items.last(where: { $0.kind == .agent || $0.kind == .user })?.text.replacingOccurrences(of: "\n", with: " ")
    }
    /// Tokens the last turn used, and which harness ran it.
    @Published private(set) var lastUsage: (input: Int, output: Int)?
    @Published private(set) var harness: Harness = .claudeCode
    /// Step ids from the harness are only unique within a turn (Codex restarts at item_0), so the
    /// chat prefixes them with the turn.
    private var turn = UUID().uuidString
    private var process: Process?
    private var activeAccountProvider: ModelProvider?

    /// `screen` describes what is on the computer's screen right now; it is put in front of the
    /// message (not shown in the chat) so the agent never answers from an out-of-date picture.
    func send(_ prompt: String, harness: Harness, shell: GuestShell, ip: String, model: String, ollamaURL: String,
              provider: ModelProvider, apiKey: String?, tunnelTo: UInt16, screen: @escaping () async -> String?) {
        guard !isRunning else { return }
        items.append(ChatItem(id: UUID().uuidString, kind: .user, text: prompt))
        isRunning = true
        self.harness = harness
        activeAccountProvider = nil
        turn = UUID().uuidString
        Task { @MainActor in
            let now = await screen()
            let message = now.map { "[\($0)]\n\n\(prompt)" } ?? prompt
            start(message, harness: harness, shell: shell, ip: ip, model: model, ollamaURL: ollamaURL,
                  provider: provider, apiKey: apiKey, tunnelTo: tunnelTo)
        }
    }

    private func start(_ prompt: String, harness: Harness, shell: GuestShell, ip: String, model: String, ollamaURL: String,
                       provider: ModelProvider, apiKey: String?, tunnelTo: UInt16) {
        guard isRunning else { return } // stopped while the screen was being read
        let command = harness.remoteShell(prompt: prompt, resume: sessions[harness], model: model, ollamaURL: ollamaURL,
                                          provider: provider, apiKey: apiKey)
        do {
            process = try shell.stream(command, on: ip, tunnelTo: tunnelTo, line: { [weak self] line in
                let events = harness.parse(line: line)
                guard !events.isEmpty else { return }
                Task { @MainActor in self?.apply(events) }
            }, finished: { [weak self] status, stderr in
                Task { @MainActor in self?.finish(status: status, stderr: stderr) }
            })
        } catch {
            notice("Couldn't reach the computer: \(error.localizedDescription)", isError: true)
            isRunning = false
        }
    }

    func stop() {
        if let process, process.isRunning { process.terminate() }
        if let appProcess, appProcess.isRunning { appProcess.terminate() }
        closeClaudeMacTool()
        hostedTask?.cancel()
        stopHostedTurn?()
        denyPendingApprovals()
        if let saiTask {
            saiTask.cancel()
            if let session = sessions[.sai] { Task { await SaiAPI.abort(session: session) } }
        }
    }

    // MARK: Box agents (Claude Code in the app, box tools only)

    /// Runs one turn of Claude Code on the Mac for a box agent: no built-in tools, box's MCP only,
    /// its own config folder; the box key passes through the environment only.
    func sendOnBox(_ prompt: String, command: BoxClaudeCommand, claude: URL, boxToken: String, dataFolder: URL) {
        guard !isRunning else { return }
        var command = command
        let usesAccount = command.provider == .claudeAccount
        command.resume = usesAccount ? accountSessions[.claudeAccount] : sessions[.claudeCode]
        let folder = dataFolder.appendingPathComponent("box-agents/\(config.id.uuidString)", isDirectory: true)
        let mcp = folder.appendingPathComponent("mcp.json")
        runClaudeInApp(prompt, claude: claude, folder: folder, provider: command.provider,
                       accountProfile: usesAccount ? ClaudeAccount.profile(in: dataFolder) : nil, prepare: { configDir in
            try Data(Box.mcpConfig.utf8).write(to: mcp, options: .atomic)
            return (command.arguments(mcpConfigPath: mcp.path),
                    command.environment(home: FileManager.default.homeDirectoryForCurrentUser.path,
                                        configDir: configDir.path, boxToken: boxToken))
        }, toolsAllowed: { tools in
            usesAccount ? Box.onlyBoxAccountTools(tools) : Box.onlyBoxTools(tools)
        }, toolsRule: "box agents may only use box's tools")
    }

    // MARK: This Mac agents (Claude Code in the app, every tool use approved first)

    /// Account turns use the app's Mac tool. Proxy turns keep Claude's existing approval hook.
    func sendOnMac(_ prompt: String, command: MacClaudeCommand, claude: URL, dataFolder: URL, settingsPath: String,
                   path: String, approvalPort: UInt16, approvalToken: String) {
        guard !isRunning else { return }
        var command = command
        let usesAccount = command.provider == .claudeAccount
        command.resume = usesAccount ? accountSessions[.claudeAccount] : sessions[.claudeCode]
        allowRestOfTurn = false
        let folder = dataFolder.appendingPathComponent("mac-agents/\(config.id.uuidString)", isDirectory: true)
        let mcpConfig = usesAccount ? folder.appendingPathComponent("mac-tools.json") : nil
        let allowedTools = usesAccount ? [MacAgent.mcpCommandTool, "EndConversation"] : MacAgent.tools
        runClaudeInApp(prompt, claude: claude, folder: folder, provider: command.provider,
                       accountProfile: usesAccount ? ClaudeAccount.profile(in: dataFolder) : nil,
                       workingDirectory: usesAccount ? nil : FileManager.default.homeDirectoryForCurrentUser,
                       prepare: { configDir in
            var environment = command.environment(home: FileManager.default.homeDirectoryForCurrentUser.path,
                                                  configDir: configDir.path, path: path,
                                                  approvalPort: approvalPort, approvalToken: approvalToken)
            environment["AGENT_S_AGENT"] = self.config.id.uuidString
            if let mcpConfig {
                try ApprovalServer.claudeToolConfig(port: approvalPort).write(to: mcpConfig, options: .atomic)
                self.claudeMacTool = HarnessMacCommand()
                environment["AGENT_S_TURN"] = self.turn
            }
            return (command.arguments(settingsPath: settingsPath, mcpConfigPath: mcpConfig?.path), environment)
        }, toolsAllowed: { tools in tools.allSatisfy { allowedTools.contains($0) } },
           toolsRule: "This Mac agents may only use Agent S's approved tools")
    }

    private var claudeMacTool: HarnessMacCommand?
    private var claudeMacToolBusy = false

    func runClaudeMacCommand(_ command: String, turn requestedTurn: String) async -> ApprovalServer.CommandResult {
        guard turn == requestedTurn, isRunning, let runner = claudeMacTool else {
            return ("This turn has ended.", false, nil)
        }
        guard !claudeMacToolBusy else { return ("Wait for the previous Mac command to finish.", false, nil) }
        claudeMacToolBusy = true
        defer { if claudeMacTool === runner { claudeMacToolBusy = false } }
        guard await requestApproval(tool: "Bash", input: ["command": command], macToolPath: MacAccess.macTool),
              claudeMacTool === runner else {
            return ("The command was not approved.", false, nil)
        }
        do { return try await runner.run(command) }
        catch { return (error.localizedDescription, false, nil) }
    }

    private func closeClaudeMacTool() {
        claudeMacTool?.stop()
        claudeMacTool = nil
        claudeMacToolBusy = false
    }

    /// Steps waiting for the person, by approval id.
    private var pendingApprovals: [String: CheckedContinuation<Bool, Never>] = [:]
    var pendingMacApprovals: [ChatItem] { pendingMacApprovals(in: items) }

    /// The publisher emits its new items before the stored `items` value changes.
    func pendingMacApprovals(in transcript: [ChatItem]) -> [ChatItem] {
        guard !pendingApprovals.isEmpty else { return [] }
        return transcript.filter { item in
            guard case let .approval(id, _, _, nil) = item.kind else { return false }
            return pendingApprovals[id] != nil
        }
    }

    /// "Allow for this task": the rest of this turn runs without asking.
    private var allowRestOfTurn = false

    /// Asked by the hook before a tool runs on the Mac: shows the step and waits for Allow or Deny.
    /// Looking (screenshots, the window's tree, the app list) runs without asking.
    func requestApproval(tool: String, input: [String: Any], macToolPath: String) async -> Bool {
        guard isRunning else { return false }
        if allowRestOfTurn || MacAgent.isLookOnly(tool: tool, input: input, macToolPath: macToolPath) { return true }
        let id = UUID().uuidString
        return await withCheckedContinuation { continuation in
            pendingApprovals[id] = continuation
            items.append(ChatItem(id: "approval/\(id)", kind: .approval(id: id, linkOnly: false, url: nil, answer: nil),
                                  text: Self.describe(tool: tool, input: input, macToolPath: macToolPath)))
        }
    }

    /// What a step will do, for the approval card: the exact command or file.
    static func describe(tool: String, input: [String: Any], macToolPath: String) -> String {
        switch tool {
        case "Bash":
            let command = (input["command"] as? String ?? "").replacingOccurrences(of: macToolPath + " ", with: "mac ")
            return "Run on this Mac:\n\(command)"
        case "Write": return "Write the file \(input["file_path"] as? String ?? "?")"
        case "Edit": return "Change the file \(input["file_path"] as? String ?? "?")"
        case "Read": return "Read the file \(input["file_path"] as? String ?? "?")"
        default: return "Use \(tool)"
        }
    }

    /// Ends every step still waiting (the turn ended or was stopped): they don't run.
    private func denyPendingApprovals() {
        for (id, continuation) in pendingApprovals {
            continuation.resume(returning: false)
            if let index = items.firstIndex(where: { $0.id == "approval/\(id)" }),
               case let .approval(_, linkOnly, url, nil) = items[index].kind {
                items[index].kind = .approval(id: id, linkOnly: linkOnly, url: url, answer: "deny")
            }
        }
        pendingApprovals = [:]
    }

    // MARK: Claude Code in the app

    private var appProcess: Process?

    /// One turn of Claude Code run by the app (box and This Mac agents): its own config folder,
    /// the whole environment given (nothing inherited), and a stop if it starts with tools it
    /// shouldn't have.
    private func runClaudeInApp(_ prompt: String, claude: URL, folder: URL, provider: ModelProvider,
                                accountProfile: URL?, workingDirectory: URL? = nil,
                                prepare: (URL) throws -> ([String], [String: String]),
                                toolsAllowed: @escaping @Sendable ([String]) -> Bool, toolsRule: String) {
        items.append(ChatItem(id: UUID().uuidString, kind: .user, text: prompt))
        isRunning = true
        harness = .claudeCode
        activeAccountProvider = provider.usesAccount ? provider : nil
        turn = UUID().uuidString
        let turnID = turn
        let configDir = accountProfile ?? folder.appendingPathComponent("claude", isDirectory: true)
        let work = folder.appendingPathComponent("work", isDirectory: true)
        do {
            for dir in [configDir, work] { try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true) }
            let (arguments, environment) = try prepare(configDir)
            let process = Process()
            process.executableURL = claude
            process.arguments = arguments
            process.environment = environment
            process.currentDirectoryURL = workingDirectory ?? work
            process.standardInput = FileHandle.nullDevice
            let out = Pipe(), err = Pipe()
            process.standardOutput = out
            process.standardError = err
            let reader = LineReader { [weak self, weak process] line in
                // Defense in depth: if Claude Code ever starts with a tool it shouldn't have, stop
                // before the model can use it.
                if line.contains("\"subtype\":\"init\""),
                   let data = line.data(using: .utf8),
                   let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let tools = object["tools"] as? [String], !toolsAllowed(tools) {
                    process?.terminate()
                    Task { @MainActor in
                        guard self?.turn == turnID else { return }
                        self?.notice("Stopped: Claude Code started with other tools (\(tools.joined(separator: ", "))). \(toolsRule).", isError: true)
                    }
                    return
                }
                let events = ClaudeStream.parse(line: line)
                guard !events.isEmpty else { return }
                Task { @MainActor in
                    if self?.turn == turnID { self?.apply(events) }
                }
            }
            out.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty { handle.readabilityHandler = nil } else { reader.feed(data) }
            }
            process.terminationHandler = { [weak self] process in
                out.fileHandleForReading.readabilityHandler = nil
                reader.feed(out.fileHandleForReading.readDataToEndOfFile())
                reader.flush()
                let stderr = String(decoding: err.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                    .split(separator: "\n").filter { !$0.contains("unrecognized_model") && !$0.contains("no stdin data") }
                    .joined(separator: "\n")
                let status = process.terminationStatus
                Task { @MainActor in
                    self?.appProcess = nil
                    self?.closeClaudeMacTool()
                    self?.denyPendingApprovals()
                    if self?.turn == turnID { self?.finish(status: status, stderr: stderr) }
                    else { self?.isRunning = false }
                }
            }
            try process.run()
            appProcess = process
        } catch {
            closeClaudeMacTool()
            notice("Couldn't start Claude Code: \(error.localizedDescription)", isError: true)
            isRunning = false
        }
    }

    // MARK: App-managed harness turns

    private var stopHostedTurn: (() -> Void)?
    private var hostedTask: Task<Void, Never>?

    func sendWithCodex(_ prompt: String, dataFolder: URL, model: String, baseURL: String = "", apiKey: String = "") {
        let runner = CodexAccountTurn()
        let config = config
        let usesBox = config.computer == .box
        runHostedTurn(prompt, harness: .codex, stop: runner.stop) { [weak self] resume, onEvent, onTextDelta in
            if usesBox {
                try await runner.runOnBox(prompt: prompt, resume: resume, dataFolder: dataFolder, agentID: config.id,
                                          model: model, provider: config.provider, baseURL: baseURL, apiKey: apiKey,
                                          boxName: config.boxName, onEvent: onEvent, onTextDelta: onTextDelta)
            } else {
                try await runner.run(prompt: prompt, resume: resume, dataFolder: dataFolder, agentID: config.id, approve: { command in
                    guard let self else { return false }
                    return await self.requestApproval(tool: "Bash", input: ["command": command], macToolPath: MacAccess.macTool)
                }, onEvent: onEvent, onTextDelta: onTextDelta)
            }
        }
    }

    func sendWithDeepSeekOnBox(_ prompt: String, dataFolder: URL, model: String, baseURL: String, apiKey: String) {
        let runner = DeepSeekBoxTurn()
        let config = config
        runHostedTurn(prompt, harness: .deepSeek, stop: runner.stop) { resume, onEvent, _ in
            try await runner.run(prompt: prompt, resume: resume, model: model, baseURL: baseURL, apiKey: apiKey,
                                 dataFolder: dataFolder, agentID: config.id, boxName: config.boxName, onEvent: onEvent)
        }
    }

    private func runHostedTurn(_ prompt: String, harness: Harness, stop: @escaping () -> Void,
                               run: @escaping (String?, @escaping (ClaudeEvent) -> Void,
                                               @escaping (String, String) -> Void) async throws -> Void) {
        guard !isRunning else { return }
        let resume = currentSessionID
        let context = boxSessionKey
        stopHostedTurn = stop
        items.append(ChatItem(id: UUID().uuidString, kind: .user, text: prompt))
        isRunning = true
        self.harness = harness
        activeAccountProvider = context == nil && config.provider.usesAccount ? config.provider : nil
        allowRestOfTurn = false
        turn = UUID().uuidString
        let turnID = turn
        hostedTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                self.stopHostedTurn = nil
                self.hostedTask = nil
                self.denyPendingApprovals()
                self.isRunning = false
                self.onChange?()
            }
            do {
                try await run(resume, { event in
                    guard self.turn == turnID else { return }
                    if let context, case let .started(id, _) = event { self.boxSessions[context] = id }
                    else { self.apply([event]) }
                }, { id, delta in
                    guard self.turn == turnID, !delta.isEmpty else { return }
                    let textID = "\(turnID)/text/\(id)"
                    if let index = self.items.lastIndex(where: { $0.id == textID }) {
                        self.items[index].text += delta
                    } else {
                        self.items.append(ChatItem(id: textID, kind: .agent, text: delta))
                    }
                })
            } catch {
                guard self.turn == turnID else { return }
                if Task.isCancelled { self.notice("Stopped.", isError: false) }
                else { self.notice("\(harness.displayName) stopped: \(error.localizedDescription)", isError: true) }
            }
        }
    }

    // MARK: Sai harness (Sai API, from the Mac)

    private var saiTask: Task<Void, Never>?
    private var saiMachine: String?
    /// The text item each streamed text id is being written into.
    private var saiText: [String: String] = [:]

    /// Sends one message to Sai on `machine`. Follow-ups continue the same Sai session while the
    /// machine stays the same.
    func sendToSai(_ prompt: String, machine: String, model: String?) {
        guard !isRunning else { return }
        items.append(ChatItem(id: UUID().uuidString, kind: .user, text: prompt))
        isRunning = true
        harness = .sai
        activeAccountProvider = nil
        turn = UUID().uuidString
        if saiMachine != machine { sessions[.sai] = nil; saiMachine = machine }
        let session = sessions[.sai]
        saiTask = Task { @MainActor [weak self] in
            do {
                try await SaiAPI.run(message: prompt, machine: machine, session: session, model: model) { event in
                    self?.applySai(event)
                }
            } catch is CancellationError {
                self?.notice("Stopped.", isError: false)
            } catch {
                if !Task.isCancelled { self?.notice(error.localizedDescription, isError: true) }
            }
            self?.isRunning = false
            self?.saiTask = nil
            self?.onChange?()
        }
    }

    /// Answers a yes/no approval card.
    func answer(approval id: String, with response: String) {
        guard let index = items.firstIndex(where: { $0.id == "approval/\(id)" }),
              case let .approval(_, linkOnly, url, nil) = items[index].kind else { return }
        // A step on this Mac, waiting in the hook.
        if let continuation = pendingApprovals.removeValue(forKey: id) {
            items[index].kind = .approval(id: id, linkOnly: linkOnly, url: url, answer: response)
            if response == "approve_for_task" { allowRestOfTurn = true }
            continuation.resume(returning: response != "deny")
            return
        }
        guard config.harness == .sai else { return }
        items[index].kind = .approval(id: id, linkOnly: linkOnly, url: url, answer: response)
        Task { @MainActor in
            do { try await SaiAPI.approve(id, response: response) } catch {
                notice(error.localizedDescription, isError: true)
            }
        }
    }

    private func applySai(_ event: [String: Any]) {
        let data = event["data"] as? [String: Any] ?? [:]
        switch event["type"] as? String {
        case "data-session":
            if let id = data["sessionId"] as? String { sessions[.sai] = id }
        case "text-delta":
            let textId = "\(turn)/text/\(event["id"] as? String ?? "")"
            let delta = event["delta"] as? String ?? ""
            if let index = items.lastIndex(where: { $0.id == textId }) {
                items[index].text += delta
            } else if !delta.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                items.append(ChatItem(id: textId, kind: .agent, text: delta))
            }
        case "tool-input-start", "tool-input-available":
            guard let callId = event["toolCallId"] as? String else { return }
            let id = "\(turn)/\(callId)"
            let title = (event["input"] as? [String: Any])?["title"] as? String
            if let index = items.lastIndex(where: { $0.id == id }) {
                if let title { items[index].text = title }
            } else {
                let name = event["toolName"] as? String ?? "tool"
                items.append(ChatItem(id: id, kind: .step(name: name, done: false, failed: false, output: ""),
                                      text: title ?? name))
            }
        case "data-progress":
            // Progress lines belong to the latest running step.
            guard let text = data["text"] as? String,
                  let index = items.lastIndex(where: { if case .step(_, false, _, _) = $0.kind { return true }; return false }),
                  case let .step(name, _, failed, output) = items[index].kind else { return }
            items[index].kind = .step(name: name, done: false, failed: failed, output: output.isEmpty ? text : output + "\n" + text)
        case "tool-output-error":
            guard let callId = event["toolCallId"] as? String,
                  let index = items.lastIndex(where: { $0.id == "\(turn)/\(callId)" }),
                  case let .step(name, _, _, output) = items[index].kind else { return }
            let error = event["errorText"] as? String ?? "failed"
            items[index].kind = .step(name: name, done: true, failed: true, output: output.isEmpty ? error : output + "\n" + error)
        case "data-approval-request":
            guard let id = data["approvalId"] as? String else { return }
            let title = [data["title"] as? String, data["description"] as? String].compactMap { $0 }.joined(separator: ": ")
            items.append(ChatItem(id: "approval/\(id)",
                                  kind: .approval(id: id, linkOnly: data["isLinkOnly"] as? Bool ?? false,
                                                  url: SaiAPI.approvalURL(data), answer: nil),
                                  text: title.isEmpty ? "Sai is asking for approval." : title))
        case "data-turn-result":
            if let usage = data["usage"] as? [String: Any] {
                let input = (usage["inputTokens"] as? Int) ?? 0, output = (usage["outputTokens"] as? Int) ?? 0
                if input + output > 0 { lastUsage = (input, output) }
            }
        case "finish":
            // Steps still open when the turn ends finished without an error.
            for index in items.indices {
                if case let .step(name, false, failed, output) = items[index].kind {
                    items[index].kind = .step(name: name, done: true, failed: failed, output: output)
                }
            }
        case "error":
            notice(event["errorText"] as? String ?? "Sai stopped with an error.", isError: true)
        default:
            break
        }
    }

    /// Forgets the conversation: the next message starts a new session.
    /// A fresh context: forgets the chat and the harness sessions; the next message starts new.
    func reset() {
        stop()
        if hostedTask != nil || (isRunning && activeAccountProvider != nil) { turn = UUID().uuidString }
        items = []
        sessions = [:]
        accountSessions = [:]
        boxSessions = [:]
        lastUsage = nil
        onChange?()
    }

    private func apply(_ events: [ClaudeEvent]) {
        for event in events {
            switch event {
            case let .started(id, _):
                if let activeAccountProvider { accountSessions[activeAccountProvider] = id }
                else { sessions[harness] = id }
            case let .text(text):
                items.append(ChatItem(id: UUID().uuidString, kind: .agent, text: text))
            case let .toolCall(id, name, summary):
                items.append(ChatItem(id: id.isEmpty ? UUID().uuidString : "\(turn)/\(id)",
                                      kind: .step(name: name, done: false, failed: false, output: ""),
                                      text: summary))
            case let .toolResult(id, output, isError):
                if let index = items.lastIndex(where: { $0.id == "\(turn)/\(id)" }),
                   case let .step(name, _, _, _) = items[index].kind {
                    items[index].kind = .step(name: name, done: true, failed: isError, output: output)
                }
            case let .usage(input, output):
                lastUsage = (input, output)
            case let .finished(result, isError, _):
                if isError { notice(result.map { "The agent stopped: \($0)" } ?? "The agent stopped with an error.", isError: true) }
            }
        }
    }

    private func finish(status: Int32, stderr: String) {
        process = nil
        isRunning = false
        defer { onChange?() }
        // 15: stopped by the user (SIGTERM).
        if status == 15 { notice("Stopped.", isError: false); return }
        if status != 0 {
            let reason = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            notice(reason.isEmpty ? "The agent exited with status \(status)." : String(reason.prefix(400)), isError: true)
        }
    }

    /// An error from the app about this agent, shown in its chat.
    func noticeFromApp(_ text: String) { notice(text, isError: true) }

    private func notice(_ text: String, isError: Bool) {
        items.append(ChatItem(id: UUID().uuidString, kind: .notice(isError: isError), text: text))
    }
}
