import Foundation
import Network
import os
import AgentSCore

/// Where DeepSeek Harness comes from. Release builds bundle a signed Node and a pinned, pruned dsh
/// (scripts/vendor-deepseek.sh). Development builds can point `AGENT_S_DSH_RUNTIME` at the vendored
/// folder, or fall back to a `dsh` the developer installed.
struct DeepSeekInstallation: Equatable {
    /// Node when `entry` is set; otherwise a `dsh` launcher.
    let executable: URL
    /// dsh's CLI entry, run by `executable`.
    let entry: URL?

    static var current: DeepSeekInstallation? { bundled ?? developmentRuntime ?? userInstalled }

    static var bundled: DeepSeekInstallation? {
        let contents = Bundle.main.bundleURL.appendingPathComponent("Contents", isDirectory: true)
        return runtime(node: contents.appendingPathComponent("Helpers/node"),
                       root: contents.appendingPathComponent("Resources/DeepSeek", isDirectory: true))
    }

    private static var developmentRuntime: DeepSeekInstallation? {
        guard let path = ProcessInfo.processInfo.environment["AGENT_S_DSH_RUNTIME"], !path.isEmpty else { return nil }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        return runtime(node: root.appendingPathComponent("bin/node"), root: root)
    }

    private static func runtime(node: URL, root: URL) -> DeepSeekInstallation? {
        let entry = root.appendingPathComponent("node_modules/@deepseek-ai/dsh/lib/bin.js")
        guard FileManager.default.isExecutableFile(atPath: node.path),
              FileManager.default.fileExists(atPath: entry.path) else { return nil }
        return DeepSeekInstallation(executable: node, entry: entry)
    }

    private static var userInstalled: DeepSeekInstallation? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let versions = home.appendingPathComponent(".nvm/versions/node", isDirectory: true)
        let nvm = (try? FileManager.default.contentsOfDirectory(atPath: versions.path)) ?? []
        let paths = [home.appendingPathComponent(".local/bin/dsh").path,
                     "/opt/homebrew/bin/dsh", "/usr/local/bin/dsh",
                     home.appendingPathComponent(".npm-global/bin/dsh").path]
            + nvm.sorted(by: >).map { versions.appendingPathComponent("\($0)/bin/dsh").path }
        return paths.first(where: FileManager.default.isExecutableFile(atPath:))
            .map { DeepSeekInstallation(executable: URL(fileURLWithPath: $0), entry: nil) }
    }

    /// A bundled runtime is pinned at build time; an installed `dsh` is checked before use.
    var isPinned: Bool { entry != nil }

    var path: String {
        executable.deletingLastPathComponent().path + ":/usr/bin:/bin" + (isPinned ? "" : ":/opt/homebrew/bin:/usr/local/bin")
    }
}

/// One message to a DeepSeek Harness agent. The agent's runtime outlives the message: follow-ups
/// reuse the same process and session, and Stop ends the process (the SDK has no cancel).
@MainActor
final class DeepSeekTurn {
    enum Target {
        case box(String?)
        case mac(approve: (String) async -> Bool)
    }

    static func instructions(boxName: String?) -> String {
        Box.agentInstructions(boxName: boxName) + "\nDeepSeek lists these tools with the mcp__box__ prefix."
    }

    static var macInstructions: String { MacAgent.instructions(macToolPath: MacAccess.macTool, viaMCP: true) }

    static var isAvailable: Bool { DeepSeekInstallation.current != nil }

    private var runtime: DeepSeekRuntime?
    private var stopped = false

    func run(prompt: String, resume: String?, model: String, baseURL: String, apiKey: String,
             dataFolder: URL, agentID: UUID, target: Target,
             onEvent: @escaping (ClaudeEvent) -> Void) async throws {
        guard let installation = DeepSeekInstallation.current else {
            throw DeepSeekFailure("DeepSeek Harness isn't included in this build of Agent S.")
        }
        guard var endpoint = URLComponents(string: baseURL), endpoint.scheme == "http",
              endpoint.host == "127.0.0.1", endpoint.port != nil, endpoint.user == nil, endpoint.password == nil,
              endpoint.query == nil, endpoint.fragment == nil, ["", "/", "/v1", "/v1/"].contains(endpoint.path) else {
            throw DeepSeekFailure("DeepSeek Harness needs the Agent S model endpoint.")
        }
        endpoint.path = "/v1"
        guard let modelURL = endpoint.url else { throw DeepSeekFailure("The model endpoint is invalid.") }
        let provider: ModelProvider = endpoint.port == Int(Ollama.port) ? .ollama : .openRouter

        let configuration: DeepSeekRuntime.Configuration
        switch target {
        case let .box(boxName):
            configuration = .init(target: .box(boxName), installation: installation, model: model,
                                  modelURL: modelURL.absoluteString, apiKey: apiKey,
                                  contextTokens: provider.contextTokens, outputTokens: provider.maxOutputTokens,
                                  instructions: Self.instructions(boxName: boxName))
        case .mac:
            configuration = .init(target: .mac, installation: installation, model: model,
                                  modelURL: modelURL.absoluteString, apiKey: apiKey,
                                  contextTokens: provider.contextTokens, outputTokens: provider.maxOutputTokens,
                                  instructions: Self.macInstructions)
        }
        try checkRunning()
        let runtime = try await DeepSeekRuntime.acquire(agentID: agentID, configuration: configuration, dataFolder: dataFolder)
        self.runtime = runtime
        try checkRunning()
        let approve: ((String) async -> Bool)? = if case let .mac(approve) = target { approve } else { nil }
        try await withTaskCancellationHandler {
            try await runtime.run(prompt: prompt, resume: resume, approveMac: approve, onEvent: onEvent)
        } onCancel: {
            Task { @MainActor in runtime.stop() }
        }
    }

    /// Sends more input to the running turn. The runtime reads it before the turn ends.
    func steer(_ text: String) async throws {
        guard let runtime, !stopped else { throw SteeringDeliveryError.notDelivered }
        try await runtime.steer(text)
    }

    func stop() {
        stopped = true
        runtime?.stop()
    }

    /// Confirms the runtime is no longer writing to its session.
    func waitUntilStopped() async -> Bool {
        guard let runtime else { return true }
        return await runtime.waitUntilExited()
    }

    private func checkRunning() throws {
        try Task.checkCancellation()
        if stopped { throw CancellationError() }
    }
}

struct DeepSeekFailure: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

/// One long-lived DeepSeek Harness SDK process for one agent, with its turn-scoped tool bridge.
@MainActor
final class DeepSeekRuntime {
    enum Computer: Equatable {
        case mac
        case box(String?)
    }

    struct Configuration: Equatable {
        var target: Computer
        var installation: DeepSeekInstallation
        var model: String
        var modelURL: String
        var apiKey: String
        var contextTokens: Int
        var outputTokens: Int
        var instructions: String
    }

    /// A runtime with nothing to do for this long exits; the next message starts it again.
    static let idleLifetime: Duration = .seconds(15 * 60)
    private static var live: [UUID: DeepSeekRuntime] = [:]
    private static let log = Logger(subsystem: "ai.simular.agent-s", category: "DeepSeek")

    /// Returns the agent's runtime for `configuration`, starting one when none matches.
    static func acquire(agentID: UUID, configuration: Configuration, dataFolder: URL) async throws -> DeepSeekRuntime {
        if let existing = live[agentID] {
            if existing.configuration == configuration, !existing.exited {
                existing.idleShutdown?.cancel()
                return existing
            }
            // A session is owned by one process at a time: let the old one release it first.
            existing.stop()
            _ = await existing.waitUntilExited()
            if live[agentID] === existing { live[agentID] = nil }
        }
        let runtime = try DeepSeekRuntime(agentID: agentID, configuration: configuration, dataFolder: dataFolder)
        live[agentID] = runtime
        do {
            try await runtime.start()
        } catch {
            runtime.stop()
            if live[agentID] === runtime { live[agentID] = nil }
            throw error
        }
        return runtime
    }

    /// Stops every runtime, when Agent S quits.
    static func stopAll() {
        for runtime in live.values { runtime.stop() }
        live = [:]
    }

    let agentID: UUID
    let configuration: Configuration
    private let home: URL
    private let work: URL
    private let sessions: URL
    private var connection: DeepSeekRPCConnection?
    private var bridge: DeepSeekToolBridge?
    private var loadedSessions: Set<String> = []
    private var turn: ActiveTurn?
    private var idleShutdown: Task<Void, Never>?
    private var stopping = false
    private(set) var exited = false
    private var initialized = false

    @MainActor
    private final class ActiveTurn {
        var tracker: DeepSeekSDKTurnTracker
        var projection: DeepSeekSDKProjection
        let onEvent: (ClaudeEvent) -> Void
        let approveMac: ((String) async -> Bool)?
        let macCommand = HarnessMacCommand()
        var macCommandBusy = false
        private(set) var terminalResult: Result<Void, Error>?
        var finish: CheckedContinuation<Void, Error>?
        var commits: [String: CheckedContinuation<Void, Error>] = [:]

        init(sessionId: String, approveMac: ((String) async -> Bool)?, onEvent: @escaping (ClaudeEvent) -> Void) {
            tracker = DeepSeekSDKTurnTracker(sessionId: sessionId)
            projection = DeepSeekSDKProjection(sessionId: sessionId)
            self.approveMac = approveMac
            self.onEvent = onEvent
        }

        func end(_ result: Result<Void, Error>) {
            guard terminalResult == nil else { return }
            terminalResult = result
            finish?.resume(with: result)
            finish = nil
            let waiting = commits
            commits = [:]
            for (_, waiter) in waiting { waiter.resume(throwing: result.failure ?? CancellationError()) }
        }
    }

    private init(agentID: UUID, configuration: Configuration, dataFolder: URL) throws {
        self.agentID = agentID
        self.configuration = configuration
        let folder = switch configuration.target {
        case .mac: "deepseek-mac-agents"
        case .box: "deepseek-box-agents"
        }
        let root = dataFolder.appendingPathComponent("\(folder)/\(agentID.uuidString)", isDirectory: true)
        // Each runtime gets fresh configuration. Session logs are the only state shared between runs.
        home = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        work = root.appendingPathComponent("work", isDirectory: true)
        sessions = root.appendingPathComponent("sessions", isDirectory: true)
        for folder in [home, work, sessions] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        }
    }

    private func start() async throws {
        let installation = configuration.installation
        let patch = home.appendingPathComponent("tools.patch.yml")
        let (profile, toolEnvironment) = switch configuration.target {
        case .mac: (DeepSeekMacProfile.patch, "AGENT_S_MAC")
        case .box: (DeepSeekBoxProfile.patch, "AGENT_S_BOX")
        }
        try profile.write(to: patch, atomically: true, encoding: .utf8)
        try DeepSeekStreamingPlugin.source.write(to: home.appendingPathComponent(DeepSeekStreamingPlugin.filename),
                                                 atomically: true, encoding: .utf8)
        var environment = [
            "HOME": home.path, "DSH_HOME": home.path, "DSH_AGENTS_HOME": home.path, "PATH": installation.path,
            "DSH_TELEMETRY_MODE": "DISABLED",
        ]
        if !installation.isPinned { try await checkVersion(installation, environment: environment) }

        let bridge = try await makeBridge()
        self.bridge = bridge
        let port = try await bridge.start()
        environment.merge([
            "AGENT_S_SDK": "1",
            "AGENT_S_MODEL": configuration.model, "AGENT_S_MODEL_URL": configuration.modelURL,
            "AGENT_S_MODEL_TOKEN": configuration.apiKey,
            "AGENT_S_MODEL_CONTEXT": String(configuration.contextTokens),
            "AGENT_S_MODEL_OUTPUT": String(configuration.outputTokens),
            "\(toolEnvironment)_URL": "http://127.0.0.1:\(port)/mcp",
            "\(toolEnvironment)_AUTHORIZATION": bridge.authorization,
            "AGENT_S_DSH_SESSIONS": sessions.path,
            "DSH_SYSTEM_PROMPT": configuration.instructions,
        ]) { _, value in value }

        let connection = DeepSeekRPCConnection(
            executable: installation.executable,
            arguments: DeepSeekSDKProfile.arguments(entry: installation.entry?.path, patchPath: patch.path),
            environment: environment, currentDirectory: work
        ) { [weak self] method, params in
            // Main-queue order keeps notifications in the order dsh wrote them.
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self?.receive(method, params.object) }
            }
        }
        self.connection = connection
        try connection.start()
        try await withTimeout(.seconds(90)) {
            _ = try await connection.request("initialize", params: [
                "cwd": self.work.path, "provider": DeepSeekSDKProfile.provider,
                "model": self.configuration.model, "maxTokens": self.configuration.outputTokens,
            ])
        }
        initialized = true
    }

    func run(prompt: String, resume: String?, approveMac: ((String) async -> Bool)?,
             onEvent: @escaping (ClaudeEvent) -> Void) async throws {
        guard turn == nil else { throw DeepSeekFailure("This agent is already working on a message.") }
        idleShutdown?.cancel()
        defer { scheduleIdleShutdown() }
        let session = try await openSession(resume)
        onEvent(.started(sessionId: session, model: nil))
        let turn = ActiveTurn(sessionId: session, approveMac: approveMac, onEvent: onEvent)
        self.turn = turn
        defer {
            turn.end(.failure(CancellationError()))
            turn.macCommand.stop()
            if self.turn === turn { self.turn = nil }
        }
        try await send(prompt, in: turn)
        try await withCheckedThrowingContinuation { (finish: CheckedContinuation<Void, Error>) in
            if exited { finish.resume(throwing: exitError()) }
            else if let result = turn.terminalResult { finish.resume(with: result) }
            else if turn.tracker.isFinished { finish.resume() }
            else { turn.finish = finish }
        }
        if let failure = turn.tracker.failure { throw DeepSeekFailure(failure) }
    }

    func steer(_ text: String) async throws {
        guard let turn, turn.terminalResult == nil, turn.tracker.started, !turn.tracker.isFinished, !exited else {
            throw SteeringDeliveryError.notDelivered
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, text.utf8.count <= 65_536 else {
            throw SteeringDeliveryError.notDelivered
        }
        let id = try await send(text, in: turn)
        guard !turn.tracker.isCommitted(id) else { return }
        try await withCheckedThrowingContinuation { (committed: CheckedContinuation<Void, Error>) in
            // The turn can end while the RPC is in flight. Input was sent, so delivery is uncertain.
            if self.turn !== turn || exited || turn.terminalResult != nil { committed.resume(throwing: CancellationError()) }
            else { turn.commits[id] = committed }
        }
    }

    func stop() {
        guard !stopping else { return }
        stopping = true
        idleShutdown?.cancel()
        turn?.macCommand.stop()
        bridge?.stop()
        connection?.terminate()
        if connection == nil { didExit(nil) }
        if Self.live[agentID] === self { Self.live[agentID] = nil }
    }

    func waitUntilExited() async -> Bool {
        guard let connection, !exited else { return true }
        return await connection.waitForExit() != nil
    }

    @discardableResult
    private func send(_ text: String, in turn: ActiveTurn) async throws -> String {
        guard let connection else { throw CancellationError() }
        let result = try await connection.request("session/prompt", params: [
            "sessionId": turn.tracker.sessionId, "contentBlocks": [["type": "text", "text": text]],
        ])
        guard let id = result.object["messageId"] as? String else {
            throw DeepSeekFailure("DeepSeek Harness didn't accept the message.")
        }
        turn.tracker.expect(messageId: id)
        return id
    }

    /// Continues `resume` when this runtime can, or starts a new session.
    private func openSession(_ resume: String?) async throws -> String {
        guard let connection else { throw CancellationError() }
        guard let resume, DeepSeekSDKProfile.isSessionID(resume) else { return DeepSeekSDKProfile.newSessionID() }
        if loadedSessions.contains(resume) { return resume }
        do {
            _ = try await connection.request("session/resume", params: ["sessionId": resume])
            loadedSessions.insert(resume)
            return resume
        } catch let error as DeepSeekRPCConnection.RPCError {
            // A conversation dsh can't restore (missing or from an incompatible release) starts over.
            Self.log.error("Couldn't resume a DeepSeek session: \(error.message, privacy: .private)")
            return DeepSeekSDKProfile.newSessionID()
        }
    }

    private func receive(_ method: String, _ params: [String: Any]) {
        if method == "agent_s/exit" {
            didExit(DeepSeekRPCConnection.ExitStatus(code: Int32((params["code"] as? Int) ?? 1),
                                                  signaled: params["signaled"] as? Bool ?? false))
            return
        }
        if method == "session.event", let id = params["sessionId"] as? String,
           (params["event"] as? [String: Any])?["type"] as? String == "user/message" {
            loadedSessions.insert(id)
        }
        guard let turn else { return }
        if turn.tracker.observe(method: method, params: params) { turn.onEvent(.steeringReady) }
        for event in turn.projection.consume(method: method, params: params) { turn.onEvent(event) }
        for (id, waiter) in turn.commits where turn.tracker.isCommitted(id) {
            turn.commits[id] = nil
            waiter.resume()
        }
        if turn.tracker.isFinished { turn.end(.success(())) }
    }

    private func didExit(_ status: DeepSeekRPCConnection.ExitStatus?) {
        guard !exited else { return }
        exited = true
        idleShutdown?.cancel()
        bridge?.stop()
        if Self.live[agentID] === self { Self.live[agentID] = nil }
        try? FileManager.default.removeItem(at: home)
        if !stopping, let status {
            let detail = connection?.diagnostics.suffix(2_000) ?? ""
            Self.log.error("DeepSeek runtime exited (status \(status.code), signal \(status.signaled)): \(String(detail), privacy: .private)")
        }
        turn?.end(.failure(exitError()))
    }

    private func exitError() -> Error {
        guard !stopping, let status = connection?.status else { return CancellationError() }
        if status.signaled, status.code == SIGKILL {
            return DeepSeekFailure(initialized
                ? "DeepSeek Harness was stopped by the system (signal 9)."
                : "DeepSeek Harness was stopped by the system as it started (signal 9). Security software on this Mac may have blocked it.")
        }
        return DeepSeekFailure(initialized ? "DeepSeek Harness exited before finishing the message."
                                           : "DeepSeek Harness couldn't start. Try again.")
    }

    private func scheduleIdleShutdown() {
        idleShutdown?.cancel()
        guard !exited, !stopping else { return }
        idleShutdown = Task { [weak self] in
            try? await Task.sleep(for: Self.idleLifetime)
            guard !Task.isCancelled, let self, self.turn == nil else { return }
            self.stop()
        }
    }

    // MARK: Tools

    private func makeBridge() async throws -> DeepSeekToolBridge {
        switch configuration.target {
        case let .box(boxName):
            let tools = try await BoxMCP.shared.agentTools(box: boxName)
            return try DeepSeekToolBridge(tools: tools, serverName: "Agent S Box") { [weak self] name, arguments in
                guard self?.turn != nil else { throw DeepSeekFailure("No message is running.") }
                return try await BoxMCP.shared.callAgentTool(name, arguments: arguments, box: boxName)
            }
        case .mac:
            return try DeepSeekToolBridge(tools: [Self.macTool], serverName: "Agent S") { [weak self] _, arguments in
                guard let self else { throw CancellationError() }
                return try await self.callMacTool(arguments)
            }
        }
    }

    private static let macTool: [String: Any] = [
        "name": "mac_command",
        "description": "Run a shell command on this Mac with the person's approval.",
        "inputSchema": ["type": "object", "properties": ["command": ["type": "string"]],
                        "required": ["command"], "additionalProperties": false],
    ]

    private func callMacTool(_ arguments: [String: Any]) async throws -> [String: Any] {
        // Only a running message can act on the Mac, and only with its own approval.
        guard let turn, let approve = turn.approveMac, !stopping else { throw DeepSeekFailure("No message is running.") }
        guard let command = arguments["command"] as? String, arguments.count == 1,
              !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !command.contains("\0") else {
            throw DeepSeekFailure("The Mac command is missing or invalid.")
        }
        guard !turn.macCommandBusy else { throw DeepSeekFailure("Wait for the previous Mac command to finish.") }
        turn.macCommandBusy = true
        defer { turn.macCommandBusy = false }

        guard await approve(command) else {
            throw DeepSeekFailure("The person denied this step. Ask them what they’d prefer.")
        }
        guard self.turn === turn, !stopping else { throw CancellationError() }
        let result = try await turn.macCommand.run(command)
        guard self.turn === turn, !stopping else { throw CancellationError() }
        var content = [["type": "text", "text": result.text]]
        if let image = result.image {
            content.append(["type": "image", "mimeType": "image/png", "data": image.base64EncodedString()])
        }
        return ["isError": !result.success, "content": content]
    }

    // MARK: Startup

    /// An installed `dsh` must be the pinned release; the bundled one is pinned at build time.
    private func checkVersion(_ installation: DeepSeekInstallation, environment: [String: String]) async throws {
        let process = Process()
        let output = Pipe()
        process.executableURL = installation.executable
        process.arguments = (installation.entry.map { [$0.path] } ?? []) + ["--version"]
        process.environment = environment
        process.currentDirectoryURL = work
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        let status: Int32 = try await withCheckedThrowingContinuation { finished in
            process.terminationHandler = { finished.resume(returning: $0.terminationStatus) }
            do { try process.run() } catch { finished.resume(throwing: error) }
        }
        let version = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        guard status == 0, version.trimmingCharacters(in: .whitespacesAndNewlines) == DeepSeekCommand.version else {
            throw DeepSeekFailure("Use DeepSeek Harness \(DeepSeekCommand.version).")
        }
    }

    private func withTimeout(_ limit: Duration, _ body: @escaping @MainActor () async throws -> Void) async throws {
        let timer = Task { [weak self] in
            try await Task.sleep(for: limit)
            self?.stop()
        }
        defer { timer.cancel() }
        do { try await body() } catch {
            throw exited || stopping ? exitError() : error
        }
    }
}

private extension Result {
    var failure: Failure? { if case let .failure(error) = self { error } else { nil } }
}

/// A runtime-scoped loopback MCP listener exposing only the selected computer's tools.
@MainActor
private final class DeepSeekToolBridge {
    let authorization = "Bearer \(UUID().uuidString)\(UUID().uuidString)"

    private let listener: NWListener
    private let tools: [[String: Any]]
    private let names: Set<String>
    private let serverName: String
    private let call: (String, [String: Any]) async throws -> [String: Any]
    private let queue = DispatchQueue(label: "agent-s.deepseek-tools")
    private var stopped = false

    init(tools: [[String: Any]], serverName: String,
         call: @escaping (String, [String: Any]) async throws -> [String: Any]) throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
        self.tools = tools
        names = Set(tools.compactMap { $0["name"] as? String })
        self.serverName = serverName
        self.call = call
    }

    func start() async throws -> UInt16 {
        listener.newConnectionHandler = { [weak self, queue] connection in
            guard let bridge = self else { connection.cancel(); return }
            connection.start(queue: queue)
            Self.read(connection, buffer: Data()) { [weak bridge] request in
                guard let bridge else { connection.cancel(); return }
                Task { @MainActor in
                    await bridge.reply(request, on: connection)
                }
            }
        }
        return try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { [listener] state in
                switch state {
                case .ready:
                    listener.stateUpdateHandler = nil
                    if let port = listener.port { continuation.resume(returning: port.rawValue) }
                    else { continuation.resume(throwing: CocoaError(.featureUnsupported)) }
                case let .failed(error):
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                case .cancelled:
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: CancellationError())
                default: break
                }
            }
            listener.start(queue: queue)
        }
    }

    func stop() {
        stopped = true
        listener.cancel()
    }

    nonisolated private static func read(_ connection: NWConnection, buffer: Data,
                                         onRequest: @escaping @Sendable (OpenRouterProxy.Request) -> Void) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { data, _, complete, error in
            var buffer = buffer
            if let data { buffer.append(data) }
            if buffer.count > 2 * 1024 * 1024 {
                connection.cancel()
            } else if let request = OpenRouterProxy.parse(buffer) {
                onRequest(request)
            } else if complete || error != nil {
                connection.cancel()
            } else {
                read(connection, buffer: buffer, onRequest: onRequest)
            }
        }
    }

    private func reply(_ request: OpenRouterProxy.Request, on connection: NWConnection) async {
        guard !stopped, request.path == "/mcp",
              request.headers.first(where: { $0.0.lowercased() == "authorization" })?.1 == authorization else {
            return respond(on: connection, status: "403 Forbidden")
        }
        guard request.method == "POST" else { return respond(on: connection, status: "405 Method Not Allowed") }
        guard let message = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
              message["jsonrpc"] as? String == "2.0", let method = message["method"] as? String else {
            return respond(on: connection, status: "400 Bad Request")
        }
        guard let id = message["id"] else { return respond(on: connection, status: "202 Accepted") }
        let params = message["params"] as? [String: Any] ?? [:]
        var result: [String: Any] = ["jsonrpc": "2.0", "id": id]
        switch method {
        case "initialize":
            let supported = ["2025-11-25", "2025-06-18", "2025-03-26"]
            let requested = params["protocolVersion"] as? String ?? ""
            result["result"] = ["protocolVersion": supported.contains(requested) ? requested : supported[0],
                                "capabilities": ["tools": [String: Bool]()],
                                "serverInfo": ["name": serverName, "version": "1"]]
        case "tools/list":
            result["result"] = ["tools": tools]
        case "tools/call":
            guard let name = params["name"] as? String, names.contains(name) else {
                result["error"] = ["code": -32602, "message": "That tool is unavailable."]
                return respond(on: connection, packet: result)
            }
            do {
                result["result"] = try await call(name, params["arguments"] as? [String: Any] ?? [:])
            } catch {
                result["result"] = ["isError": true, "content": [["type": "text", "text": error.localizedDescription]]]
            }
        case "ping":
            result["result"] = [String: String]()
        default:
            result["error"] = ["code": -32601, "message": "Unknown method."]
        }
        respond(on: connection, packet: result)
    }

    private func respond(on connection: NWConnection, status: String = "200 OK", packet: [String: Any]? = nil) {
        let body = packet.flatMap { try? JSONSerialization.data(withJSONObject: $0) } ?? Data()
        let headers = "HTTP/1.1 \(status)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        connection.send(content: Data(headers.utf8) + body, completion: .contentProcessed { _ in connection.cancel() })
    }
}
