import Foundation
import Network
import AgentSCore

/// DeepSeek Harness runs locally with tools supplied by its turn-scoped MCP bridge.
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

    static var executable: URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let versions = home.appendingPathComponent(".nvm/versions/node", isDirectory: true)
        let nvm = (try? FileManager.default.contentsOfDirectory(atPath: versions.path)) ?? []
        let paths = [home.appendingPathComponent(".local/bin/dsh").path,
                     "/opt/homebrew/bin/dsh", "/usr/local/bin/dsh",
                     home.appendingPathComponent(".npm-global/bin/dsh").path]
            + nvm.sorted(by: >).map { versions.appendingPathComponent("\($0)/bin/dsh").path }
        return paths.first(where: FileManager.default.isExecutableFile(atPath:)).map(URL.init(fileURLWithPath:))
    }

    private var process: Process?
    private var bridge: DeepSeekToolBridge?
    private let macCommand = HarnessMacCommand()
    private var macCommandBusy = false
    private var stopped = false

    func run(prompt: String, resume: String?, model: String, baseURL: String, apiKey: String,
             dataFolder: URL, agentID: UUID, target: Target, standardInput: Pipe? = nil,
             onEvent: @escaping (ClaudeEvent) -> Void) async throws {
        guard let executable = Self.executable else {
            throw Failure("Install DeepSeek Harness \(DeepSeekCommand.version) on this Mac.")
        }
        guard var endpoint = URLComponents(string: baseURL), endpoint.scheme == "http",
              endpoint.host == "127.0.0.1", endpoint.port != nil, endpoint.user == nil, endpoint.password == nil,
              endpoint.query == nil, endpoint.fragment == nil, ["", "/", "/v1", "/v1/"].contains(endpoint.path) else {
            throw Failure("DeepSeek Harness needs the Simular model endpoint.")
        }
        endpoint.path = "/v1"
        guard let modelURL = endpoint.url else { throw Failure("The model endpoint is invalid.") }

        let folder: String
        let config: String
        let instructions: String
        let toolEnvironment: String
        let targetName: String
        switch target {
        case let .box(boxName):
            folder = "deepseek-box-agents"
            config = DeepSeekBoxProfile.patch
            instructions = Self.instructions(boxName: boxName)
            toolEnvironment = "AGENT_S_BOX"
            targetName = "Simular Box"
        case .mac:
            folder = "deepseek-mac-agents"
            config = DeepSeekMacProfile.patch
            instructions = Self.macInstructions
            toolEnvironment = "AGENT_S_MAC"
            targetName = "This Mac"
        }
        let root = dataFolder.appendingPathComponent("\(folder)/\(agentID.uuidString)", isDirectory: true)
        // Each launch uses fresh configuration. Session logs are the only shared part of the runtime.
        let home = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let work = root.appendingPathComponent("work", isDirectory: true)
        let sessions = root.appendingPathComponent("sessions", isDirectory: true)
        for folder in [home, work, sessions] {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        }
        defer { try? FileManager.default.removeItem(at: home) }
        let patch = home.appendingPathComponent("tools.patch.yml")
        try config.write(to: patch, atomically: true, encoding: .utf8)
        let streamPlugin = home.appendingPathComponent(DeepSeekStreamingPlugin.filename)
        try DeepSeekStreamingPlugin.source.write(to: streamPlugin, atomically: true, encoding: .utf8)

        let path = executable.deletingLastPathComponent().path + ":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        var environment = [
            "HOME": home.path, "DSH_HOME": home.path, "DSH_AGENTS_HOME": home.path, "PATH": path,
            "DSH_TELEMETRY_MODE": "DISABLED",
        ]
        var version = ""
        let check = try await launch(executable: executable, arguments: ["--version"], environment: environment, work: work) { version = $0 }
        guard check == 0, version.trimmingCharacters(in: .whitespacesAndNewlines) == DeepSeekCommand.version else {
            throw Failure("Use DeepSeek Harness \(DeepSeekCommand.version) for \(targetName).")
        }

        let bridge = try await toolBridge(for: target)
        self.bridge = bridge
        defer { stop(); self.bridge = nil }
        let port = try await bridge.start()
        let provider: ModelProvider = endpoint.port == Int(Ollama.port) ? .ollama : .openRouter
        environment.merge([
            "AGENT_S_MODEL": model, "AGENT_S_MODEL_URL": modelURL.absoluteString, "AGENT_S_MODEL_TOKEN": apiKey,
            "AGENT_S_MODEL_CONTEXT": String(provider.contextTokens), "AGENT_S_MODEL_OUTPUT": String(provider.maxOutputTokens),
            "\(toolEnvironment)_URL": "http://127.0.0.1:\(port)/mcp",
            "\(toolEnvironment)_AUTHORIZATION": bridge.authorization, "AGENT_S_DSH_SESSIONS": sessions.path,
            "DSH_SYSTEM_PROMPT": instructions,
        ]) { _, value in value }
        if standardInput != nil { environment["AGENT_S_STEERING"] = "1" }

        let arguments = switch target {
        case .box: DeepSeekBoxProfile.arguments(prompt: prompt, resume: resume, patchPath: patch.path)
        case .mac: DeepSeekMacProfile.arguments(prompt: prompt, resume: resume, patchPath: patch.path)
        }
        var stream = DeepSeekStream()
        var final: ClaudeEvent?
        let status = try await launch(executable: executable, arguments: arguments, environment: environment, work: work,
                                      standardInput: standardInput) { line in
            guard bridge.listedTools else { return }
            for event in stream.consume(line: line) {
                if case .finished = event { final = event }
                else { onEvent(event) }
            }
        }
        guard bridge.listedTools else { throw Failure("DeepSeek Harness couldn't load the tools for \(targetName).") }
        if case let .finished(reason, true, _) = final { throw Failure(reason ?? "DeepSeek Harness stopped.") }
        guard status == 0, final != nil else { throw Failure("DeepSeek Harness exited before completing the turn.") }
    }

    func stop() {
        stopped = true
        bridge?.stop()
        macCommand.stop()
        if let process, process.isRunning { process.terminate() }
    }

    private func toolBridge(for target: Target) async throws -> DeepSeekToolBridge {
        switch target {
        case let .box(boxName):
            let tools = try await BoxMCP.shared.agentTools(box: boxName)
            try checkRunning()
            return try DeepSeekToolBridge(tools: tools, serverName: "Agent S Box") { name, arguments in
                try await BoxMCP.shared.callAgentTool(name, arguments: arguments, box: boxName)
            }
        case let .mac(approve):
            try checkRunning()
            return try DeepSeekToolBridge(tools: [Self.macTool], serverName: "Simular") { [weak self] _, arguments in
                guard let self else { throw CancellationError() }
                return try await self.callMacTool(arguments, approve: approve)
            }
        }
    }

    private static let macTool: [String: Any] = [
        "name": "mac_command",
        "description": "Run a shell command on this Mac with the person's approval.",
        "inputSchema": ["type": "object", "properties": ["command": ["type": "string"]],
                        "required": ["command"], "additionalProperties": false],
    ]

    private func callMacTool(_ arguments: [String: Any], approve: (String) async -> Bool) async throws -> [String: Any] {
        try checkRunning()
        guard let command = arguments["command"] as? String, arguments.count == 1,
              !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !command.contains("\0") else {
            throw Failure("The Mac command is missing or invalid.")
        }
        guard !macCommandBusy else { throw Failure("Wait for the previous Mac command to finish.") }
        macCommandBusy = true
        defer { macCommandBusy = false }

        guard await approve(command) else {
            throw Failure("The person denied this step. Ask them what they’d prefer.")
        }
        try checkRunning()
        let result = try await macCommand.run(command)
        try checkRunning()
        var content = [["type": "text", "text": result.text]]
        if let image = result.image {
            content.append(["type": "image", "mimeType": "image/png", "data": image.base64EncodedString()])
        }
        return ["isError": !result.success, "content": content]
    }

    private func checkRunning() throws {
        try Task.checkCancellation()
        if stopped { throw CancellationError() }
    }

    private func launch(executable: URL, arguments: [String], environment: [String: String], work: URL,
                        standardInput: Pipe? = nil, onLine: (String) -> Void) async throws -> Int32 {
        try checkRunning()
        let process = Process()
        let output = Pipe()
        let stream = AsyncStream<String>.makeStream()
        let reader = LineReader { stream.continuation.yield($0) }
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.currentDirectoryURL = work
        if let standardInput { process.standardInput = standardInput }
        else { process.standardInput = FileHandle.nullDevice }
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        output.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                reader.flush()
                stream.continuation.finish()
            } else { reader.feed(data) }
        }
        self.process = process
        defer {
            output.fileHandleForReading.readabilityHandler = nil
            self.process = nil
            if process.isRunning { process.terminate() }
        }
        try process.run()
        let (statuses, finish) = AsyncStream<Int32>.makeStream()
        process.terminationHandler = { process in
            finish.yield(process.terminationStatus)
            finish.finish()
        }
        if !process.isRunning {
            finish.yield(process.terminationStatus)
            finish.finish()
        }
        for await line in stream.stream {
            try checkRunning()
            onLine(line)
        }
        let status = await statuses.first(where: { _ in true })
        try checkRunning()
        return status ?? 1
    }

    private struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }
}

/// A turn-scoped loopback MCP listener exposing only the selected computer's tools.
@MainActor
private final class DeepSeekToolBridge {
    let authorization = "Bearer \(UUID().uuidString)\(UUID().uuidString)"
    private(set) var listedTools = false

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
            listedTools = true
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
