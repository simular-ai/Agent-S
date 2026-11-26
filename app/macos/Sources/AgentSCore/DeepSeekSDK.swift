import Foundation

/// Projects the SDK runtime's notifications for one session onto chat events. It mirrors dsh's
/// headless `--json` projection so both harness modes share `DeepSeekStream`: committed text and
/// tool activity come from durable session events, live text from the bundled plugin.
public struct DeepSeekSDKProjection {
    public let sessionId: String
    private var stream = DeepSeekStream()
    private var stepUsage: (input: Int, output: Int)?

    public init(sessionId: String) {
        self.sessionId = sessionId
    }

    public mutating func consume(method: String, params: [String: Any]) -> [ClaudeEvent] {
        switch method {
        case "agent_s/stream":
            // One runtime serves one root session; subagents never stream through the plugin.
            if let id = params["sessionId"] as? String, id != sessionId { return [] }
            return stream.consume(object: params)
        case "session.event":
            guard params["sessionId"] as? String == sessionId,
                  let event = params["event"] as? [String: Any] else { return [] }
            return consume(event: event)
        default:
            return []
        }
    }

    private mutating func consume(event: [String: Any]) -> [ClaudeEvent] {
        let data = event["data"] as? [String: Any] ?? [:]
        switch event["type"] as? String {
        case "assistant/message":
            if let usage = data["usage"] as? [String: Any] {
                func n(_ key: String) -> Int { (usage[key] as? NSNumber)?.intValue ?? 0 }
                let input = n("inputTokens") + n("cacheReadTokens"), output = n("outputTokens")
                stepUsage = ((stepUsage?.input ?? 0) + input, (stepUsage?.output ?? 0) + output)
            }
            let content = (data["message"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
            return content.flatMap { block -> [ClaudeEvent] in
                guard block["type"] as? String == "text", let text = block["text"] as? String else { return [] }
                return stream.consume(object: ["type": "text", "text": text])
            }
        case "step/end":
            defer { stepUsage = nil }
            guard let usage = stepUsage, usage.input + usage.output > 0 else { return [] }
            return [.usage(input: usage.input, output: usage.output)]
        case "tool/call":
            let raw = data["arguments"] as? String ?? ""
            let input: Any = raw.isEmpty ? [String: Any]()
                : (raw.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? raw)
            return stream.consume(object: ["type": "tool_call", "callId": data["callId"] as? String ?? "",
                                           "tool": data["name"] as? String ?? "tool", "input": input])
        case "tool/result":
            guard event["surfaceOp"] as? String == "append",
                  let message = data["message"] as? [String: Any] else { return [] }
            let blocks = message["content"] as? [[String: Any]] ?? []
            let text = blocks.compactMap { $0["type"] as? String == "text" ? $0["text"] as? String : nil }.joined()
            return stream.consume(object: ["type": "tool_result", "callId": message["toolCallId"] as? String ?? "",
                                           "status": message["isError"] as? Bool == true ? "error" : "completed",
                                           "result": text])
        default:
            return []
        }
    }
}

/// Decides when one chat turn on the SDK runtime has finished. The runtime acknowledges a prompt
/// before the agent reads it, and a prompt sent while the agent runs starts the next turn, so a
/// turn ends once every prompt it sent is committed and the agent has gone idle after that.
public struct DeepSeekSDKTurnTracker {
    public let sessionId: String
    private var expected: Set<String> = []
    private var committed: Set<String> = []
    private var idleAfterCommit = false
    public private(set) var started = false
    /// The last committed turn ending: `completed`, or a failure description.
    public private(set) var failure: String?

    public init(sessionId: String) {
        self.sessionId = sessionId
    }

    public mutating func expect(messageId: String) {
        expected.insert(messageId)
        if !committed.contains(messageId) { idleAfterCommit = false }
    }

    public func isCommitted(_ messageId: String) -> Bool { committed.contains(messageId) }

    public var isFinished: Bool { !expected.isEmpty && expected.isSubset(of: committed) && idleAfterCommit }

    /// Records one notification. Returns true when the agent began running for this turn.
    @discardableResult
    public mutating func observe(method: String, params: [String: Any]) -> Bool {
        guard params["sessionId"] as? String == sessionId else { return false }
        switch method {
        case "session.status":
            switch params["status"] as? String {
            case "running":
                defer { started = true }
                return !started
            case "idle":
                if !expected.isEmpty, expected.isSubset(of: committed) { idleAfterCommit = true }
            default: break
            }
        case "session.event":
            guard let event = params["event"] as? [String: Any],
                  let data = event["data"] as? [String: Any] else { return false }
            switch event["type"] as? String {
            case "user/message":
                if let id = data["id"] as? String {
                    committed.insert(id)
                    idleAfterCommit = false
                }
            case "turn/end":
                let reason = data["reason"] as? [String: Any]
                let kind = reason?["kind"] as? String
                if kind == "completed" { failure = nil }
                else {
                    let error = (reason?["error"] as? [String: Any])?["message"] as? String
                    failure = error ?? "DeepSeek Harness stopped: \(kind ?? "unknown")"
                }
            default: break
            }
        default: break
        }
        return false
    }
}

/// A JSON object crossing from the reader thread. Only plain JSON values are inside.
public struct DeepSeekJSON: @unchecked Sendable {
    public let object: [String: Any]
    public init(_ object: [String: Any]) { self.object = object }
}

/// The client end of dsh's SDK protocol over one child process's stdio: newline-delimited
/// JSON-RPC 2.0. Stdout carries only frames; stderr is kept as a bounded tail for diagnostics.
public final class DeepSeekRPCConnection: @unchecked Sendable {
    public struct ExitStatus: Sendable, Equatable {
        public let code: Int32
        public let signaled: Bool

        public init(code: Int32, signaled: Bool) {
            self.code = code
            self.signaled = signaled
        }
    }

    public struct RPCError: LocalizedError {
        public let code: Int
        public let message: String
        public var errorDescription: String? { message }
    }

    public struct Exited: LocalizedError {
        public let status: ExitStatus
        public var errorDescription: String? {
            status.signaled ? "DeepSeek Harness was stopped by signal \(status.code)."
                : "DeepSeek Harness exited (status \(status.code))."
        }
    }

    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private let errors = Pipe()
    private let lock = NSLock()
    private var nextID = 0
    private var pending: [Int: CheckedContinuation<DeepSeekJSON, Error>] = [:]
    private var buffer = Data()
    private var stderrTail = Data()
    private var exitStatus: ExitStatus?
    private var exitWaiters: [CheckedContinuation<ExitStatus, Never>] = []
    private let onNotification: @Sendable (String, DeepSeekJSON) -> Void
    private static let stderrLimit = 16 * 1024

    public init(executable: URL, arguments: [String], environment: [String: String], currentDirectory: URL,
                onNotification: @escaping @Sendable (String, DeepSeekJSON) -> Void) {
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.currentDirectoryURL = currentDirectory
        self.onNotification = onNotification
    }

    public var processIdentifier: Int32 { process.processIdentifier }

    public var status: ExitStatus? { lock.withLock { exitStatus } }

    /// The end of the runtime's stderr, for logs. It can name local paths, so never show it.
    public var diagnostics: String { lock.withLock { String(decoding: stderrTail, as: UTF8.self) } }

    public func start() throws {
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
        // Writes after an unexpected exit must fail, not raise SIGPIPE in the app.
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil } else { self?.receive(data) }
        }
        errors.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }
            guard let self else { return }
            self.lock.withLock {
                self.stderrTail.append(data)
                if self.stderrTail.count > Self.stderrLimit { self.stderrTail.removeFirst(self.stderrTail.count - Self.stderrLimit) }
            }
        }
        process.terminationHandler = { [weak self] process in
            self?.finish(ExitStatus(code: process.terminationStatus, signaled: process.terminationReason == .uncaughtSignal))
        }
        try process.run()
    }

    /// Sends one request and waits for its response's `result`.
    public func request(_ method: String, params: [String: Any]? = nil) async throws -> DeepSeekJSON {
        try await withCheckedThrowingContinuation { (reply: CheckedContinuation<DeepSeekJSON, Error>) in
            let id: Int? = lock.withLock {
                if let exitStatus {
                    reply.resume(throwing: Exited(status: exitStatus))
                    return nil
                }
                nextID += 1
                pending[nextID] = reply
                return nextID
            }
            guard let id else { return }
            var frame: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
            if let params { frame["params"] = params }
            send(frame) { [weak self] error in
                guard let self, let reply = self.lock.withLock({ self.pending.removeValue(forKey: id) }) else { return }
                reply.resume(throwing: error)
            }
        }
    }

    /// Asks the runtime to stop. dsh disposes its sessions and flushes persistence on SIGTERM.
    public func terminate() {
        if process.isRunning { process.terminate() }
    }

    /// Waits for the process to exit, up to `timeout`.
    public func waitForExit(timeout: Duration = .seconds(8)) async -> ExitStatus? {
        if let status { return status }
        return await withTaskGroup(of: ExitStatus?.self) { group in
            group.addTask {
                await withCheckedContinuation { (waiter: CheckedContinuation<ExitStatus, Never>) in
                    let done: ExitStatus? = self.lock.withLock {
                        if let exitStatus = self.exitStatus { return exitStatus }
                        self.exitWaiters.append(waiter)
                        return nil
                    }
                    if let done { waiter.resume(returning: done) }
                }
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    private func send(_ frame: [String: Any], failed: (@Sendable (Error) -> Void)? = nil) {
        do {
            var line = try JSONSerialization.data(withJSONObject: frame)
            line.append(0x0A)
            try input.fileHandleForWriting.write(contentsOf: line)
        } catch {
            failed?(error)
        }
    }

    private func receive(_ data: Data) {
        let lines: [Data] = lock.withLock {
            buffer.append(data)
            var lines: [Data] = []
            while let newline = buffer.firstIndex(of: 0x0A) {
                lines.append(buffer[buffer.startIndex..<newline])
                buffer.removeSubrange(buffer.startIndex...newline)
            }
            return lines
        }
        for line in lines { handle(line) }
    }

    private func handle(_ line: Data) {
        guard let frame = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { return }
        let method = frame["method"] as? String
        if let method {
            if let id = frame["id"] {
                // Server requests (approvals, questions) are not enabled; refuse rather than hang.
                send(["jsonrpc": "2.0", "id": id, "error": ["code": -32601, "message": "Unsupported by Agent S."]])
            } else {
                onNotification(method, DeepSeekJSON(frame["params"] as? [String: Any] ?? [:]))
            }
            return
        }
        guard let id = (frame["id"] as? NSNumber)?.intValue,
              let reply = lock.withLock({ pending.removeValue(forKey: id) }) else { return }
        if let error = frame["error"] as? [String: Any] {
            reply.resume(throwing: RPCError(code: (error["code"] as? NSNumber)?.intValue ?? -32603,
                                            message: error["message"] as? String ?? "DeepSeek Harness reported an error."))
        } else {
            reply.resume(returning: DeepSeekJSON(frame["result"] as? [String: Any] ?? [:]))
        }
    }

    private func finish(_ status: ExitStatus) {
        // Frames already read stay ordered before the exit.
        output.fileHandleForReading.readabilityHandler = nil
        let rest = output.fileHandleForReading.readDataToEndOfFile()
        if !rest.isEmpty { receive(rest) }
        let (replies, waiters): ([CheckedContinuation<DeepSeekJSON, Error>], [CheckedContinuation<ExitStatus, Never>]) = lock.withLock {
            exitStatus = status
            defer { pending = [:]; exitWaiters = [] }
            return (Array(pending.values), exitWaiters)
        }
        try? input.fileHandleForWriting.close()
        for reply in replies { reply.resume(throwing: Exited(status: status)) }
        for waiter in waiters { waiter.resume(returning: status) }
        onNotification("agent_s/exit", DeepSeekJSON(["code": Int(status.code), "signaled": status.signaled]))
    }
}
