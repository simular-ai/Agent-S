import Foundation
import Testing
@testable import AgentSCore

private func event(_ session: String, _ type: String, _ data: [String: Any] = [:], surface: String? = nil) -> [String: Any] {
    var event: [String: Any] = ["type": type, "data": data]
    if let surface { event["surfaceOp"] = surface }
    return ["sessionId": session, "event": event]
}

@Suite("DeepSeek SDK events")
struct DeepSeekSDKProjectionTests {
    @Test func projectsToolActivityTextAndUsageForItsSessionOnly() {
        var projection = DeepSeekSDKProjection(sessionId: "session-a")
        #expect(projection.consume(method: "session.event", params: event("session-a", "tool/call",
            ["callId": "call_1", "name": "mcp__agent_s__mac_command", "arguments": #"{"command":"ls"}"#])) ==
            [.toolCall(id: "call_1", name: "mcp__agent_s__mac_command", summary: "ls", detail: "ls")])
        #expect(projection.consume(method: "session.event", params: event("session-a", "tool/result",
            ["message": ["toolCallId": "call_1", "isError": false, "content": [["type": "text", "text": "file"]]]],
            surface: "append")) == [.toolResult(id: "call_1", output: "file", isError: false)])
        #expect(projection.consume(method: "session.event", params: event("session-a", "tool/result",
            ["message": ["toolCallId": "call_1"]], surface: "replace")).isEmpty)
        #expect(projection.consume(method: "session.event", params: event("session-a", "assistant/message",
            ["message": ["content": [["type": "reasoning", "text": "hidden"], ["type": "text", "text": "Done."]]],
             "usage": ["inputTokens": 10, "cacheReadTokens": 5, "outputTokens": 3]])) == [.text("Done.")])
        #expect(projection.consume(method: "session.event", params: event("session-a", "step/end")) ==
                [.usage(input: 15, output: 3)])
        #expect(projection.consume(method: "session.event", params: event("session-b", "assistant/message",
            ["message": ["content": [["type": "text", "text": "A subagent."]]]])).isEmpty)
    }

    @Test func prefersLiveTextFromThePlugin() {
        var projection = DeepSeekSDKProjection(sessionId: "session-a")
        #expect(projection.consume(method: "agent_s/stream",
                                   params: ["type": "agent_s_text_stream", "id": "a1", "sessionId": "session-a"]).isEmpty)
        #expect(projection.consume(method: "agent_s/stream", params: ["type": "agent_s_text_delta", "id": "a1", "text": "Hi"]) ==
                [.textDelta(id: "a1", text: "Hi")])
        #expect(projection.consume(method: "session.event", params: event("session-a", "assistant/message",
            ["message": ["content": [["type": "text", "text": "Hi there"]]]])).isEmpty)
        #expect(projection.consume(method: "agent_s/stream", params: ["type": "agent_s_text_snapshot", "id": "a1", "text": "Hi there"]) ==
                [.textSnapshot(id: "a1", text: "Hi there")])
        #expect(projection.consume(method: "agent_s/stream",
                                   params: ["type": "agent_s_text_stream", "id": "b1", "sessionId": "session-b"]).isEmpty)
    }
}

@Suite("DeepSeek SDK turns")
struct DeepSeekSDKTurnTrackerTests {
    @Test func finishesWhenEveryPromptIsCommittedAndTheAgentIsIdle() {
        var tracker = DeepSeekSDKTurnTracker(sessionId: "s")
        #expect(!tracker.isFinished)
        // The prompt's receipt can arrive after the agent has already read it.
        let began = tracker.observe(method: "session.status", params: ["sessionId": "s", "status": "running"])
        #expect(began)
        tracker.observe(method: "session.event", params: event("s", "user/message", ["id": "m1"]))
        tracker.expect(messageId: "m1")
        #expect(!tracker.isFinished)
        // A steer sent while running keeps the turn open until it is read and answered.
        tracker.expect(messageId: "m2")
        tracker.observe(method: "session.event", params: event("s", "turn/end", ["reason": ["kind": "completed"]]))
        tracker.observe(method: "session.status", params: ["sessionId": "s", "status": "idle"])
        #expect(!tracker.isFinished)
        let beganAgain = tracker.observe(method: "session.status", params: ["sessionId": "s", "status": "running"])
        #expect(!beganAgain)
        tracker.observe(method: "session.event", params: event("s", "user/message", ["id": "m2"]))
        tracker.observe(method: "session.status", params: ["sessionId": "other", "status": "idle"])
        #expect(!tracker.isFinished)
        tracker.observe(method: "session.status", params: ["sessionId": "s", "status": "idle"])
        #expect(tracker.isFinished)
        #expect(tracker.failure == nil)
    }

    @Test func reportsTheLastTurnFailure() {
        var tracker = DeepSeekSDKTurnTracker(sessionId: "s")
        tracker.expect(messageId: "m1")
        tracker.observe(method: "session.event", params: event("s", "user/message", ["id": "m1"]))
        tracker.observe(method: "session.event", params: event("s", "turn/end",
            ["reason": ["kind": "error", "error": ["message": "The model is unavailable."]]]))
        tracker.observe(method: "session.status", params: ["sessionId": "s", "status": "idle"])
        #expect(tracker.isFinished)
        #expect(tracker.failure == "The model is unavailable.")
    }
}

/// Drives the vendored runtime end to end with synthetic model and MCP servers. Runs when
/// AGENT_S_DSH_RUNTIME names a runtime folder (`scripts/vendor-deepseek.sh --check`).
@Suite("DeepSeek SDK runtime", .enabled(if: ProcessInfo.processInfo.environment["AGENT_S_DSH_RUNTIME"] != nil),
       .timeLimit(.minutes(3)))
struct DeepSeekSDKRuntimeTests {
    private final class Inbox: @unchecked Sendable {
        let stream: AsyncStream<(String, DeepSeekJSON)>
        let continuation: AsyncStream<(String, DeepSeekJSON)>.Continuation
        init() { (stream, continuation) = AsyncStream.makeStream() }
    }

    private struct Runtime {
        let connection: DeepSeekRPCConnection
        var notifications: AsyncStream<(String, DeepSeekJSON)>.Iterator

        /// Sends prompts on `session` (later ones while the first runs) and collects its events.
        mutating func turn(_ session: String, _ prompts: [String]) async throws -> (events: [ClaudeEvent], failure: String?) {
            var tracker = DeepSeekSDKTurnTracker(sessionId: session)
            var projection = DeepSeekSDKProjection(sessionId: session)
            var events: [ClaudeEvent] = []
            var queued = prompts
            func send(_ text: String) async throws {
                let result = try await connection.request("session/prompt", params: [
                    "sessionId": session, "contentBlocks": [["type": "text", "text": text]]])
                tracker.expect(messageId: try #require(result.object["messageId"] as? String))
            }
            try await send(queued.removeFirst())
            while !tracker.isFinished, let (method, params) = await notifications.next() {
                if method == "agent_s/exit" { throw DeepSeekRPCConnection.Exited(status: connection.status!) }
                if tracker.observe(method: method, params: params.object), !queued.isEmpty {
                    try await send(queued.removeFirst())
                }
                events += projection.consume(method: method, params: params.object)
            }
            return (events, tracker.failure)
        }
    }

    private let root = FileManager.default.temporaryDirectory.appendingPathComponent("agent-s-dsh-\(UUID().uuidString)")

    private func launch(services: [String: Int], extra: [String: String]) async throws -> Runtime {
        let installation = URL(fileURLWithPath: ProcessInfo.processInfo.environment["AGENT_S_DSH_RUNTIME"]!)
        let home = root.appendingPathComponent(UUID().uuidString)
        let work = root.appendingPathComponent("work"), sessions = root.appendingPathComponent("sessions")
        for folder in [home, work, sessions] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        let patch = home.appendingPathComponent("tools.patch.yml")
        try DeepSeekMacProfile.patch.write(to: patch, atomically: true, encoding: .utf8)
        try DeepSeekStreamingPlugin.source.write(to: home.appendingPathComponent(DeepSeekStreamingPlugin.filename),
                                                 atomically: true, encoding: .utf8)
        let environment = [
            "HOME": home.path, "DSH_HOME": home.path, "DSH_AGENTS_HOME": home.path, "PATH": "/usr/bin:/bin",
            "DSH_TELEMETRY_MODE": "DISABLED", "AGENT_S_SDK": "1",
            "AGENT_S_MODEL": "synthetic/model", "AGENT_S_MODEL_URL": "http://127.0.0.1:\(services["model"]!)/v1",
            "AGENT_S_MODEL_TOKEN": "synthetic-key", "AGENT_S_MODEL_CONTEXT": "100000", "AGENT_S_MODEL_OUTPUT": "4096",
            "AGENT_S_MAC_URL": "http://127.0.0.1:\(services["mcp"]!)/mcp", "AGENT_S_MAC_AUTHORIZATION": "Bearer synthetic",
            "AGENT_S_DSH_SESSIONS": sessions.path, "DSH_SYSTEM_PROMPT": "You are a synthetic test agent.",
        ].merging(extra) { _, value in value }
        let inbox = Inbox()
        let connection = DeepSeekRPCConnection(
            executable: installation.appendingPathComponent("bin/node"),
            arguments: DeepSeekSDKProfile.arguments(
                entry: installation.appendingPathComponent("node_modules/@deepseek-ai/dsh/lib/bin.js").path,
                patchPath: patch.path),
            environment: environment, currentDirectory: work
        ) { method, params in inbox.continuation.yield((method, params)) }
        try connection.start()
        _ = try await connection.request("initialize", params: [
            "cwd": work.path, "provider": DeepSeekSDKProfile.provider, "model": "synthetic/model", "maxTokens": 4096])
        return Runtime(connection: connection, notifications: inbox.stream.makeAsyncIterator())
    }

    @Test func runsTurnsSteersStopsAndResumes() async throws {
        let tools = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../DeepSeekRuntime/tools/fake_services.py").standardizedFileURL
        let services = Process()
        let servicesInput = Pipe(), servicesOutput = Pipe()
        services.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        services.arguments = ["-I", tools.path]
        services.standardInput = servicesInput
        services.standardOutput = servicesOutput
        try services.run()
        defer { try? servicesInput.fileHandleForWriting.close(); services.waitUntilExit(); try? FileManager.default.removeItem(at: root) }
        var line = Data()
        while let byte = try servicesOutput.fileHandleForReading.read(upToCount: 1), byte != Data([0x0A]) { line += byte }
        let ports = try #require(try JSONSerialization.jsonObject(with: line) as? [String: Int])
        // Tracing for scripts/vendor-deepseek.sh --trace.
        let extra = ProcessInfo.processInfo.environment.filter { ["NODE_OPTIONS", "AGENT_S_TRACE"].contains($0.key) }

        var runtime = try await launch(services: ports, extra: extra)
        let session = DeepSeekSDKProfile.newSessionID()
        let first = try await runtime.turn(session, ["Please use tool"])
        #expect(first.failure == nil)
        #expect(first.events.contains { if case let .toolCall(_, name, _, _) = $0 { name == "mcp__agent_s__mac_command" } else { false } })
        #expect(first.events.contains { if case let .toolResult(_, output, false) = $0 { output == "synthetic output" } else { false } })

        let clock = ContinuousClock()
        let started = clock.now
        let second = try await runtime.turn(session, ["Second message"])
        #expect(clock.now - started < .seconds(5))
        #expect(second.events.contains { if case let .textSnapshot(_, text) = $0 { text.contains("Second message") } else { false } })

        let steered = try await runtime.turn(session, ["Please go slow", "Add detail"])
        #expect(steered.events.contains { if case let .textSnapshot(_, text) = $0 { text.contains("Add detail") } else { false } })

        // Stop has no protocol method: the runtime exits on SIGTERM and a new one resumes.
        runtime.connection.terminate()
        let stopped = await runtime.connection.waitForExit()
        #expect(stopped?.code == 0)

        var resumed = try await launch(services: ports, extra: extra)
        _ = try await resumed.connection.request("session/resume", params: ["sessionId": session])
        let after = try await resumed.turn(session, ["After resume"])
        #expect(after.failure == nil)
        await #expect(throws: DeepSeekRPCConnection.RPCError.self) {
            _ = try await resumed.connection.request("session/resume", params: ["sessionId": DeepSeekSDKProfile.newSessionID()])
        }
        _ = try await resumed.connection.request("shutdown")
        #expect(await resumed.connection.waitForExit()?.code == 0)
    }
}
