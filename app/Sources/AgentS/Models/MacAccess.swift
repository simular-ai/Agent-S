import AppKit
import ApplicationServices
import CoreGraphics
import Foundation
import Network
import AgentSCore

/// What a "This Mac" agent needs on the Mac: the `mac` tool and approval hook (installed in
/// ~/.agent-s/bin, a path without spaces the model can type), simulang, and macOS's Accessibility
/// and Screen Recording permissions for Agent S.
enum MacAccess {
    static let bin = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".agent-s/bin", isDirectory: true)
    static var macTool: String { bin.appendingPathComponent("mac").path }
    static var hook: String { bin.appendingPathComponent("approve-hook").path }
    static var settings: String { bin.deletingLastPathComponent().appendingPathComponent("claude-settings.json").path }

    /// Writes the tools (again on every turn, so updates reach them). Only this user can change them.
    static func install() throws {
        let fm = FileManager.default
        try fm.createDirectory(at: bin, withIntermediateDirectories: true)
        try? fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: bin.deletingLastPathComponent().path)
        let files: [(String, String, Int)] = [
            (macTool, MacAgent.macTool, 0o700),
            (bin.appendingPathComponent("mac-ui.mts").path, MacAgent.macUIScript, 0o600),
            (hook, MacAgent.approvalHook, 0o700),
            (settings, MacAgent.settings(hookPath: hook), 0o600),
        ]
        for (path, content, mode) in files {
            try Data(content.utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
            try fm.setAttributes([.posixPermissions: mode], ofItemAtPath: path)
        }
    }

    /// simulang, from npm (nvm, Homebrew's node or the system's).
    static var simulang: URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        var candidates = ["/opt/homebrew/bin/simulang", "/usr/local/bin/simulang"]
        let nvm = "\(home)/.nvm/versions/node"
        if let versions = try? FileManager.default.contentsOfDirectory(atPath: nvm) {
            candidates += versions.sorted().reversed().map { "\(nvm)/\($0)/bin/simulang" }
        }
        return candidates.map(URL.init(fileURLWithPath:)).first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// PATH for the agent's commands: the tools, simulang's node, then the usual places.
    static var path: String {
        var dirs = [bin.path]
        if let simulang { dirs.append(simulang.deletingLastPathComponent().path) }
        dirs += ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        return dirs.joined(separator: ":")
    }

    /// Accessibility: clicking and typing in other apps, and reading their windows.
    static var hasAccessibility: Bool { AXIsProcessTrusted() }

    static func askForAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        if !AXIsProcessTrustedWithOptions(options) { openPrivacy("Privacy_Accessibility") }
    }

    /// Screen Recording: screenshots.
    static var hasScreenRecording: Bool { CGPreflightScreenCaptureAccess() }

    static func askForScreenRecording() {
        if !CGRequestScreenCaptureAccess() { openPrivacy("Privacy_ScreenCapture") }
    }

    private static func openPrivacy(_ pane: String) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)") {
            NSWorkspace.shared.open(url)
        }
    }
}

/// Listens on the Mac's loopback for the approval hook: each request is one tool use a "This Mac"
/// agent wants to run. It is answered "allow" only after the person allows it in the chat.
/// Requests need this launch's token (other apps or web pages on the Mac can't ask).
final class ApprovalServer: @unchecked Sendable {
    let token = UUID().uuidString + UUID().uuidString
    private(set) var port: UInt16 = 0
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "agent-s.approvals")
    /// Finds the agent and asks the person; true to allow.
    var decide: @MainActor (_ agent: String, _ tool: String, _ input: [String: Any]) async -> Bool = { _, _, _ in false }
    typealias CommandResult = (text: String, success: Bool, image: Data?)
    var runCommand: @MainActor (_ agent: String, _ turn: String, _ command: String) async -> CommandResult = {
        _, _, _ in ("This turn has ended.", false, nil)
    }

    static func claudeToolConfig(port: UInt16) throws -> Data {
        try JSONSerialization.data(withJSONObject: ["mcpServers": ["agent_s": [
            "type": "http",
            "url": "http://127.0.0.1:\(port)/mac-tools",
            "headers": [
                "Authorization": "Bearer ${AGENT_S_APPROVAL_TOKEN}",
                "X-Agent-S-Agent": "${AGENT_S_AGENT}",
                "X-Agent-S-Turn": "${AGENT_S_TURN}",
            ],
        ]]])
    }

    func start() throws {
        guard listener == nil else { return }
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in self?.serve(connection) }
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed, .cancelled: ready.signal()
            default: break
            }
        }
        listener.start(queue: queue)
        _ = ready.wait(timeout: .now() + 3)
        guard let port = listener.port?.rawValue, port != 0 else {
            listener.cancel()
            throw CocoaError(.featureUnsupported)
        }
        self.port = port
        self.listener = listener
    }

    private func serve(_ connection: NWConnection) {
        connection.start(queue: queue)
        read(connection, buffer: Data())
    }

    private func read(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let request = OpenRouterProxy.parse(buffer) {
                Task { await self.answer(request, on: connection) }
            } else if isComplete || error != nil || buffer.count > 4 << 20 {
                connection.cancel()
            } else {
                self.read(connection, buffer: buffer)
            }
        }
    }

    private func answer(_ request: OpenRouterProxy.Request, on connection: NWConnection) async {
        let header = { (name: String) in request.headers.first { $0.0.lowercased() == name }?.1 }
        if request.path == "/mac-tools" {
            guard header("authorization") == "Bearer \(token)", let agent = header("x-agent-s-agent"),
                  let turn = header("x-agent-s-turn") else {
                respond(connection, status: "403 Forbidden")
                return
            }
            await respondToMCP(request, on: connection, agent: agent, turn: turn)
            return
        }
        guard request.method == "POST", request.path == "/approve",
              header("authorization") == "Bearer \(token)",
              let body = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
              let tool = body["tool_name"] as? String
        else {
            respond(connection, body: Data("deny".utf8))
            return
        }
        let agent = header("x-agent-s-agent") ?? ""
        let input = body["tool_input"] as? [String: Any] ?? [:]
        let allowed = await decide(agent, tool, input)
        respond(connection, body: Data((allowed ? "allow" : "deny").utf8))
    }

    /// Stateless MCP over loopback. The Claude process can request actions only through Agent S.
    private func respondToMCP(_ request: OpenRouterProxy.Request, on connection: NWConnection,
                              agent: String, turn: String) async {
        guard request.method == "POST" else {
            respond(connection, status: "405 Method Not Allowed")
            return
        }
        guard let message = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
              message["jsonrpc"] as? String == "2.0", let method = message["method"] as? String else {
            respond(connection, status: "400 Bad Request")
            return
        }
        guard let id = message["id"] else {
            respond(connection, status: "202 Accepted")
            return
        }
        let params = message["params"] as? [String: Any] ?? [:]
        var reply: [String: Any] = ["jsonrpc": "2.0", "id": id]
        switch method {
        case "initialize":
            let supported = ["2025-11-25", "2025-06-18", "2025-03-26"]
            let requested = params["protocolVersion"] as? String ?? ""
            reply["result"] = ["protocolVersion": supported.contains(requested) ? requested : supported[0],
                               "capabilities": ["tools": [String: Bool]()],
                               "serverInfo": ["name": "Agent S", "version": "1"]]
        case "ping":
            reply["result"] = [String: String]()
        case "tools/list":
            reply["result"] = ["tools": [[
                "name": "mac_command", "description": "Ask Agent S to run a shell command on this Mac.",
                "inputSchema": ["type": "object", "properties": ["command": ["type": "string"]],
                                "required": ["command"], "additionalProperties": false],
            ]]]
        case "tools/call":
            if params["name"] as? String == "mac_command",
               let command = (params["arguments"] as? [String: Any])?["command"] as? String,
               !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !command.contains("\0") {
                let result = await runCommand(agent, turn, command)
                var content = [["type": "text", "text": result.text]]
                if let image = result.image {
                    content.append(["type": "image", "mimeType": "image/png", "data": image.base64EncodedString()])
                }
                reply["result"] = ["isError": !result.success, "content": content]
            } else {
                reply["error"] = ["code": -32602, "message": "The Mac command is missing or invalid."]
            }
        default:
            reply["error"] = ["code": -32601, "message": "Unknown method."]
        }
        if let body = try? JSONSerialization.data(withJSONObject: reply) {
            respond(connection, type: "application/json", body: body)
        } else {
            respond(connection, status: "400 Bad Request")
        }
    }

    private func respond(_ connection: NWConnection, status: String = "200 OK", type: String = "text/plain", body: Data = Data()) {
        let headers = "HTTP/1.1 \(status)\r\nContent-Type: \(type)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        connection.send(content: Data(headers.utf8) + body, completion: .contentProcessed { _ in connection.cancel() })
    }
}
