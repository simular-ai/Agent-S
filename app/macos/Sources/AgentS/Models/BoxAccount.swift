import Foundation
import AgentSCore

/// The user's personal Box key lives in the Keychain and is used to authenticate to Box MCP.
enum BoxAccount {
    private static let keychain = KeychainKey(name: "Agent S · Simular Box key", legacyService: "ai.simular.saige.box")

    static var hasKey: Bool { keychain.exists }

    static func key() -> String? { keychain.read() }

    static func save(key: String) -> Bool { keychain.save(key) }

    @discardableResult
    static func delete() -> Bool { keychain.delete() }

    /// Claude Code, as installed on this Mac (the app runs it for box agents).
    static var claude: URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return ["\(home)/.local/bin/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude", "\(home)/.claude/local/claude"]
            .map(URL.init(fileURLWithPath:))
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }
}

/// A few calls to box's MCP server from the app itself (with the user's key), for things the app
/// shows rather than the agent does, like the box's live screen.
extension BoxAccount {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// The page that shows the box's screen (box.simular.ai, signed in with the user's account).
    static func viewURL(box: String?) async throws -> URL? {
        var arguments: [String: Any] = [:]
        if let box, !box.isEmpty { arguments["box"] = box }
        let status = try await callTool("box_status", arguments: arguments)
        guard let link = status["view_url"] as? String, let url = URL(string: link),
              url.scheme == "https", url.host == "box.simular.ai" || url.host?.hasSuffix(".simular.ai") == true
        else { return nil }
        return url
    }

    /// Calls `name` and returns its result's JSON text as an object.
    static func callTool(_ name: String, arguments: [String: Any]) async throws -> [String: Any] {
        let result = try await BoxMCP.shared.call(name, arguments: arguments)
        if let structured = result["structuredContent"] as? [String: Any], !structured.isEmpty { return structured }
        // The JSON may come as one text item or several, possibly with words around it.
        for text in (result["content"] as? [[String: Any]])?.compactMap({ $0["text"] as? String }) ?? [] {
            if let object = Self.jsonObject(in: text) { return object }
        }
        return [:]
    }

    /// The first JSON object in `text`.
    static func jsonObject(in text: String) -> [String: Any]? {
        if let object = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] { return object }
        guard let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}"), start < end else { return nil }
        return try? JSONSerialization.jsonObject(with: Data(text[start...end].utf8)) as? [String: Any]
    }

    /// One input on the box's screen (computer_act), at x,y in the pixels of a 1280-wide
    /// screenshot. Returns the screenshot box takes after it.
    static func act(_ action: [String: Any], box: String?) async throws -> Data? {
        var arguments = action
        arguments["max_edge"] = 1280
        if let box, !box.isEmpty { arguments["box"] = box }
        let result = try await BoxMCP.shared.call("computer_act", arguments: arguments)
        let image = (result["content"] as? [[String: Any]])?.first { $0["type"] as? String == "image" }
        return (image?["data"] as? String).flatMap { Data(base64Encoded: $0) }
    }

    /// Wakes the box (it then shows its screen).
    static func wake(box: String?) async throws {
        var arguments: [String: Any] = [:]
        if let box, !box.isEmpty { arguments["box"] = box }
        _ = try await BoxMCP.shared.call("box_wake", arguments: arguments)
    }

    /// Whether the box is awake; nil when it can't be told. Asking does not wake it.
    static func isRunning(box: String?) async throws -> Bool {
        var arguments: [String: Any] = [:]
        if let box, !box.isEmpty { arguments["box"] = box }
        let state = (try await callTool("box_status", arguments: arguments)["state"] as? String)?.lowercased()
        // Only a box that says it is asleep counts as asleep; an answer that can't be read is
        // treated as awake, so the screen is tried rather than hidden.
        return !["hibernated", "hibernating", "stopped", "sleeping", "asleep", "suspended"].contains(state ?? "")
    }

    /// The box's screen as a PNG or JPEG, read-only.
    static func screenshot(box: String?) async throws -> Data? {
        var arguments: [String: Any] = ["max_edge": 1280]
        if let box, !box.isEmpty { arguments["box"] = box }
        let result = try await BoxMCP.shared.call("screenshot", arguments: arguments)
        let image = (result["content"] as? [[String: Any]])?.first { $0["type"] as? String == "image" }
        return (image?["data"] as? String).flatMap { Data(base64Encoded: $0) }
    }

    /// One JSON-RPC message; the reply may be plain JSON or a server-sent event stream.
    static func post(key: String, session: String?, body: [String: Any]) async throws -> ([String: Any], String?) {
        var request = URLRequest(url: URL(string: Box.mcpURL)!)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("2025-06-18", forHTTPHeaderField: "MCP-Protocol-Version")
        if let session { request.setValue(session, forHTTPHeaderField: "Mcp-Session-Id") }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as? HTTPURLResponse
        let status = http?.statusCode ?? 0
        if status == 401 || status == 403 { throw Failure(message: "Simular Box rejected the key (it may be revoked).") }
        guard (200..<300).contains(status) else { throw Failure(message: "Simular Box: HTTP \(status)") }
        let session = http?.value(forHTTPHeaderField: "Mcp-Session-Id") ?? session
        let text = String(decoding: data, as: UTF8.self)
        let json = text.hasPrefix("{") ? text
            : text.split(separator: "\n").filter { $0.hasPrefix("data:") }.map { $0.dropFirst(5).trimmingCharacters(in: .whitespaces) }.last ?? ""
        return ((try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]) ?? [:], session)
    }
}

/// One MCP session with box, kept across calls (the screen view asks every few seconds).
actor BoxMCP {
    static let shared = BoxMCP()
    private var session: String?
    private var sessionKey: String?
    private var nextID = 1
    private var catalogueKey: String?
    private var agentToolScopes: [String: Bool] = [:]

    func call(_ name: String, arguments: [String: Any]) async throws -> [String: Any] {
        guard let key = BoxAccount.key() else { throw BoxAccount.Failure(message: "Add your Simular Box key first.") }
        do {
            return try await callOnce(name, arguments: arguments, key: key)
        } catch {
            // The server may have ended the session: start a new one and try once more.
            clearSession(for: key)
            return try await callOnce(name, arguments: arguments, key: key)
        }
    }

    /// Get Box's tool schemas and hide the selector, which is bound to the agent's Box by the app.
    func agentTools(box: String? = nil) async throws -> [[String: Any]] {
        guard let key = BoxAccount.key() else { throw BoxAccount.Failure(message: "Add your Simular Box key first.") }
        let definitions: [[String: Any]]
        do {
            definitions = try await listTools(key: key)
        } catch {
            clearSession(for: key)
            definitions = try await listTools(key: key)
        }
        var exposed: [[String: Any]] = []
        var scopes: [String: Bool] = [:]
        for var tool in definitions {
            guard let name = tool["name"] as? String, Box.allowsAgentTool(name), scopes[name] == nil,
                  var schema = tool["inputSchema"] as? [String: Any], schema["type"] as? String == "object" else { continue }
            var properties = schema["properties"] as? [String: Any] ?? [:]
            let acceptsBox = properties.removeValue(forKey: "box") != nil
            scopes[name] = acceptsBox
            guard Box.scopedAgentArguments([:], acceptsBox: acceptsBox, box: box) != nil else { continue }
            let required = schema["required"] as? [String] ?? []
            if required.contains("box"), box?.isEmpty != false { continue }
            schema["properties"] = properties
            schema["required"] = required.filter { $0 != "box" }
            tool["inputSchema"] = schema
            exposed.append(tool)
        }
        catalogueKey = key
        agentToolScopes = scopes
        guard !exposed.isEmpty else {
            throw BoxAccount.Failure(message: "Simular Box returned no tools for this computer.")
        }
        return exposed
    }

    /// Run an agent action once. A timeout may happen after the Box acted, so it cannot be replayed.
    func callAgentTool(_ name: String, arguments: [String: Any], box: String?) async throws -> [String: Any] {
        try Task.checkCancellation()
        guard let key = BoxAccount.key(), catalogueKey == key else {
            throw BoxAccount.Failure(message: "The Box connection changed. Start another turn to reconnect.")
        }
        guard Box.allowsAgentTool(name), let acceptsBox = agentToolScopes[name],
              let scoped = Box.scopedAgentArguments(arguments, acceptsBox: acceptsBox, box: box) else {
            throw BoxAccount.Failure(message: "That tool isn't available for this Simular Box.")
        }
        do {
            return try await callOnce(name, arguments: scoped, key: key)
        } catch {
            clearSession(for: key)
            throw error
        }
    }

    private func listTools(key: String) async throws -> [[String: Any]] {
        var definitions: [[String: Any]] = []
        var seen = Set<String>()
        var cursor: String?
        repeat {
            let params = cursor.map { ["cursor": $0] } ?? [:]
            let result = try await request("tools/list", params: params, key: key)
            definitions += result["tools"] as? [[String: Any]] ?? []
            cursor = result["nextCursor"] as? String
        } while cursor.map({ seen.insert($0).inserted }) == true
        return definitions
    }

    private func clearSession(for key: String) {
        guard sessionKey == key else { return }
        session = nil
        sessionKey = nil
    }

    private func request(_ method: String, params: [String: Any], key: String) async throws -> [String: Any] {
        let requestSession: String?
        if sessionKey != key {
            let (initialized, id) = try await BoxAccount.post(key: key, session: nil, body: [
                "jsonrpc": "2.0", "id": take(), "method": "initialize",
                "params": ["protocolVersion": "2025-06-18", "capabilities": [:] as [String: Any],
                           "clientInfo": ["name": "Agent S", "version": "1"]],
            ])
            _ = try Self.result(initialized)
            _ = try? await BoxAccount.post(key: key, session: id, body: ["jsonrpc": "2.0", "method": "notifications/initialized"])
            session = id
            sessionKey = key
            requestSession = id
        } else {
            requestSession = session
        }
        try Task.checkCancellation()
        let (reply, _) = try await BoxAccount.post(key: key, session: requestSession, body: [
            "jsonrpc": "2.0", "id": take(), "method": method, "params": params,
        ])
        return try Self.result(reply)
    }

    private static func result(_ reply: [String: Any]) throws -> [String: Any] {
        if let error = reply["error"] as? [String: Any] {
            throw BoxAccount.Failure(message: "Simular Box: \(error["message"] as? String ?? "error")")
        }
        guard let result = reply["result"] as? [String: Any] else {
            throw BoxAccount.Failure(message: "Simular Box returned an invalid response.")
        }
        return result
    }

    private func callOnce(_ name: String, arguments: [String: Any], key: String) async throws -> [String: Any] {
        let result = try await request("tools/call", params: ["name": name, "arguments": arguments], key: key)
        if result["isError"] as? Bool == true {
            let text = (result["content"] as? [[String: Any]])?.compactMap { $0["text"] as? String }.joined() ?? ""
            throw BoxAccount.Failure(message: "Simular Box: \(text)")
        }
        return result
    }

    private func take() -> Int { defer { nextID += 1 }; return nextID }
}
