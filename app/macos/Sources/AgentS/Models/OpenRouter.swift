import Foundation
import Network
import AgentSCore

/// OpenRouter, bring-your-own-key. The key lives in the Mac's Keychain and is added to requests by
/// a proxy on the Mac's loopback; the computer reaches the proxy through the turn's SSH tunnel and
/// never holds the key, so the agent (or a page that tricks it) cannot read or leak it.
enum OpenRouter {
    static let api = URL(string: "https://openrouter.ai/api")!
    /// Where the proxy listens on the Mac (loopback only).
    static let proxyPort: UInt16 = 11435

    /// A few good starting points; any model id from openrouter.ai/models works.
    static let suggested = [
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-opus-5.5",
        "qwen/qwen3-coder",
        "openai/gpt-oss-120b",
        "google/gemma-4-31b-it",
    ]

    struct Model: Identifiable, Hashable, Sendable {
        let id: String
        let name: String
        let contextLength: Int?
        let inputCapability: ModelInputCapability?
    }

    /// OpenRouter returns the full text-output model list when pagination is omitted.
    static func models() async -> [Model]? {
        guard let data = await publicCatalogData(at: "v1/models"),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = object["data"] as? [[String: Any]] else { return nil }
        return list.compactMap { item in
            guard let id = item["id"] as? String else { return nil }
            let architecture = item["architecture"] as? [String: Any]
            let capability = ModelInputCapability.fromInputModalities(architecture?["input_modalities"] as? [String])
            return Model(id: id, name: item["name"] as? String ?? id,
                         contextLength: item["context_length"] as? Int, inputCapability: capability)
        }
    }

    static func zdrCatalog() async -> OpenRouterZDRCatalog? {
        guard let data = await publicCatalogData(at: "v1/endpoints/zdr") else { return nil }
        return try? OpenRouterZDRCatalog(data: data)
    }

    private static func publicCatalogData(at path: String) async -> Data? {
        let request = URLRequest(url: api.appendingPathComponent(path), cachePolicy: .reloadIgnoringLocalCacheData,
                                 timeoutInterval: 15)
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return data
    }

    // MARK: Keychain

    private static let keychain = KeychainKey(name: "Agent S · OpenRouter key", legacyService: "ai.simular.saige.openrouter")

    static var hasKey: Bool { keychain.exists }

    static func key() -> String? { keychain.read() }

    static func save(key: String) -> Bool { keychain.save(key) }

    @discardableResult
    static func delete() -> Bool { keychain.delete() }
}

/// A small HTTP proxy on 127.0.0.1:`OpenRouter.proxyPort`: one request per connection, forwarded
/// to https://openrouter.ai/api with the key from the Keychain, the response streamed back (so
/// server-sent events arrive as they are produced). Whatever key the client sent is dropped.
final class OpenRouterProxy: @unchecked Sendable {
    /// Required on every request (as the API key), so other apps on the Mac, or web pages
    /// calling localhost, cannot spend the user's key. New on every launch; only Agent S’s own
    /// turns in the computer are given it.
    let token = "agent-s-" + UUID().uuidString
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "ai.simular.agent-s.openrouter-proxy")
    private let requiresZDR: @Sendable () -> Bool

    init(requiresZDR: @escaping @Sendable () -> Bool) {
        self.requiresZDR = requiresZDR
    }

    var isRunning: Bool { listener != nil }

    func start() throws {
        guard listener == nil else { return }
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: OpenRouter.proxyPort)!)
        parameters.allowLocalEndpointReuse = true
        let listener = try NWListener(using: parameters)
        listener.newConnectionHandler = { [weak self] connection in self?.serve(connection) }
        listener.stateUpdateHandler = { state in
            if case let .failed(error) = state { NSLog("Agent S: OpenRouter proxy failed: %@", "\(error)") }
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: One request

    private func serve(_ connection: NWConnection) {
        connection.start(queue: queue)
        readRequest(connection, buffer: Data())
    }

    /// Reads until the headers and the whole body (by Content-Length) have arrived.
    private func readRequest(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let request = Self.parse(buffer) {
                Task { await self.forward(request, to: connection) }
            } else if isComplete || error != nil || buffer.count > 32 << 20 {
                connection.cancel()
            } else {
                self.readRequest(connection, buffer: buffer)
            }
        }
    }

    struct Request {
        let method: String
        let path: String
        let headers: [(String, String)]
        let body: Data
    }

    static func parse(_ data: Data) -> Request? {
        guard let end = data.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let head = String(decoding: data[data.startIndex..<end.lowerBound], as: UTF8.self)
        var lines = head.components(separatedBy: "\r\n")
        let start = lines.removeFirst().split(separator: " ")
        guard start.count >= 2 else { return nil }
        let headers: [(String, String)] = lines.compactMap { line in
            guard let colon = line.firstIndex(of: ":") else { return nil }
            return (String(line[..<colon]), line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces))
        }
        let length = headers.first { $0.0.lowercased() == "content-length" }.flatMap { Int($0.1) } ?? 0
        let body = data[end.upperBound...]
        guard length >= 0, body.count >= length else { return nil }
        return Request(method: String(start[0]), path: String(start[1]), headers: headers, body: Data(body.prefix(length)))
    }

    private func forward(_ request: Request, to connection: NWConnection) async {
        let presented = request.headers.first { ["authorization", "x-api-key"].contains($0.0.lowercased()) }?.1
            .replacingOccurrences(of: "Bearer ", with: "")
        guard presented == token else {
            respond(connection, status: 403, body: #"{"error":{"message":"Forbidden"}}"#)
            return
        }
        guard let key = OpenRouter.key() else {
            respond(connection, status: 401, body: #"{"error":{"message":"Add your OpenRouter key in Agent S’s Settings."}}"#)
            return
        }
        // Only API paths under /v1; nothing else on openrouter.ai is reachable through the proxy.
        guard request.path.hasPrefix("/v1/"), !request.path.contains(".."), !request.path.contains("%2e"), !request.path.contains("%2E"),
              let url = URL(string: OpenRouter.api.absoluteString + request.path) else {
            respond(connection, status: 404, body: #"{"error":{"message":"Not found"}}"#)
            return
        }
        let enforceZDR = requiresZDR()
        let body: Data
        do {
            body = try OpenRouterRequestPolicy.body(request.body, method: request.method, path: url.path,
                                                    requiresZDR: enforceZDR)
        } catch {
            respond(connection, status: 400, body: #"{"error":{"message":"This request could not be routed with Agent S’s OpenRouter retention setting."}}"#)
            return
        }
        var upstream = URLRequest(url: url)
        upstream.httpMethod = request.method
        upstream.timeoutInterval = 600
        let dropped: Set<String> = ["host", "authorization", "x-api-key", "content-length", "connection",
                                    "accept-encoding", "transfer-encoding", "keep-alive"]
        for (name, value) in request.headers where !dropped.contains(name.lowercased()) {
            upstream.addValue(value, forHTTPHeaderField: name)
        }
        upstream.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        upstream.setValue("Agent S", forHTTPHeaderField: "X-Title")
        if !body.isEmpty {
            if enforceZDR { upstream.setValue("application/json", forHTTPHeaderField: "Content-Type") }
            upstream.httpBody = body
        }

        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: upstream)
            let http = response as? HTTPURLResponse
            var head = "HTTP/1.1 \(http?.statusCode ?? 502) \(HTTPURLResponse.localizedString(forStatusCode: http?.statusCode ?? 502))\r\n"
            for (name, value) in http?.allHeaderFields ?? [:] {
                let name = "\(name)"
                if ["content-length", "transfer-encoding", "connection", "content-encoding"].contains(name.lowercased()) { continue }
                head += "\(name): \(value)\r\n"
            }
            head += "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            send(connection, Data(head.utf8))
            var chunk = Data()
            for try await byte in bytes {
                chunk.append(byte)
                // Flush at line ends so streamed events reach the agent as they arrive.
                if byte == 0x0A || chunk.count >= 16 << 10 {
                    send(connection, Self.chunked(chunk))
                    chunk.removeAll(keepingCapacity: true)
                }
            }
            if !chunk.isEmpty { send(connection, Self.chunked(chunk)) }
            send(connection, Data("0\r\n\r\n".utf8), last: true)
        } catch {
            respond(connection, status: 502, body: #"{"error":{"message":"Could not reach OpenRouter."}}"#)
        }
    }

    static func chunked(_ data: Data) -> Data {
        Data(String(data.count, radix: 16).utf8) + Data("\r\n".utf8) + data + Data("\r\n".utf8)
    }

    private func respond(_ connection: NWConnection, status: Int, body: String) {
        let head = "HTTP/1.1 \(status) \(HTTPURLResponse.localizedString(forStatusCode: status))\r\n"
            + "Content-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n"
        send(connection, Data((head + body).utf8), last: true)
    }

    private func send(_ connection: NWConnection, _ data: Data, last: Bool = false) {
        connection.send(content: data, completion: .contentProcessed { _ in if last { connection.cancel() } })
    }
}
