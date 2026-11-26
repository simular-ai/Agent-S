import Foundation

/// The Sai API (platform.simular.ai), bring-your-own-key: Simular's Sai agent runs a task on one
/// of the user's Sai computers (a Simular cloud computer, or a Mac/PC enrolled with the Sai
/// runtime). Agent S calls it from the Mac; the key lives in the Keychain and never enters the
/// local computer.
enum SaiAPI {
    static let base = URL(string: "https://api.simular.ai")!
    /// The picker's entry for Agent S’s own computer, until the API can drive it.
    static let localMachine = "saige-local"

    enum ScreenState: Equatable {
        case unknown, asleep, starting, running, offline
    }

    struct Machine: Identifiable, Hashable {
        let id: String
        let name: String
        let kind: String
        let os: String?
        let online: Bool
        let status: String?
        let canWake: Bool

        var screenState: ScreenState {
            if kind == "own" { return online ? .running : .offline }
            // Sai reports a hibernated cloud computer as online because it can receive a task.
            switch status?.lowercased() {
            case "hibernated", "hibernating": return .asleep
            case "provisioning", "wakingup": return .starting
            case "active": return .running
            default: return online ? .running : .offline
            }
        }
    }

    struct Model: Identifiable, Hashable {
        enum CostTier: String {
            case free
            case low = "$"
            case medium = "$$"
            case high = "$$$"

            var label: String { self == .free ? "Free" : rawValue }
        }

        let id: String
        let name: String
        let allowed: Bool
        let costTier: CostTier?

        func optionTitle(recommended: Bool = false) -> String {
            let title = recommended ? "\(name) (recommended)" : name
            return "\(title) · \(costTier?.label ?? "Cost unavailable")"
        }
    }

    struct Failure: LocalizedError {
        let message: String
        let isMachineNotRunning: Bool
        var errorDescription: String? { message }

        init(message: String, isMachineNotRunning: Bool = false) {
            self.message = message
            self.isMachineNotRunning = isMachineNotRunning
        }
    }

    // MARK: Keychain

    private static let keychain = KeychainKey(name: "Agent S · Sai API key", legacyService: "ai.simular.saige.sai")

    static var hasKey: Bool { keychain.exists }

    static func key() -> String? { keychain.read() }

    static func save(key: String) -> Bool { keychain.save(key) }

    @discardableResult
    static func delete() -> Bool { keychain.delete() }

    // MARK: Calls

    private static func request(_ path: String, method: String = "GET", body: [String: Any]? = nil,
                                accept: String = "application/json") throws -> URLRequest {
        guard let key = key() else { throw Failure(message: "Add your Sai API key in Settings.") }
        var request = URLRequest(url: URL(string: base.absoluteString + path)!)
        request.httpMethod = method
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.setValue(accept, forHTTPHeaderField: "Accept")
        request.setValue("Agent S (macOS)", forHTTPHeaderField: "User-Agent")
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        request.timeoutInterval = 30
        return request
    }

    private static func json(_ path: String, method: String = "GET", body: [String: Any]? = nil,
                             timeout: TimeInterval = 30) async throws -> [String: Any] {
        var request = try request(path, method: method, body: body)
        request.timeoutInterval = timeout
        let (data, response) = try await URLSession.shared.data(for: request)
        let object = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw failure(status: status, body: object) }
        return object
    }

    static func failure(status: Int, body: [String: Any]) -> Failure {
        let code = (body["error"] as? String) ?? (body["code"] as? String)
        let detail = (body["message"] as? String) ?? (body["error"] as? String) ?? "HTTP \(status)"
        let isMachineNotRunning = status == 409 && (code == "machine_not_running" ||
            detail.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().hasPrefix("the computer is not running right now"))
        return Failure(message: status == 401 ? "Sai rejected the API key (it may be revoked)." : "Sai: \(detail)",
                       isMachineNotRunning: isMachineNotRunning)
    }

    static func machines() async throws -> [Machine] {
        let list = try await json("/v1/agents/machines")["machines"] as? [[String: Any]] ?? []
        return list.compactMap { row in
            guard let id = row["machineId"] as? String else { return nil }
            return Machine(id: id, name: row["name"] as? String ?? id, kind: row["kind"] as? String ?? "cloud",
                           os: row["os"] as? String, online: row["online"] as? Bool ?? false,
                           status: row["status"] as? String, canWake: row["canWake"] as? Bool ?? false)
        }
    }

    static func models() async throws -> [Model] {
        let list = try await json("/v1/agents/models")["models"] as? [[String: Any]] ?? []
        return list.compactMap { row in
            guard let id = row["id"] as? String else { return nil }
            let costTier = (row["costTier"] as? String).flatMap(Model.CostTier.init(rawValue:))
            return Model(id: id, name: row["name"] as? String ?? id, allowed: row["allowed"] as? Bool ?? true,
                         costTier: costTier)
        }
    }

    /// Starts a wakeable cloud computer. A successful request can still leave it starting up.
    static func wake(machine: String) async throws {
        let response = try await json("/v1/agents/wake", method: "POST", body: ["machineId": machine])
        let status = (response["status"] as? String)?.lowercased()
        // The reported status is from before the request. An active computer is already awake.
        if response["startingUp"] as? Bool == true || response["waking"] as? Bool == true ||
            status == "active" || status == "wakingup" { return }
        if response["canWake"] as? Bool == false {
            throw Failure(message: "Sai can't wake this computer right now.")
        }
        throw Failure(message: "Sai didn't confirm that the computer is starting.")
    }

    /// A page showing the machine's screen, for a person. It carries a session token: it goes to
    /// the web view only, never to a log or the model. A link stays good for about 30 minutes and
    /// can open many screen sessions, while Sai limits how many links an hour a user may make, so
    /// one is kept per machine and reused until it is about to expire.
    static func liveURL(machine: String, fresh: Bool = false, keyRevision: Int) async throws -> URL? {
        let cacheKey = "\(keyRevision):\(machine)"
        if !fresh, let link = await LiveLinks.shared.link(for: cacheKey) { return link }
        let object = try await json("/v1/agents/machines/\(machine.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? machine)/live", method: "POST")
        guard let url = (object["liveUrl"] as? String).flatMap(URL.init(string:)), url.scheme == "https" else { return nil }
        let expires = (object["expiresAt"] as? Double).map { Date(timeIntervalSince1970: $0 / 1000) }
            ?? Date().addingTimeInterval(25 * 60)
        await LiveLinks.shared.keep(url, until: expires, for: cacheKey)
        return url
    }

    static func approve(_ approvalId: String, response: String) async throws {
        _ = try await json("/v1/agents/approve", method: "POST", body: ["approvalId": approvalId, "response": response])
    }

    static func abort(session: String) async {
        _ = try? await requestAbort(session: session)
    }

    /// Wait until Sai reports that this session has no running task.
    static func abortConfirmed(session: String) async throws {
        guard !session.isEmpty else { throw Failure(message: "Sai hasn’t reported a session to stop yet.") }
        let reply = try await requestAbort(session: session, timeout: 5)
        let alreadyStopping = ["no active session", "session already idle or aborting"]
        guard reply["ok"] as? Bool == true, let aborted = reply["aborted"] as? Bool,
              aborted || alreadyStopping.contains(reply["reason"] as? String ?? "") else {
            throw Failure(message: "Sai didn’t acknowledge the stop request.")
        }

        // The abort response may say that the task is already aborting. A terminal session
        // status is needed before a new message can be sent without joining the old task.
        let deadline = ContinuousClock.now.advanced(by: .seconds(12))
        let unconfirmed = Failure(message: "Sai didn’t confirm that the task stopped.")
        var cursor: String?
        repeat {
            try Task.checkCancellation()
            var query = [URLQueryItem(name: "sessionId", value: session)]
            if let cursor { query.append(URLQueryItem(name: "since", value: cursor)) }
            var components = URLComponents()
            components.queryItems = query
            guard let encoded = components.percentEncodedQuery else { throw unconfirmed }
            let events = try await json("/v1/agents/events?\(encoded)", timeout: 3)
            guard events["sessionId"] as? String == session else { throw unconfirmed }
            switch events["status"] as? String {
            case "idle", "error": return
            case "running", "needs_approval": cursor = events["cursor"] as? String
            default: throw unconfirmed
            }
            try await Task.sleep(for: .milliseconds(500))
        } while ContinuousClock.now < deadline
        throw unconfirmed
    }

    private static func requestAbort(session: String, timeout: TimeInterval = 30) async throws -> [String: Any] {
        try await json("/v1/agents/abort", method: "POST", body: ["sessionId": session], timeout: timeout)
    }

    /// Runs one message as a task and hands each stream event (a JSON object) to `event`.
    static func run(message: String, machine: String, session: String?, model: String?,
                    event: @escaping @MainActor ([String: Any]) -> Void) async throws {
        var body: [String: Any] = ["message": message, "machineId": machine]
        if let session { body["sessionId"] = session }
        // Sai retains the previous session model unless we send the new choice, including "auto".
        if let model { body["model"] = model }
        var request = try request("/v1/agents/message", method: "POST", body: body, accept: "text/event-stream")
        request.timeoutInterval = 60 * 60
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            var data = Data()
            for try await byte in bytes { data.append(byte) }
            throw failure(status: status, body: (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:])
        }
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let object = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] else { continue }
            await event(object)
        }
    }

    /// The link a person opens to answer a link-only approval (a sign-in, a phone check).
    static func approvalURL(_ approval: [String: Any]) -> URL? {
        // Only web links: the stream carries text a page or the model can influence.
        (approval["approvalUrl"] as? String).flatMap(URL.init(string:)).flatMap { $0.scheme?.lowercased() == "https" ? $0 : nil }
    }
}

/// The live links made this launch, by key revision and machine.
actor LiveLinks {
    static let shared = LiveLinks()
    private var links: [String: (url: URL, expires: Date)] = [:]

    /// A kept link with at least two minutes left.
    func link(for key: String) -> URL? {
        guard let kept = links[key], kept.expires.timeIntervalSinceNow > 120 else { return nil }
        return kept.url
    }

    func keep(_ url: URL, until expires: Date, for key: String) {
        links[key] = (url, expires)
    }
}
