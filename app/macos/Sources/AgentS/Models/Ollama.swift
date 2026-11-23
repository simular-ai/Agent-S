import Foundation

/// The host's Ollama: whether it is up, which models it has, pulling the ones the app needs.
struct Ollama {
    static let base = URL(string: "http://127.0.0.1:11434")!
    static let port: UInt16 = 11434

    struct Model: Identifiable, Hashable {
        let name: String
        let size: Int64
        let usesRemoteInference: Bool
        var id: String { name }

        func matches(_ model: String) -> Bool { name == model || name == model + ":latest" }
    }

    static func isCloudReference(_ model: String) -> Bool {
        guard let colon = model.lastIndex(of: ":") else { return false }
        let tag = model[model.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !tag.contains("/") && (tag == "cloud" || tag.hasSuffix("-cloud"))
    }

    /// Ollama's version, or nil when it is not running (or older than its Anthropic endpoint).
    static func version() async -> String? {
        guard let (data, _) = try? await URLSession.shared.data(from: base.appendingPathComponent("api/version")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json["version"] as? String
    }

    static func models(timeout: TimeInterval = 60) async -> [Model] {
        var request = URLRequest(url: base.appendingPathComponent("api/tags"))
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return [] }
        return parseModels(data)
    }

    static func parseModels(_ data: Data) -> [Model] {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = json["models"] as? [[String: Any]] else { return [] }
        return list.compactMap { item in
            guard let name = item["name"] as? String else { return nil }
            let isRemote = ["remote_model", "remote_host"].contains { (item[$0] as? String)?.isEmpty == false }
            return Model(name: name, size: (item["size"] as? NSNumber)?.int64Value ?? 0, usesRemoteInference: isRemote)
        }
    }

    /// Pulls a model, reporting Ollama's status line and progress (0…1) as it streams.
    static func pull(_ name: String, update: @escaping @MainActor (String, Double?) -> Void) async throws {
        var request = URLRequest(url: base.appendingPathComponent("api/pull"))
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": name, "stream": true])
        let (bytes, _) = try await URLSession.shared.bytes(for: request)
        for try await line in bytes.lines {
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let error = json["error"] as? String { throw GuestImage.Failure(message: error) }
            let status = json["status"] as? String ?? ""
            let total = (json["total"] as? NSNumber)?.doubleValue
            let done = (json["completed"] as? NSNumber)?.doubleValue
            let fraction = (total != nil && done != nil && total! > 0) ? done! / total! : nil
            await update(status, fraction)
        }
    }

    /// Ollama's version string compared as dotted numbers; true when at least `minimum`.
    static func isAtLeast(_ version: String, _ minimum: String) -> Bool {
        let a = version.split(separator: ".").map { Int($0.prefix { $0.isNumber }) ?? 0 }
        let b = minimum.split(separator: ".").map { Int($0) ?? 0 }
        for i in 0..<max(a.count, b.count) {
            let x = i < a.count ? a[i] : 0, y = i < b.count ? b[i] : 0
            if x != y { return x > y }
        }
        return true
    }
}
