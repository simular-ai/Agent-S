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

        func matches(_ model: String) -> Bool {
            Ollama.normalizedName(name) == Ollama.normalizedName(model)
        }
    }

    static func normalizedName(_ model: String) -> String {
        model.hasSuffix(":latest") ? String(model.dropLast(7)) : model
    }

    static func isCloudReference(_ model: String) -> Bool {
        guard let colon = model.lastIndex(of: ":") else { return false }
        let tag = model[model.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !tag.contains("/") && (tag == "cloud" || tag.hasSuffix("-cloud"))
    }

    /// Ollama's version, or nil when it is not running (or older than its Anthropic endpoint).
    static func version(timeout: TimeInterval = 10) async -> String? {
        var request = URLRequest(url: base.appendingPathComponent("api/version"))
        request.timeoutInterval = timeout
        guard let (data, _) = try? await URLSession.shared.data(for: request),
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

    /// Pulls a model, reporting Ollama's status and the fraction of reported bytes downloaded.
    static func pull(_ name: String, update: @escaping @MainActor (String, Double?) -> Void) async throws {
        var request = URLRequest(url: base.appendingPathComponent("api/pull"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": name, "stream": true])
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let response = response as? HTTPURLResponse else { throw PullFailure.incomplete }
        guard (200..<300).contains(response.statusCode) else { throw PullFailure.httpStatus(response.statusCode) }
        var progress = PullProgress()
        for try await line in bytes.lines {
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let error = json["error"] as? String { throw PullFailure.server(error) }
            guard let status = json["status"] as? String else { continue }
            let total = (json["total"] as? NSNumber)?.doubleValue
            let completed = (json["completed"] as? NSNumber)?.doubleValue
            let fraction = progress.update(status: status, digest: json["digest"] as? String,
                                           total: total, completed: completed)
            await update(status, fraction)
            if status == "success" { return }
        }
        throw PullFailure.incomplete
    }

    private struct PullProgress {
        private struct Layer {
            let total: Double
            let completed: Double
        }

        private var layers: [String?: Layer] = [:]

        mutating func update(status: String, digest: String?, total: Double?, completed: Double?) -> Double? {
            switch status {
            case "verifying sha256 digest", "writing manifest", "removing any unused layers", "success":
                return 1
            default:
                break
            }

            // Ollama announces each layer separately and omits `completed` when it is zero.
            if let total, total.isFinite, total > 0 {
                let completed = completed ?? 0
                if completed.isFinite {
                    layers[digest] = Layer(total: total, completed: min(max(completed, 0), total))
                }
            }

            let totalBytes = layers.values.reduce(0) { $0 + $1.total }
            guard totalBytes > 0 else { return nil }
            let completedBytes = layers.values.reduce(0) { $0 + $1.completed }
            return completedBytes / totalBytes
        }
    }

    private enum PullFailure: LocalizedError {
        case httpStatus(Int)
        case server(String)
        case incomplete

        var errorDescription: String? {
            switch self {
            case let .httpStatus(status): "Ollama returned HTTP \(status)."
            case let .server(message): message
            case .incomplete: "Ollama didn’t finish the download."
            }
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
