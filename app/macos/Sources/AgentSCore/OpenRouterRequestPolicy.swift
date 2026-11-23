import Foundation

public enum OpenRouterRequestPolicy {
    public enum Rejection: Error, Equatable {
        case unsupportedRoute
        case invalidJSONBody
        case invalidProviderPreferences
    }

    private static let inferencePaths: Set<String> = [
        "/api/v1/messages",
        "/api/v1/responses",
        "/api/v1/chat/completions",
    ]

    /// `path` is the validated upstream URL's path, independent of query parameters.
    public static func body(_ body: Data, method: String, path: String, requiresZDR: Bool) throws -> Data {
        guard requiresZDR else { return body }
        if (method == "GET" || method == "HEAD") && body.isEmpty { return body }
        guard method == "POST", inferencePaths.contains(path) else { throw Rejection.unsupportedRoute }
        guard var payload = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            throw Rejection.invalidJSONBody
        }

        var provider: [String: Any] = [:]
        if let value = payload["provider"], !(value is NSNull) {
            guard let preferences = value as? [String: Any] else { throw Rejection.invalidProviderPreferences }
            provider = preferences
        }
        provider["zdr"] = true
        payload["provider"] = provider
        return try JSONSerialization.data(withJSONObject: payload)
    }
}
