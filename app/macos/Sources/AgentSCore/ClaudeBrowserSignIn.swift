import Foundation

/// One automatic callback from the native Claude Code subscription sign-in.
public struct ClaudeBrowserSignIn: Equatable, Sendable {
    public let authorizationURL: URL
    public let callbackPort: UInt16
    public let callbackHost: String

    static let announcementPrefix = "AGENT_S_CLAUDE_BROWSER "
    private static let clientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    private static let allowedParameters: Set<String> = [
        "code", "client_id", "response_type", "redirect_uri", "scope", "code_challenge",
        "code_challenge_method", "state", "orgUUID", "login_hint", "login_method",
    ]

    public static func parseAnnouncement(_ line: String) -> Self? {
        guard line.hasPrefix(announcementPrefix) else { return nil }
        let link = String(line.dropFirst(announcementPrefix.count))
        guard link.utf8.count <= 4096,
              link.utf8.allSatisfy({ (33...126).contains($0) }),
              let parts = URLComponents(string: link),
              parts.scheme == "https", parts.user == nil, parts.password == nil,
              parts.port == nil, parts.fragment == nil,
              let host = parts.host, parts.percentEncodedHost == host,
              (host == "claude.com" && parts.percentEncodedPath == "/cai/oauth/authorize")
                || (host == "claude.ai" && parts.percentEncodedPath == "/oauth/authorize"),
              let url = parts.url, let items = parts.queryItems else { return nil }

        var parameters: [String: String] = [:]
        for item in items {
            guard allowedParameters.contains(item.name), parameters[item.name] == nil,
                  let value = item.value, !value.isEmpty else { return nil }
            parameters[item.name] = value
        }
        guard parameters["code"] == "true", parameters["client_id"] == clientID,
              parameters["response_type"] == "code", parameters["code_challenge_method"] == "S256",
              isPKCEValue(parameters["code_challenge"]), isPKCEValue(parameters["state"]),
              let redirect = parameters["redirect_uri"], let callback = URLComponents(string: redirect),
              let callbackHost = callback.host, ["localhost", "127.0.0.1"].contains(callbackHost),
              let port = callback.port.flatMap(UInt16.init(exactly:)), port >= 1024,
              redirect == "http://\(callbackHost):\(port)/callback" else { return nil }
        return Self(authorizationURL: url, callbackPort: port, callbackHost: callbackHost)
    }

    private static func isPKCEValue(_ value: String?) -> Bool {
        guard let value, value.utf8.count == 43 else { return false }
        return value.utf8.allSatisfy { byte in
            (65...90).contains(byte) || (97...122).contains(byte) || (48...57).contains(byte)
                || byte == 45 || byte == 95
        }
    }
}
