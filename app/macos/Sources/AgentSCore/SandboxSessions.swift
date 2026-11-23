/// Each inference route sees only the harness sessions it created inside the shared sandbox.
public enum SandboxSessions {
    public static let prefix = "sandbox/"

    public static func key(harness: Harness, provider: ModelProvider) -> String {
        "\(prefix)\(harness.rawValue)/\(provider.rawValue)"
    }

    public static func migrating(_ sessions: [String: String], from mode: PrivacyMode) -> [String: String] {
        guard mode.usesLocalComputer else { return sessions }
        // Older builds allowed exactly one provider in each saved sandbox mode.
        let legacyProvider: ModelProvider = mode == .fullyPrivate ? .ollama : .openRouter
        var migrated = sessions
        for harness in Harness.allowed(in: mode) {
            guard let session = migrated.removeValue(forKey: harness.rawValue) else { continue }
            let key = key(harness: harness, provider: legacyProvider)
            if migrated[key] == nil { migrated[key] = session }
        }
        return migrated
    }
}
