import Foundation

/// How an agent reaches its model. Account providers let the official harness manage sign-in
/// and model selection on the Mac.
public enum ModelProvider: String, CaseIterable, Identifiable, Codable, Sendable {
    case ollama
    case openRouter = "openrouter"
    case claudeAccount = "claude-account"
    case codexAccount = "codex-account"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .ollama: "Ollama (on this Mac)"
        case .openRouter: "OpenRouter"
        case .claudeAccount: "Use Claude Code account"
        case .codexAccount: "Use Codex account"
        }
    }

    public var usesAccount: Bool { self == .claudeAccount || self == .codexAccount }

    /// The available routes depend on both where tools run and which harness controls them.
    public static func allowed(in privacy: PrivacyMode, harness: Harness, computer: CloudComputer?) -> [Self] {
        guard Harness.allowed(in: privacy, computer: computer).contains(harness) else { return [] }
        if privacy == .fullyPrivate { return [.ollama] }
        switch harness {
        case .claudeCode: return [.ollama, .openRouter, .claudeAccount]
        case .codex: return privacy == .thisMac ? [.codexAccount] : [.ollama, .openRouter, .codexAccount]
        case .deepSeek: return [.ollama, .openRouter]
        case .sai: return []
        }
    }

    public static func fitting(_ provider: Self, in privacy: PrivacyMode, harness: Harness,
                               computer: CloudComputer?) -> Self? {
        let choices = allowed(in: privacy, harness: harness, computer: computer)
        if choices.contains(provider) { return provider }
        if provider.usesAccount, let account = choices.first(where: \.usesAccount) { return account }
        return choices.first
    }

    /// Context Claude Code may fill: small for a local model's window, large for hosted models.
    public var contextTokens: Int {
        switch self {
        case .ollama: 32_768
        case .openRouter, .claudeAccount, .codexAccount: 200_000
        }
    }

    /// The longest answer a turn may write. Claude Code keeps this much free in the window, so a
    /// local model needs it small (otherwise it compacts every few steps).
    public var maxOutputTokens: Int {
        switch self {
        case .ollama: 4096
        case .openRouter, .claudeAccount, .codexAccount: 16_384
        }
    }

    /// Sent as the API key from the guest. Not a secret: Ollama ignores it, and the Mac's
    /// OpenRouter proxy replaces it with the real key.
    public var placeholderKey: String {
        switch self {
        case .ollama: "ollama"
        case .openRouter: "agent-s-proxy"
        case .claudeAccount, .codexAccount: ""
        }
    }
}

/// Where an agent works, and how much may leave the Mac. Private: a Linux computer that runs on
/// this Mac, with the local model only. This Mac: the Mac itself (its apps, shell and files),
/// every action approved by the person first. Cloud: hosted models and cloud computers, with the
/// user's own keys.
public enum PrivacyMode: String, CaseIterable, Identifiable, Codable, Sendable {
    case fullyPrivate = "private"
    case thisMac = "mac"
    case remote

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .fullyPrivate: "Private"
        case .thisMac: "This Mac"
        case .remote: "Cloud"
        }
    }

    /// SF Symbol for the mode chip.
    public var symbol: String {
        switch self {
        case .fullyPrivate: "lock.fill"
        case .thisMac: "laptopcomputer"
        case .remote: "cloud"
        }
    }

    public var summary: String {
        switch self {
        case .fullyPrivate: "Works on a private Linux computer that runs on this Mac, with a model on this Mac. It can't touch your Mac's own files or apps. Only the web pages it visits leave this Mac."
        case .thisMac: "Works on this Mac itself: its apps, Terminal and files. It asks you before every command, click or change, and you can stop it any time. With a hosted model, your messages and what the agent reads go to its provider."
        case .remote: "Works on a Sai computer or Simular Box. Your messages and what the agent reads go to the services you choose."
        }
    }

    /// The providers this mode allows.
    public var providers: [ModelProvider] {
        switch self {
        case .fullyPrivate: [.ollama]
        case .thisMac, .remote: ModelProvider.allCases
        }
    }
}
