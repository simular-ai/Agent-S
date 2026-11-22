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
        if privacy.usesLocalComputer { return privacy.providers }
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

    /// Ollama ignores its placeholder. Live OpenRouter turns use the loopback proxy's token.
    public var placeholderKey: String {
        switch self {
        case .ollama: "ollama"
        case .openRouter: "agent-s-proxy"
        case .claudeAccount, .codexAccount: ""
        }
    }
}

/// Where an agent works and which routes it can use to reach a model.
public enum PrivacyMode: String, CaseIterable, Identifiable, Codable, Sendable {
    case fullyPrivate = "private"
    case localVM = "local-vm"
    case thisMac = "mac"
    case remote

    public var id: String { rawValue }

    public var usesLocalComputer: Bool { self == .fullyPrivate || self == .localVM }

    public var displayName: String {
        switch self {
        case .fullyPrivate: "Private"
        case .localVM: "Local VM"
        case .thisMac: "This Mac"
        case .remote: "Cloud"
        }
    }

    /// SF Symbol for the mode chip.
    public var symbol: String {
        switch self {
        case .fullyPrivate: "lock.fill"
        case .localVM: "desktopcomputer"
        case .thisMac: "laptopcomputer"
        case .remote: "cloud"
        }
    }

    public var summary: String {
        switch self {
        case .fullyPrivate: "Works in a Linux VM on this Mac and uses a local Ollama model."
        case .localVM: "Works in the same Linux VM as Private agents. OpenRouter receives your messages and content the agent reads."
        case .thisMac: "Works on this Mac itself: its apps, Terminal and files. It asks you before every command, click or change, and you can stop it any time. With a hosted model, your messages and what the agent reads go to its provider."
        case .remote: "Works on a Sai computer or Simular Box. Your messages and what the agent reads go to the services you choose."
        }
    }

    /// The providers this mode allows.
    public var providers: [ModelProvider] {
        switch self {
        case .fullyPrivate: [.ollama]
        case .localVM: [.openRouter]
        case .thisMac, .remote: ModelProvider.allCases
        }
    }
}
