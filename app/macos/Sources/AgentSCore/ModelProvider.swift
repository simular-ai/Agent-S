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
        fitting(provider, among: allowed(in: privacy, harness: harness, computer: computer))
    }

    /// Provider choices for a new agent, before its data route is saved.
    public static func allowed(at location: AgentLocation, harness: Harness, computer: CloudComputer?) -> [Self] {
        let choices = allowed(in: location.mode(for: .ollama), harness: harness, computer: computer)
        if location == .linuxVM {
            return choices + allowed(in: .localVM, harness: harness, computer: computer)
        }
        return choices
    }

    public static func fitting(_ provider: Self, at location: AgentLocation, harness: Harness,
                               computer: CloudComputer?) -> Self? {
        fitting(provider, among: allowed(at: location, harness: harness, computer: computer))
    }

    private static func fitting(_ provider: Self, among choices: [Self]) -> Self? {
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

/// What a harness reports about the signed-in account's plan.
public enum HarnessAccountPlan: Equatable, Sendable {
    case account
    case subscription

    public init(provider: ModelProvider, reportedPlan: String?) {
        let plan = reportedPlan?.lowercased()
        switch (provider, plan) {
        case (.claudeAccount, "pro"), (.claudeAccount, "max"),
             (.codexAccount, "go"), (.codexAccount, "plus"), (.codexAccount, "pro"):
            self = .subscription
        default:
            self = .account
        }
    }
}

/// The computer the agent controls, independent of its model provider.
public enum AgentLocation: String, CaseIterable, Identifiable, Sendable {
    case linuxVM
    case thisMac
    case cloud

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .linuxVM: "Local Sandbox"
        case .thisMac: "This Mac"
        case .cloud: "Cloud"
        }
    }

    public var symbol: String {
        switch self {
        case .linuxVM: "desktopcomputer"
        case .thisMac: "laptopcomputer"
        case .cloud: "cloud"
        }
    }

    public var summary: String {
        switch self {
        case .linuxVM: "Runs in a Linux VM on this Mac."
        case .thisMac: "Works in your Mac apps, Terminal, and files. You approve actions as it works."
        case .cloud: "Works on a Sai computer or Simular Box. Your messages and what the agent reads go to the services you choose."
        }
    }

    public func mode(for provider: ModelProvider) -> PrivacyMode {
        switch self {
        case .linuxVM: provider == .openRouter ? .localVM : .fullyPrivate
        case .thisMac: .thisMac
        case .cloud: .remote
        }
    }
}

/// The saved computer and provider route. Keep the existing values for saved conversations.
public enum PrivacyMode: String, CaseIterable, Identifiable, Codable, Sendable {
    case fullyPrivate = "private"
    case localVM = "local-vm"
    case thisMac = "mac"
    case remote

    public var id: String { rawValue }

    public var location: AgentLocation {
        switch self {
        case .fullyPrivate, .localVM: .linuxVM
        case .thisMac: .thisMac
        case .remote: .cloud
        }
    }

    public var usesLocalComputer: Bool { location == .linuxVM }

    /// The providers this mode allows.
    public var providers: [ModelProvider] {
        switch self {
        case .fullyPrivate: [.ollama]
        case .localVM: [.openRouter]
        case .thisMac, .remote: ModelProvider.allCases
        }
    }
}
