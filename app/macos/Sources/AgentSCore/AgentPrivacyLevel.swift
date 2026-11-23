/// A configuration estimate based on the agent's computer, inference, and retention settings.
public struct AgentPrivacyLevel: Equatable, Sendable {
    public static let maximum = 10
    public let score: Int
    public let summary: String

    public init(location: AgentLocation, computer: CloudComputer?, provider: ModelProvider?,
                ollamaInference: AgentMemoryRoute.OllamaInference, requestsZDR: Bool) {
        let toolSummary: String = switch location {
        case .linuxVM: "Tools run in a sandbox on this Mac."
        case .thisMac: "Tools run on this Mac with approval controls."
        case .cloud: computer == .sai || provider == nil
            ? "Tools run on a Sai computer." : "Tools run on Simular Box."
        }

        let inference: (points: Int, summary: String) = switch provider {
        case .ollama:
            switch ollamaInference {
            case .local: (4, "Model inference runs on this Mac with Ollama.")
            case .hosted: (0, "Ollama forwards model requests to a hosted model.")
            case .unconfirmed: (0, "Ollama inference location is unconfirmed.")
            }
        case .openRouter:
            requestsZDR
                ? (2, "OpenRouter receives model requests. Simular requests Zero Data Retention. Safety Retention follows your OpenRouter settings.")
                : (0, "OpenRouter receives model requests. The app’s Zero Data Retention setting is off. Your OpenRouter account rules still apply.")
        case .claudeAccount: (0, "Anthropic handles inference through your Claude Code account.")
        case .codexAccount: (0, "OpenAI handles inference through your Codex account.")
        case nil: (0, "Sai manages model inference.")
        }

        score = 5 + (location == .linuxVM ? 1 : 0) + inference.points
        summary = "\(toolSummary) \(inference.summary)"
    }
}
