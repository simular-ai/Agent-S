public extension AgentMemoryRoute {
    /// Both endpoints must have a known inference location before transferring a handoff.
    var canRouteHandoff: Bool { allowsAutomaticRecall }

    /// Even agents on the same computer may send prompts to different model providers.
    func requiresHandoffApproval(to destination: AgentMemoryRoute) -> Bool {
        // Ollama controls the routing of model tags outside the app. Review even a currently local tag.
        destination.provider == .ollama || privacy != .remote || self != destination
    }
}
