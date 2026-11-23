import Testing
import AgentSCore

@Suite("Agent privacy levels")
struct AgentPrivacyLevelTests {
    @Test(arguments: [(AgentLocation.linuxVM, 10), (.thisMac, 9), (.cloud, 9)])
    func scoresLocalInferenceAndSandboxIsolation(location: AgentLocation, score: Int) {
        let level = AgentPrivacyLevel(location: location, computer: location == .cloud ? .box : nil,
                                      provider: .ollama, ollamaInference: .local, requestsZDR: true)
        #expect(level.score == score)
        #expect(level.summary.contains("Model inference runs on this Mac"))
        if location == .linuxVM { #expect(level.score == AgentPrivacyLevel.maximum) }
    }

    @Test(arguments: [(AgentLocation.linuxVM, 6, 8), (.thisMac, 5, 7), (.cloud, 5, 7)])
    func scoresTheOpenRouterRequestPreference(location: AgentLocation, offScore: Int, onScore: Int) {
        let computer: CloudComputer? = location == .cloud ? .box : nil
        let off = AgentPrivacyLevel(location: location, computer: computer, provider: .openRouter,
                                    ollamaInference: .local, requestsZDR: false)
        let on = AgentPrivacyLevel(location: location, computer: computer, provider: .openRouter,
                                   ollamaInference: .local, requestsZDR: true)
        #expect(off.score == offScore)
        #expect(on.score == onScore)
        #expect(off.summary.contains("Your OpenRouter account rules still apply."))
        #expect(on.summary.contains("Safety Retention follows your OpenRouter settings."))
        if location == .cloud { #expect(off.summary.contains("Simular Box")) }
    }

    @Test func hostedAndUnconfirmedOllamaEarnNoLocalInferencePoints() {
        let hosted = AgentPrivacyLevel(location: .linuxVM, computer: nil, provider: .ollama,
                                       ollamaInference: .hosted, requestsZDR: true)
        let unconfirmed = AgentPrivacyLevel(location: .linuxVM, computer: nil, provider: .ollama,
                                            ollamaInference: .unconfirmed, requestsZDR: true)
        #expect(hosted.score == 6)
        #expect(unconfirmed.score == 6)
        #expect(hosted.summary.contains("hosted model"))
        #expect(unconfirmed.summary.contains("unconfirmed"))
    }

    @Test func managedSaiIgnoresOllamaMetadataAndOpenRouterPreference() {
        let sai = AgentPrivacyLevel(location: .cloud, computer: .sai, provider: nil,
                                    ollamaInference: .local, requestsZDR: true)
        #expect(sai.score == 5)
        #expect(sai.summary.contains("Tools run on a Sai computer."))
        #expect(sai.summary.contains("Sai manages model inference."))
    }

    @Test(arguments: [(ModelProvider.claudeAccount, "Anthropic"), (.codexAccount, "OpenAI")])
    func nativeAccountsUseHostedInference(provider: ModelProvider, company: String) {
        let level = AgentPrivacyLevel(location: .thisMac, computer: nil, provider: provider,
                                      ollamaInference: .local, requestsZDR: true)
        #expect(level.score == 5)
        #expect(level.summary.contains("\(company) handles inference"))
    }
}
