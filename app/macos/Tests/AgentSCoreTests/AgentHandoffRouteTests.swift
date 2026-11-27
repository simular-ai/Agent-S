import Testing
@testable import AgentSCore

@Suite("Agent handoff routes")
struct AgentHandoffRouteTests {
    private let local = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                         ollamaInference: .local, model: "synthetic")
    private let sandboxHosted = AgentMemoryRoute(privacy: .localVM, harness: .deepSeek, provider: .openRouter,
                                                 model: "synthetic")
    private let remote = AgentMemoryRoute(privacy: .remote, harness: .sai, provider: nil, saiMachine: "remote")

    @Test func requiresApprovalForHostedHandoffsEvenWhenSharingASandbox() {
        #expect(local.canRouteHandoff)
        #expect(sandboxHosted.canRouteHandoff)
        #expect(remote.canRouteHandoff)
        #expect(local.requiresHandoffApproval(to: sandboxHosted))
        #expect(local.requiresHandoffApproval(to: remote))
        #expect(sandboxHosted.requiresHandoffApproval(to: sandboxHosted))
    }

    @Test func requiresApprovalForHostedInferenceOnThisMac() {
        let macHosted = AgentMemoryRoute(privacy: .thisMac, harness: .codex, provider: .codexAccount)
        #expect(macHosted.canRouteHandoff)
        #expect(local.requiresHandoffApproval(to: macHosted))
    }

    @Test func admitsKnownOllamaRoutesAndBlocksUnknownRoutes() {
        let hosted = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                      ollamaInference: .hosted, model: "synthetic")
        let unconfirmed = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                           model: "synthetic")
        let invalid = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .openRouter)
        #expect(hosted.canRouteHandoff)
        #expect(local.requiresHandoffApproval(to: hosted))
        #expect(!unconfirmed.canRouteHandoff)
        #expect(local.requiresHandoffApproval(to: unconfirmed))
        #expect(!invalid.canRouteHandoff)
    }

    @Test func previewsChangesInTheDestinationCloudComputer() {
        let otherCloud = AgentMemoryRoute(privacy: .remote, harness: .sai, provider: nil, saiMachine: "other")
        let remoteWithLocalInference = AgentMemoryRoute(privacy: .remote, harness: .deepSeek, provider: .ollama,
                                                         computer: .box, boxName: "box", ollamaInference: .local,
                                                         model: "synthetic")
        #expect(!remote.requiresHandoffApproval(to: remote))
        #expect(remote.requiresHandoffApproval(to: otherCloud))
        #expect(local.requiresHandoffApproval(to: remoteWithLocalInference))
    }

    @Test func reviewsOllamaTransfersEvenWithCurrentlyLocalInference() {
        #expect(remote.requiresHandoffApproval(to: local))
        #expect(local.requiresHandoffApproval(to: remote))
        #expect(sandboxHosted.requiresHandoffApproval(to: local))
        #expect(local.requiresHandoffApproval(to: local))
    }
}
