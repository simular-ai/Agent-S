import Testing
@testable import AgentSCore

@Suite("Contextual handoff drafts")
struct HandoffDraftPromptTests {
    @Test func quotesTheUserRequestAndKeepsContextSeparate() {
        let prompt = HandoffDraftPrompt.prompt(request: "Share that result with @Research",
                                             recipient: "Research",
                                             context: [.init(request: "Check the report", reply: "The total is 42.")])
        #expect(prompt.contains("Current user request: \"Share that result with @Research\""))
        #expect(prompt.contains("Previous conversation, oldest first:\nUser: \"Check the report\"\nAgent: \"The total is 42.\""))
        #expect(HandoffDraftPrompt.instructions.contains("instead of instructing the recipient"))
    }

    @Test func boundsDraftContextAndKeepsTheNewestTurns() {
        let prompt = HandoffDraftPrompt.prompt(request: "Summarize", recipient: "Research",
                                             context: [
                                                .init(request: "Too old", reply: "Too old"),
                                                .init(request: "First recent", reply: "First result"),
                                                .init(request: "Second recent", reply: "Second result"),
                                                .init(request: "Latest", reply: String(repeating: "A", count: 1_400)
                                                      + "Omitted middle" + String(repeating: "B", count: 1_400)
                                                      + "Final result"),
                                             ])
        #expect(!prompt.contains("Too old"))
        #expect(!prompt.contains("Omitted middle"))
        #expect(prompt.contains("First recent"))
        #expect(prompt.contains("Latest"))
        #expect(prompt.contains("Final result"))
        #expect(prompt.contains("…"))
    }

    @Test func buildsRecipientPromptFromTheApprovedBodyOnly() {
        let message = "The total is 42.\nPlease check the calculation."
        let request = HandoffDraftPrompt.approvedRequest(message, source: "Finance")
        #expect(request.hasSuffix(String(reflecting: message)))
        #expect(request.contains("from the agent named \"Finance\""))
        #expect(!request.contains("Previous conversation"))
        #expect(!request.contains("Current user request"))
    }
}
