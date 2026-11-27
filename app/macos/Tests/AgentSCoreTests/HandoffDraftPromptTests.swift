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
        #expect(prompt.hasSuffix(String(reflecting: "Share that result with @Research")))
        #expect(HandoffDraftPrompt.instructions.contains("already resolved"))
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

    @Test(arguments: [
        #"I cannot send the request — no agent named **Sai** is currently available. Only the lead agent is active."#,
        #"{"message":"I cannot send the request — no agent named **Sai** is currently available. I can open the page instead."}"#,
        #"{"message":"**No agent named Sai** is available."}"#,
        #"{"message":"I’m unable to forward this message to Sai."}"#,
        #"{"message":"The inventory contains 49 items. Only Lead is active. There is no active agent named Sai."}"#,
        #"{"message":"Sai is currently unavailable."}"#,
        #"{"message":null}"#,
        #"{"message":"..."}"#,
        #"{"message":"Open the site.","recipient":"lead"}"#,
        "Draft:\n{\"message\":\"Open the site.\"}",
    ])
    func rejectsRoutingDiagnosticsAndNonMessageOutput(_ output: String) {
        #expect(HandoffDraftPrompt.message(from: output, recipient: "Sai") == nil)
    }

    @Test func acceptsDirectTasksAndReportsAddressedToTheRecipient() {
        let task = #"{"message":"Open https://example.test and report the page title."}"#
        #expect(HandoffDraftPrompt.message(from: task, recipient: "Sai") == "Open https://example.test and report the page title.")

        let result = #"{"message":"I couldn’t verify the total. Please compare it with the prior report."}"#
        #expect(HandoffDraftPrompt.message(from: result, recipient: "Sai")
                == "I couldn’t verify the total. Please compare it with the prior report.")

        let lookup = #"{"message":"Check whether Sai is available in the team calendar."}"#
        #expect(HandoffDraftPrompt.message(from: lookup, recipient: "Sai")
                == "Check whether Sai is available in the team calendar.")

        let fenced = "```json\n{\"message\":\"Send me the report.\"}\n```"
        #expect(HandoffDraftPrompt.message(from: fenced, recipient: "Sai") == "Send me the report.")
    }

    @Test func rejectsAnEchoOfTheRoutingInstruction() {
        let response = #"{"message":"Ask @Sai to open https://example.test."}"#
        #expect(HandoffDraftPrompt.message(from: response, recipient: "Sai",
                                         request: "Ask @Sai to open https://example.test.") == nil)
        #expect(HandoffDraftPrompt.message(from: #"{"message":"Open https://example.test."}"#,
                                         recipient: "Sai", request: "Open https://example.test.") == "Open https://example.test.")
    }

    @Test func retryKeepsFactsFromRepliesThatIncludedStaleRoutingDiagnostics() {
        let mixed = HandoffDraftTurn(request: "Count the inventory",
                                    reply: "The report at https://example.test lists 49 items. Only the lead agent is active. No agent named Sai is currently available.")
        let independent = HandoffDraftTurn(request: "Check the numbers", reply: "The total is correct.")
        let diagnostic = HandoffDraftTurn(request: "Troubleshoot", reply: "I cannot send the request.")
        #expect(HandoffDraftPrompt.retryContext([mixed, independent, diagnostic], recipient: "Sai") == [
            .init(request: mixed.request, reply: "The report at https://example.test lists 49 items."), independent,
        ])
    }
}
