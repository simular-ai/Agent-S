import Testing
@testable import AgentSCore

@Suite("Agent handoff request headings")
struct AgentHandoffRequestTests {
    @Test(arguments: [
        "Request for @Research Agent:",
        "request FOR @RESEARCH AGENT",
        "### Request for @Research Agent ###",
        "## **Request for @Research Agent:**",
        "__Request for @Research Agent__:",
    ])
    func removesLeadingHeadingAndRule(_ heading: String) {
        let text = "\n\(heading)\n\n  - - -\n\nCheck the figures.\n\nUse the latest report."
        #expect(AgentHandoffRequest.withoutRecipientHeading(text, recipientName: "Research Agent")
                == "Check the figures.\n\nUse the latest report.")
    }

    @Test(arguments: [
        "Request for @Research Agent: ",
        "**Request for @Research Agent:** ",
        "### _Request for @Research Agent_: ",
    ])
    func keepsTasksOnTheHeadingLineAndSubsequentRules(_ heading: String) {
        let task = "Check the figures.\r\n---\r\nInclude the totals."
        #expect(AgentHandoffRequest.withoutRecipientHeading(heading + task, recipientName: "Research Agent") == task)
    }

    @Test(arguments: [
        "Request for @Research Agent 2:\n---\nActual task",
        "**Request for @Other Agent:**\nActual task",
        "Request for @Research Agent today\nActual task",
        "Request for @Research Agent_name:\nActual task",
        "Please prepare a reply.\nRequest for @Research Agent:\nActual task",
        "```markdown\nRequest for @Research Agent:\n\n```",
    ])
    func preservesOtherRecipientsAndBodyText(_ request: String) {
        #expect(AgentHandoffRequest.withoutRecipientHeading(request, recipientName: "Research Agent") == request)
    }

    @Test(arguments: [
        "  ---- bullet text",
        "  - * -",
        "    ---",
        "    return forecast",
        "## Background\nRead the report first.",
    ])
    func preservesBodyTextThatIsNotAHorizontalRule(_ task: String) {
        let request = "Request for @Research Agent:\n\n" + task
        #expect(AgentHandoffRequest.withoutRecipientHeading(request, recipientName: "Research Agent") == task)
    }

    @Test func preservesMentionCharactersAndStripsAnEmptyHeading() {
        let request = "**Request for @Analysis [QA] + Metrics:**\n\n ***\n "
        #expect(AgentHandoffRequest.withoutRecipientHeading(request, recipientName: "Analysis [QA] + Metrics").isEmpty)
        #expect(AgentHandoffRequest.withoutRecipientHeading(request, recipientName: "") == request)
    }
}
