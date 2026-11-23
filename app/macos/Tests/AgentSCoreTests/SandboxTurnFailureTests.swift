import Testing
@testable import AgentSCore

@Suite("Sandbox turn failures")
struct SandboxTurnFailureTests {
    @Test(arguments: [
        "thread-store conflict: thread 12345678-1234-1234-1234-123456789abc already has an active writer",
        "thread 12345678-1234-1234-1234-123456789abc already has an active writer",
        "warning: opening /tmp/fixture/session.jsonl\nERROR: THREAD-STORE CONFLICT: THREAD FIXTURE ALREADY HAS AN ACTIVE WRITER",
    ])
    func mapsWriterConflictsToTheSameRecoveryMessage(diagnostic: String) throws {
        let message = try #require(SandboxTurnFailure.codexWriterMessage(for: diagnostic))
        #expect(message == SandboxTurnFailure.codexWriterRecoveryMessage)
        #expect(!message.contains("12345678"))
        #expect(!message.contains("/tmp/fixture"))
        #expect(message.contains("turn the sandbox off and back on"))
    }

    @Test(arguments: ["", "thread-store conflict: a different error", "ssh: connection reset", "active writer unavailable"])
    func leavesUnrelatedDiagnosticsToExistingClassifiers(diagnostic: String) {
        #expect(SandboxTurnFailure.codexWriterMessage(for: diagnostic) == nil)
    }
}
