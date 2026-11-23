import Testing
@testable import AgentSCore

@Suite("Sandbox conversation routing")
struct SandboxSessionTests {
    @Test func migratesLocalAndHostedHistoryToTheirExistingProviders() {
        let stored = [Harness.claudeCode.rawValue: "existing-claude", Harness.codex.rawValue: "existing-codex"]

        for (mode, provider) in [(PrivacyMode.fullyPrivate, ModelProvider.ollama), (.localVM, .openRouter)] {
            let migrated = SandboxSessions.migrating(stored, from: mode)
            #expect(migrated[SandboxSessions.key(harness: .claudeCode, provider: provider)] == "existing-claude")
            #expect(migrated[SandboxSessions.key(harness: .codex, provider: provider)] == "existing-codex")
            #expect(migrated[SandboxSessions.key(harness: .claudeCode, provider: .claudeAccount)] == nil)
            #expect(migrated[SandboxSessions.key(harness: .codex, provider: .codexAccount)] == nil)
            #expect(migrated[Harness.claudeCode.rawValue] == nil)
            #expect(migrated[Harness.codex.rawValue] == nil)
        }
    }

    @Test func retainsScopedSessionsWhenSavedProviderChanges() {
        let ollama = SandboxSessions.key(harness: .codex, provider: .ollama)
        let subscription = SandboxSessions.key(harness: .codex, provider: .codexAccount)
        let stored = [ollama: "ollama-session", subscription: "account-session"]

        #expect(SandboxSessions.migrating(stored, from: .localVM) == stored)
        #expect(SandboxSessions.migrating(stored, from: .fullyPrivate) == stored)
    }
}
