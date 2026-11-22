import Testing
@testable import AgentSCore

@Suite("Harness account plans")
struct HarnessAccountPlanTests {
    @Test(arguments: ["pro", "max"])
    func recognizesClaudeSubscriptions(_ plan: String) {
        #expect(HarnessAccountPlan(provider: .claudeAccount, reportedPlan: plan) == .subscription)
    }

    @Test(arguments: ["go", "plus", "pro"])
    func recognizesChatGPTSubscriptions(_ plan: String) {
        #expect(HarnessAccountPlan(provider: .codexAccount, reportedPlan: plan) == .subscription)
    }

    @Test(arguments: [nil, "free", "unknown", "self_serve_business_usage_based"] as [String?])
    func unconfirmedSubscriptionsStayAccount(_ plan: String?) {
        #expect(HarnessAccountPlan(provider: .claudeAccount, reportedPlan: plan) == .account)
        #expect(HarnessAccountPlan(provider: .codexAccount, reportedPlan: plan) == .account)
    }
}
