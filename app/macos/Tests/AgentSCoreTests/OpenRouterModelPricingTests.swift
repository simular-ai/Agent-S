import Foundation
import Testing
@testable import AgentSCore

struct OpenRouterModelPricingTests {
    @Test func convertsUSDPerTokenToUSDPerMillion() {
        let pricing = OpenRouterModelPricing(modelID: "test/model", prompt: "1.5e-8", completion: "0.000000045")
        #expect(pricing == .rates(input: Decimal(string: "0.015")!, output: Decimal(string: "0.045")!))
    }

    @Test func preservesZeroOnEitherRate() {
        #expect(OpenRouterModelPricing(modelID: "openrouter/free", prompt: "0", completion: "0.0") == .rates(input: 0, output: 0))
        #expect(OpenRouterModelPricing(modelID: "test/model", prompt: "0", completion: "0.000002") == .rates(input: 0, output: 2))
    }

    @Test func recognizesDynamicRouters() {
        #expect(OpenRouterModelPricing(modelID: "openrouter/auto") == .variable)
        #expect(OpenRouterModelPricing(modelID: "openrouter/auto-code", prompt: "-1", completion: "-1") == .variable)
        #expect(OpenRouterModelPricing(modelID: "test/routed-model", prompt: "0.000001", completion: "-1") == .variable)
    }

    @Test func distinguishesMissingAndOverflowingPrices() {
        #expect(OpenRouterModelPricing(modelID: "test/unknown") == .unavailable)
        #expect(OpenRouterModelPricing(modelID: "test/partial", prompt: "0") == .unavailable)
        #expect(OpenRouterModelPricing(modelID: "test/overflow", prompt: "1e127", completion: "0") == .unavailable)
    }

    @Test(arguments: ["", "1,5", "2extra", "0\n", "1e", "NaN", "-2", " 0.1 "])
    func rejectsInvalidPricing(_ invalid: String) {
        #expect(OpenRouterModelPricing(modelID: "test/model", prompt: invalid, completion: "0.000001") == .unavailable)
    }
}
