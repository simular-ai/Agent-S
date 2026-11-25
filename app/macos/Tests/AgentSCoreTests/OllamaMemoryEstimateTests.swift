import Testing
import AgentSCore

@Suite("Ollama memory estimate")
struct OllamaMemoryEstimateTests {
    private let gibibyte: UInt64 = 1 << 30

    @Test func usesTheUpperEndOfCatalogRanges() throws {
        let estimate = try #require(OllamaMemoryEstimate(catalogSize: "2.8 GB – 2.9 GB"))
        let sandbox = 2 * gibibyte
        let requiredMemory = sandbox + gibibyte + 2_900_000_000
        #expect(!estimate.exceedsBudget(availableMemory: requiredMemory, sandboxMemory: sandbox))
        #expect(estimate.exceedsBudget(availableMemory: requiredMemory - 1, sandboxMemory: sandbox))
    }

    @Test func distinguishesDecimalAndBinarySizes() throws {
        let decimal = try #require(OllamaMemoryEstimate(catalogSize: "1 GB"))
        let binary = try #require(OllamaMemoryEstimate(catalogSize: "1 GiB"))
        let memory = gibibyte + 1_000_000_000
        #expect(!decimal.exceedsBudget(availableMemory: memory, sandboxMemory: 0))
        #expect(binary.exceedsBudget(availableMemory: memory, sandboxMemory: 0))
    }

    @Test func subtractsOnlyPendingSandboxMemory() throws {
        let estimate = try #require(OllamaMemoryEstimate(modelBytes: 3 * gibibyte))
        let available = 4 * gibibyte
        let pending = 2 * gibibyte
        #expect(!estimate.exceedsBudget(availableMemory: available, sandboxMemory: 0))
        #expect(estimate.exceedsBudget(availableMemory: available, sandboxMemory: pending))
        #expect(!estimate.exceedsBudget(availableMemory: available + pending, sandboxMemory: pending))
        #expect(estimate.exceedsBudget(availableMemory: available + pending - 1, sandboxMemory: pending))
    }

    @Test func roundsUpRuntimeHeadroomForLargeModels() throws {
        let estimate = try #require(OllamaMemoryEstimate(modelBytes: 8 * gibibyte + 1))
        let required = 10 * gibibyte + 2
        #expect(!estimate.exceedsBudget(availableMemory: required, sandboxMemory: 0))
        #expect(estimate.exceedsBudget(availableMemory: required - 1, sandboxMemory: 0))
    }

    @Test func handlesUnknownSizesAndOutOfRangeBudgets() throws {
        for unknown in ["", "Unknown", "Medium Usage", "0 GB", "-2 GB", "32b parameters", "18 GB – Unknown", "18446744073709551616 B"] {
            #expect(OllamaMemoryEstimate(catalogSize: unknown) == nil)
        }
        #expect(OllamaMemoryEstimate(modelBytes: 0) == nil)

        let estimate = try #require(OllamaMemoryEstimate(modelBytes: .max))
        #expect(estimate.exceedsBudget(availableMemory: .max, sandboxMemory: 0))
        #expect(estimate.exceedsBudget(availableMemory: 8 * gibibyte, sandboxMemory: .max))
        #expect(estimate.exceedsBudget(availableMemory: 0, sandboxMemory: 0))
    }
}
