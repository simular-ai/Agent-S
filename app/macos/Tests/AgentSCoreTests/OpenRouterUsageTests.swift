import Foundation
import Testing
@testable import AgentSCore

struct OpenRouterUsageTests {
    @Test func recordsNonStreamingCostsAndPreservesZero() {
        var paid = OpenRouterUsage(isStreaming: false)
        paid.receive(Data(#"{"usage":{"cost":0.00000621}}"#.utf8))
        #expect(paid.cost == Decimal(string: "0.00000621"))

        var free = OpenRouterUsage(isStreaming: false)
        free.receive(Data(#"{"usage":{"cost":0}}"#.utf8))
        #expect(free.cost == 0)

        var unreported = OpenRouterUsage(isStreaming: false)
        unreported.receive(Data(#"{"usage":{"prompt_tokens":100}}"#.utf8))
        #expect(unreported.cost == nil)
    }

    @Test func observesChunkedChatUsageWithoutCountingOtherFrames() {
        var usage = OpenRouterUsage(isStreaming: true)
        for line in [": OPENROUTER PROCESSING\n", "\n",
                     #"data: {"object":"chat.completion.chunk","usage":null}"# + "\n", "\n"] {
            usage.receive(Data(line.utf8))
        }
        #expect(usage.cost == nil)
        let firstHalf = #"data: {"object":"chat.completion.chunk","usage"#
        let secondHalf = #"":{"cost":0.0123}}"# + "\r\n"
        usage.receive(Data(firstHalf.utf8))
        usage.receive(Data(secondHalf.utf8))
        #expect(usage.cost == nil)
        usage.receive(Data("\r\n".utf8))
        usage.receive(Data("data: [DONE]\n".utf8))
        usage.receive(Data("\n".utf8))
        #expect(usage.cost == Decimal(string: "0.0123"))
    }

    @Test(arguments: ["response.completed", "response.incomplete"])
    func readsNestedResponseUsage(_ terminal: String) {
        var usage = OpenRouterUsage(isStreaming: true)
        for line in ["event: \(terminal)\n", #"data: {"type":"\#(terminal)","# + "\n",
                     #"data: "response":{"usage":{"cost":0.0432}}}"# + "\n", "\n"] {
            usage.receive(Data(line.utf8))
        }
        #expect(usage.cost == Decimal(string: "0.0432"))
    }

    @Test func readsFinalAnthropicUsage() {
        var usage = OpenRouterUsage(isStreaming: true)
        for line in [#"data: {"type":"message_start","message":{"usage":{"cost":0}}}"# + "\n", "\n"] {
            usage.receive(Data(line.utf8))
        }
        #expect(usage.cost == nil)
        for line in [#"data: {"type":"message_delta","usage":{"cost":0.0101}}"# + "\n", "\n"] {
            usage.receive(Data(line.utf8))
        }
        #expect(usage.cost == Decimal(string: "0.0101"))
    }

    @Test func validatesGenerationFallbackAndDoesNotUseUpstreamCost() {
        #expect(OpenRouterUsage.generationID("gen-AbC-42") == "gen-AbC-42")
        #expect(OpenRouterUsage.generationID("req-AbC-42") == nil)
        #expect(OpenRouterUsage.generationID("gen-ok\n") == nil)
        #expect(OpenRouterUsage.generationCost(in: Data(#"{"data":{"total_cost":0,"upstream_inference_cost":0.001}}"#.utf8)) == 0)
        #expect(OpenRouterUsage.generationCost(in: Data(#"{"data":{"total_cost":null,"upstream_inference_cost":0.001}}"#.utf8)) == nil)
    }
}
