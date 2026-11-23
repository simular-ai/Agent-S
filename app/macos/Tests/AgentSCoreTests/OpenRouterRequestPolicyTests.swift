import Foundation
import Testing
@testable import AgentSCore

@Suite("OpenRouter request policy")
struct OpenRouterRequestPolicyTests {
    @Test(arguments: [
        ("/api/v1/messages", #"{"model":"example/model","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"max_tokens":256}"#),
        ("/api/v1/responses", #"{"model":"example/model","stream":true,"input":[{"role":"user","content":"hello"}],"store":false,"include":["reasoning.encrypted_content"]}"#),
        ("/api/v1/chat/completions", #"{"model":"example/model","stream":true,"messages":[{"role":"user","content":"hello"}],"stream_options":{"include_usage":true}}"#),
    ])
    func preservesEachInferenceFormat(path: String, payload: String) throws {
        let original = Data(payload.utf8)
        let body = try OpenRouterRequestPolicy.body(original, method: "POST", path: path, requiresZDR: true)
        var json = try decode(body)
        #expect(json.removeValue(forKey: "provider") as? [String: Bool] == ["zdr": true])
        #expect(try NSDictionary(dictionary: json) == NSDictionary(dictionary: decode(original)))
    }

    @Test func overridesOnlyZDRInExistingProviderPreferences() throws {
        let original = Data(#"{"provider":{"zdr":false,"order":["provider"],"allow_fallbacks":false,"sort":"throughput"}}"#.utf8)
        let body = try OpenRouterRequestPolicy.body(original, method: "POST", path: "/api/v1/messages", requiresZDR: true)
        let expected = Data(#"{"provider":{"zdr":true,"order":["provider"],"allow_fallbacks":false,"sort":"throughput"}}"#.utf8)
        #expect(try NSDictionary(dictionary: decode(body)) == NSDictionary(dictionary: decode(expected)))
    }

    @Test func acceptsNullProviderPreferences() throws {
        let original = Data(#"{"provider":null}"#.utf8)
        let body = try OpenRouterRequestPolicy.body(original, method: "POST", path: "/api/v1/responses", requiresZDR: true)
        #expect(try decode(body)["provider"] as? [String: Bool] == ["zdr": true])
    }

    @Test func classifiesInferencePathWithAQuery() throws {
        let url = try #require(URL(string: "https://openrouter.ai/api/v1/messages?beta=true"))
        let body = try OpenRouterRequestPolicy.body(Data("{}".utf8), method: "POST", path: url.path, requiresZDR: true)
        #expect(try decode(body)["provider"] as? [String: Bool] == ["zdr": true])
    }

    @Test(arguments: ["", "{", "[]"])
    func rejectsMalformedOrNonObjectInferenceBodies(payload: String) {
        #expect(throws: OpenRouterRequestPolicy.Rejection.invalidJSONBody) {
            try OpenRouterRequestPolicy.body(Data(payload.utf8), method: "POST", path: "/api/v1/responses", requiresZDR: true)
        }
    }

    @Test func rejectsInvalidProviderPreferencesAndUnsupportedBodyBearingRoutes() {
        #expect(throws: OpenRouterRequestPolicy.Rejection.invalidProviderPreferences) {
            try OpenRouterRequestPolicy.body(Data(#"{"provider":[]}"#.utf8), method: "POST",
                                              path: "/api/v1/chat/completions", requiresZDR: true)
        }
        for method in ["GET", "POST"] {
            #expect(throws: OpenRouterRequestPolicy.Rejection.unsupportedRoute) {
                try OpenRouterRequestPolicy.body(Data("{}".utf8), method: method,
                                                  path: "/api/v1/future", requiresZDR: true)
            }
        }
    }

    @Test func keepsBodylessDiscoveryAndPreservesOriginalBytesWhenDisabled() throws {
        #expect(try OpenRouterRequestPolicy.body(Data(), method: "GET", path: "/api/v1/models", requiresZDR: true).isEmpty)
        let body = Data(#"{ "provider" : { "zdr" : false } }"#.utf8)
        #expect(try OpenRouterRequestPolicy.body(body, method: "POST", path: "/api/v1/messages", requiresZDR: false) == body)
        #expect(try OpenRouterRequestPolicy.body(body, method: "POST", path: "/api/v1/future", requiresZDR: false) == body)
    }

    private func decode(_ data: Data) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
