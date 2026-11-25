import Foundation
import Testing
import AgentSCore

@Suite("OpenRouter ZDR endpoint availability")
struct OpenRouterZDRCatalogTests {
    @Test func anyStrictEndpointEstablishesAvailability() throws {
        let catalog = try OpenRouterZDRCatalog(data: Data(#"""
        {"data":[
          {"model_id":"example/mixed","safety_retention":true},
          {"model_id":"example/mixed","safety_retention":null},
          {"model_id":"example/mixed","safety_retention":false},
          {"model_id":"example/mixed"},
          {"model_id":"example/safety","safety_retention":true},
          {"model_id":"example/safety","safety_retention":true}
        ]}
        """#.utf8))
        #expect(catalog.availability(for: "example/mixed", isPublicModel: true) == .available)
        #expect(catalog.availability(for: "example/mixed", isPublicModel: false) == .available)
        #expect(catalog.availability(for: "example/safety", isPublicModel: true) == .safetyRetentionOnly)
    }

    @Test func ambiguousEndpointFlagsStayUnverified() throws {
        let catalog = try OpenRouterZDRCatalog(data: Data(#"""
        {"data":[
          {"model_id":"example/ambiguous","safety_retention":true},
          {"model_id":"example/ambiguous"},
          {"model_id":"example/null","safety_retention":null}
        ]}
        """#.utf8))
        #expect(catalog.availability(for: "example/ambiguous", isPublicModel: true) == .unknown)
        #expect(catalog.availability(for: "example/null", isPublicModel: true) == .unknown)
    }

    @Test func matchesExactIDsAndRequiresPublicMembershipToReportUnavailable() throws {
        let catalog = try OpenRouterZDRCatalog(data: Data(#"{"data":[{"model_id":"example/model:free","safety_retention":false}]}"#.utf8))
        #expect(catalog.availability(for: "example/model:free", isPublicModel: true) == .available)
        #expect(catalog.availability(for: "example/model", isPublicModel: true) == .unavailable)
        #expect(catalog.availability(for: "example/model", isPublicModel: false) == .unknown)
        #expect(catalog.availability(for: "Example/model:free", isPublicModel: false) == .unknown)
    }

    @Test(arguments: [
        #"{"error":{"message":"Could not load endpoints"}}"#,
        #"{"data":null}"#,
        #"{"data":{"model_id":"example/model"}}"#,
        #"{"data":[{"safety_retention":true}]}"#,
        #"{"data":[{"model_id":" ","safety_retention":true}]}"#,
        #"{"data":[{"model_id":"example/model","safety_retention":"false"}]}"#,
        #"{"data":[{"model_id":"example/model","safety_retention":0}]}"#,
    ])
    func rejectsMalformedCatalogs(payload: String) {
        #expect(throws: DecodingError.self) {
            try OpenRouterZDRCatalog(data: Data(payload.utf8))
        }
    }
}
