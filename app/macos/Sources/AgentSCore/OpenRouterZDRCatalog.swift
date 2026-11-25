import Foundation

public enum OpenRouterZDRAvailability: Hashable, Sendable {
    case available
    case safetyRetentionOnly
    case unavailable
    case unknown

    public var label: String {
        switch self {
        case .available: "ZDR available"
        case .safetyRetentionOnly: "Safety Retention only"
        case .unavailable: "No public ZDR endpoint"
        case .unknown: "ZDR unverified"
        }
    }
}

/// Endpoint availability reported by OpenRouter's public ZDR catalog.
public struct OpenRouterZDRCatalog: Sendable {
    private struct Response: Decodable {
        let data: [Endpoint]
    }

    private struct Endpoint: Decodable {
        let modelID: String
        let safetyRetention: Bool?

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case safetyRetention = "safety_retention"
        }
    }

    private let availabilityByModel: [String: OpenRouterZDRAvailability]

    public init(data: Data) throws {
        let endpoints = try JSONDecoder().decode(Response.self, from: data).data
        guard endpoints.allSatisfy({ !$0.modelID.isEmpty && !$0.modelID.contains(where: \.isWhitespace) }) else {
            throw DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: "Invalid OpenRouter model ID"))
        }
        availabilityByModel = Dictionary(grouping: endpoints, by: \.modelID).mapValues { matches in
            if matches.contains(where: { $0.safetyRetention == false }) { return .available }
            if matches.allSatisfy({ $0.safetyRetention == true }) { return .safetyRetentionOnly }
            return .unknown
        }
    }

    public func availability(for id: String, isPublicModel: Bool) -> OpenRouterZDRAvailability {
        availabilityByModel[id] ?? (isPublicModel ? .unavailable : .unknown)
    }
}
