import Foundation

struct OpenRouterKeyUsage: Decodable, Sendable {
    let daily: Decimal
    let weekly: Decimal
    let monthly: Decimal
    let total: Decimal

    private enum CodingKeys: String, CodingKey {
        case daily = "usage_daily"
        case weekly = "usage_weekly"
        case monthly = "usage_monthly"
        case total = "usage"
    }

    enum State {
        case idle
        case loading
        case loaded(OpenRouterKeyUsage)
        case failed(String)
    }
}

enum OpenRouterKeyUsageError: LocalizedError {
    case rejectedKey
    case unavailable

    var errorDescription: String? {
        switch self {
        case .rejectedKey: "OpenRouter didn’t accept the saved API key."
        case .unavailable: "Couldn’t load OpenRouter usage. Try again."
        }
    }
}

extension OpenRouter {
    private static let keyUsageSession = URLSession(configuration: .ephemeral)

    static func formattedSpend(_ amount: Decimal) -> String {
        amount.formatted(.currency(code: "USD").precision(.fractionLength(2...8)))
    }

    /// Reads spend for the key in the Authorization header. No management key is needed.
    static func keyUsage(using key: String) async throws -> OpenRouterKeyUsage {
        var request = URLRequest(url: api.appendingPathComponent("v1/key"),
                                 cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await keyUsageSession.data(for: request)
        switch (response as? HTTPURLResponse)?.statusCode {
        case 200:
            struct Envelope: Decodable { let data: OpenRouterKeyUsage }
            return try JSONDecoder().decode(Envelope.self, from: data).data
        case 401, 403:
            throw OpenRouterKeyUsageError.rejectedKey
        default:
            throw OpenRouterKeyUsageError.unavailable
        }
    }
}
