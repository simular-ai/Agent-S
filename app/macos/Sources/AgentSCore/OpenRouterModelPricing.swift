import Foundation

/// Published OpenRouter text token rates in USD per million tokens.
public enum OpenRouterModelPricing: Hashable, Sendable {
    case rates(input: Decimal, output: Decimal)
    case variable
    case unavailable

    public init(modelID: String, prompt: String? = nil, completion: String? = nil) {
        let input = Self.parse(prompt)
        let output = Self.parse(completion)
        // Auto bills at the routed model's rate, even before the catalog is loaded.
        if modelID == "openrouter/auto" || input == -1 || output == -1 {
            self = .variable
        } else if let input, let output,
                  let inputRate = Self.perMillion(input), let outputRate = Self.perMillion(output) {
            self = .rates(input: inputRate, output: outputRate)
        } else {
            self = .unavailable
        }
    }

    private static func parse(_ text: String?) -> Decimal? {
        // Decimal(string:) can consume only a prefix, including "0" from "0,25".
        guard let text,
              text.range(of: #"\A[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\z"#, options: .regularExpression) != nil,
              let value = Decimal(string: text, locale: Locale(identifier: "en_US_POSIX")), !value.isNaN else { return nil }
        return value
    }

    private static func perMillion(_ value: Decimal) -> Decimal? {
        guard value >= 0 else { return nil }
        var perToken = value
        var scaled = Decimal()
        guard NSDecimalMultiplyByPowerOf10(&scaled, &perToken, 6, .plain) == .noError,
              !scaled.isNaN else { return nil }
        return scaled
    }
}
