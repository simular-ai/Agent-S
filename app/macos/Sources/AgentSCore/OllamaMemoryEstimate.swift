import Foundation

/// An advisory estimate based on model size. Ollama's actual memory use depends on the workload.
public struct OllamaMemoryEstimate: Sendable {
    private static let gibibyte: UInt64 = 1 << 30
    private let modelBytes: UInt64

    public init?(modelBytes: UInt64) {
        guard modelBytes > 0 else { return nil }
        self.modelBytes = modelBytes
    }

    public init?(catalogSize: String) {
        let size = catalogSize.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let match = size.wholeMatch(of: #/([0-9]+(?:\.[0-9]+)?)\s*([kKMGTP]?i?B)(?:\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*([kKMGTP]?i?B))?/#),
              let first = Self.byteCount(match.1, unit: match.2) else { return nil }

        if let amount = match.3, let unit = match.4 {
            guard let last = Self.byteCount(amount, unit: unit) else { return nil }
            self.init(modelBytes: max(first, last))
        } else {
            self.init(modelBytes: first)
        }
    }

    /// The current available memory already accounts for the OS and any running sandbox.
    /// Pass only the memory needed to start a sandbox that is still off.
    public func exceedsBudget(availableMemory: UInt64, sandboxMemory: UInt64) -> Bool {
        guard availableMemory >= sandboxMemory else { return true }
        let modelBudget = availableMemory - sandboxMemory
        guard modelBudget >= modelBytes else { return true }

        let quarter = modelBytes / 4 + (modelBytes % 4 == 0 ? 0 : 1)
        let runtimeHeadroom = max(Self.gibibyte, quarter)
        return modelBudget - modelBytes < runtimeHeadroom
    }

    private static let unitBytes: [String: Double] = [
        "B": 1,
        "KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000,
        "TB": 1_000_000_000_000, "PB": 1_000_000_000_000_000,
        "KIB": Double(1 << 10), "MIB": Double(1 << 20), "GIB": Double(1 << 30),
        "TIB": Double(1 << 40), "PIB": Double(1 << 50),
    ]

    private static func byteCount(_ amount: Substring, unit: Substring) -> UInt64? {
        guard let number = Double(amount), let multiplier = unitBytes[unit.uppercased()] else { return nil }
        return UInt64(exactly: (number * multiplier).rounded(.up))
    }
}
