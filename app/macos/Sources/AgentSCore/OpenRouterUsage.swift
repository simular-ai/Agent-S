import Foundation

/// Observes copies of proxy response chunks. Accounting never changes the relayed response.
public struct OpenRouterUsage {
    private static let limit = 1 << 20
    private static let dataField = Data("data:".utf8)

    private let isStreaming: Bool
    private var buffer = Data()
    private var event = Data()
    private var discardLine = false
    private var discardEvent = false
    private var streamedCost: Decimal?

    public static func tracks(method: String, path: String) -> Bool {
        method == "POST" && ["/api/v1/messages", "/api/v1/responses", "/api/v1/chat/completions"].contains(path)
    }

    public static func generationID(_ value: String?) -> String? {
        guard let value, value.utf8.count <= 128,
              value.range(of: #"\Agen-[0-9A-Za-z-]+\z"#, options: .regularExpression) != nil else { return nil }
        return value
    }

    public static func generationCost(in data: Data) -> Decimal? {
        struct Generation: Decodable {
            struct Metadata: Decodable { let total_cost: Decimal }
            let data: Metadata
        }
        return validCost((try? JSONDecoder().decode(Generation.self, from: data))?.data.total_cost)
    }

    public init(isStreaming: Bool) { self.isStreaming = isStreaming }

    /// The proxy writes at most one newline per chunk, at the end of the chunk.
    public mutating func receive(_ chunk: Data) {
        if !isStreaming {
            if !discardLine {
                if buffer.count + chunk.count <= Self.limit { buffer.append(chunk) }
                else { buffer.removeAll(); discardLine = true }
            }
            return
        }

        let endsLine = chunk.last == 0x0A
        let bytes = endsLine ? chunk.dropLast() : chunk[...]
        if !discardLine {
            if buffer.count + bytes.count <= Self.limit { buffer.append(bytes) }
            else { buffer.removeAll(); discardLine = true }
        }
        guard endsLine else { return }
        if discardLine {
            discardEvent = true
        } else {
            if buffer.last == 0x0D { buffer.removeLast() }
            receiveLine()
        }
        buffer.removeAll(keepingCapacity: true)
        discardLine = false
    }

    public var cost: Decimal? {
        if isStreaming { return streamedCost }
        guard !discardLine else { return nil }
        return Self.validCost(decode(buffer)?.usage?.cost)
    }

    private mutating func receiveLine() {
        if buffer.isEmpty {
            if !discardEvent, let frame = decode(event) {
                let reported: Decimal?
                switch frame.type {
                case "response.completed", "response.incomplete": reported = frame.response?.usage?.cost
                case "message_delta": reported = frame.usage?.cost
                default: reported = frame.object == "chat.completion.chunk" ? frame.usage?.cost : nil
                }
                if let amount = Self.validCost(reported) { streamedCost = amount }
            }
            event.removeAll(keepingCapacity: true)
            discardEvent = false
        } else if !discardEvent, buffer.starts(with: Self.dataField) {
            var payload = buffer.dropFirst(Self.dataField.count)
            if payload.first == 0x20 { payload = payload.dropFirst() }
            if event.count + payload.count + 1 <= Self.limit {
                event.append(payload)
                event.append(0x0A)
            } else {
                discardEvent = true
            }
        }
    }

    private struct Frame: Decodable {
        struct Usage: Decodable { let cost: Decimal? }
        struct Response: Decodable { let usage: Usage? }
        let object: String?
        let type: String?
        let usage: Usage?
        let response: Response?
    }

    private func decode(_ data: Data) -> Frame? {
        try? JSONDecoder().decode(Frame.self, from: data)
    }

    private static func validCost(_ value: Decimal?) -> Decimal? {
        guard let value, !value.isNaN, value >= 0 else { return nil }
        return value
    }
}
