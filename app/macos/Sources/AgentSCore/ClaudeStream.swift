import Foundation

/// One thing Claude Code said or did, from its `--output-format stream-json` lines.
public enum ClaudeEvent: Equatable, Sendable {
    /// The session started; `--resume` this id to continue the conversation.
    case started(sessionId: String, model: String?)
    /// Words for the user.
    case text(String)
    case textDelta(id: String, text: String)
    case textSnapshot(id: String, text: String)
    /// A tool call: which tool, and a one-line summary of what it was asked to do.
    case toolCall(id: String, name: String, summary: String, detail: String? = nil)
    /// What a tool call returned (clipped), and whether it failed.
    case toolResult(id: String, output: String, isError: Bool)
    /// The turn ended. `isError` when Claude Code reported a failed turn.
    case finished(result: String?, isError: Bool, costUSD: Double?)
    /// Tokens the turn used: what the model read (including cached prompt) and what it wrote.
    case usage(input: Int, output: Int)
}

extension ClaudeEvent {
    /// A usage event from an API-style `usage` object, if it has numbers.
    static func usage(from object: Any?) -> ClaudeEvent? {
        guard let usage = object as? [String: Any] else { return nil }
        func n(_ key: String) -> Int { (usage[key] as? NSNumber)?.intValue ?? 0 }
        let input = n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens")
        let output = n("output_tokens")
        return input + output > 0 ? .usage(input: input, output: output) : nil
    }
}

/// Parses Claude Code's stream-json output, one JSON object per line.
///
/// Lines it does not understand are skipped rather than failing the stream: the format grows new
/// message types, and an unknown one must not stop the chat.
public struct ClaudeStream {
    /// Tool output kept for the chat; the rest is in the transcript on the computer.
    public static let maxToolOutput = 2_000
    private var latestMessages: [String: String] = [:]
    private var pendingTextBlocks: [String: [Int]] = [:]

    public init() {}

    public static func parse(line: String) -> [ClaudeEvent] {
        var stream = Self()
        return stream.consume(line: line)
    }

    public mutating func consume(line: String) -> [ClaudeEvent] {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = object["type"] as? String
        else { return [] }

        switch type {
        case "stream_event":
            return consumePartial(object)

        case "system":
            guard object["subtype"] as? String == "init",
                  let id = object["session_id"] as? String else { return [] }
            return [.started(sessionId: id, model: object["model"] as? String)]

        case "assistant":
            let message = object["message"] as? [String: Any] ?? [:]
            let messageID = message["id"] as? String
            let parent = object["parent_tool_use_id"] as? String ?? ""
            var events: [ClaudeEvent] = []
            if let messageID, latestMessages[parent] != messageID {
                events = beginMessage(messageID, parent: parent)
            }
            let blocks = message["content"] as? [[String: Any]] ?? []
            events += blocks.compactMap { block in
                switch block["type"] as? String {
                case "text":
                    guard let text = block["text"] as? String else { return nil }
                    // Claude Code emits a separate assistant message for each completed block, without its index.
                    if let messageID,
                       var indices = pendingTextBlocks[messageID], !indices.isEmpty {
                        let index = indices.removeFirst()
                        pendingTextBlocks[messageID] = indices.isEmpty ? nil : indices
                        return .textSnapshot(id: Self.textID(messageID, index), text: text)
                    }
                    return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : .text(text)
                case "tool_use":
                    let name = block["name"] as? String ?? "tool"
                    let input = block["input"] as? [String: Any] ?? [:]
                    return .toolCall(id: block["id"] as? String ?? "",
                                     name: name,
                                     summary: Self.summarize(tool: name, input: input),
                                     detail: ToolActivity.detail(input: input))
                default:
                    return nil
                }
            }
            return events

        case "user":
            let blocks = (object["message"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
            return blocks.compactMap { block in
                guard block["type"] as? String == "tool_result" else { return nil }
                return .toolResult(id: block["tool_use_id"] as? String ?? "",
                                   output: Self.clip(Self.resultText(block["content"])),
                                   isError: block["is_error"] as? Bool ?? false)
            }

        case "result":
            return [ClaudeEvent.usage(from: object["usage"]),
                    .finished(result: object["result"] as? String,
                              isError: object["is_error"] as? Bool ?? false,
                              costUSD: object["total_cost_usd"] as? Double)].compactMap { $0 }

        default:
            return []
        }
    }

    private mutating func consumePartial(_ object: [String: Any]) -> [ClaudeEvent] {
        guard let event = object["event"] as? [String: Any] else { return [] }
        let parent = object["parent_tool_use_id"] as? String ?? ""
        switch event["type"] as? String {
        case "message_start":
            guard let id = (event["message"] as? [String: Any])?["id"] as? String else { return [] }
            return beginMessage(id, parent: parent)
        default:
            break
        }
        guard let messageID = object["api_message_id"] as? String ?? latestMessages[parent],
              let index = event["index"] as? Int else { return [] }
        let text: String
        switch event["type"] as? String {
        case "content_block_start":
            guard let block = event["content_block"] as? [String: Any], block["type"] as? String == "text" else { return [] }
            text = block["text"] as? String ?? ""
        case "content_block_delta":
            guard let delta = event["delta"] as? [String: Any], delta["type"] as? String == "text_delta" else { return [] }
            text = delta["text"] as? String ?? ""
        default:
            return []
        }
        guard !text.isEmpty else { return [] }
        if pendingTextBlocks[messageID]?.last != index {
            pendingTextBlocks[messageID, default: []].append(index)
        }
        return [.textDelta(id: Self.textID(messageID, index), text: text)]
    }

    /// A replacement message may follow a cut-off stream, including after its message_stop.
    private mutating func beginMessage(_ id: String, parent: String) -> [ClaudeEvent] {
        guard let previous = latestMessages.updateValue(id, forKey: parent),
              let abandoned = pendingTextBlocks.removeValue(forKey: previous) else { return [] }
        return abandoned.map { .textSnapshot(id: Self.textID(previous, $0), text: "") }
    }

    private static func textID(_ messageID: String, _ index: Int) -> String { "\(messageID)/\(index)" }

    /// The one line a person reads for a tool call: the command, the file, the pattern.
    public static func summarize(tool: String, input: [String: Any]) -> String {
        let keys = ["description", "command", "file_path", "path", "pattern", "url", "query", "prompt"]
        for key in keys {
            if let value = input[key] as? String, !value.isEmpty {
                let line = value.split(separator: "\n", omittingEmptySubsequences: true).first.map(String.init) ?? value
                return line.count > 120 ? String(line.prefix(119)) + "…" : line
            }
        }
        return tool
    }

    static func resultText(_ content: Any?) -> String {
        if let text = content as? String { return text }
        if let blocks = content as? [[String: Any]] {
            return blocks.compactMap { $0["text"] as? String }.joined(separator: "\n")
        }
        return ""
    }

    static func clip(_ text: String) -> String {
        text.count > maxToolOutput ? String(text.prefix(maxToolOutput)) + "\n…" : text
    }
}
