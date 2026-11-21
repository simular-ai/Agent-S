import Foundation

/// One thing Claude Code said or did, from its `--output-format stream-json` lines.
public enum ClaudeEvent: Equatable, Sendable {
    /// The session started; `--resume` this id to continue the conversation.
    case started(sessionId: String, model: String?)
    /// Words for the user.
    case text(String)
    /// A tool call: which tool, and a one-line summary of what it was asked to do.
    case toolCall(id: String, name: String, summary: String)
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
public enum ClaudeStream {
    /// Tool output kept for the chat; the rest is in the transcript on the computer.
    public static let maxToolOutput = 2_000

    public static func parse(line: String) -> [ClaudeEvent] {
        guard let data = line.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = object["type"] as? String
        else { return [] }

        switch type {
        case "system":
            guard object["subtype"] as? String == "init",
                  let id = object["session_id"] as? String else { return [] }
            return [.started(sessionId: id, model: object["model"] as? String)]

        case "assistant":
            let blocks = (object["message"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
            return blocks.compactMap { block in
                switch block["type"] as? String {
                case "text":
                    guard let text = block["text"] as? String,
                          !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                    return .text(text)
                case "tool_use":
                    let name = block["name"] as? String ?? "tool"
                    let input = block["input"] as? [String: Any] ?? [:]
                    return .toolCall(id: block["id"] as? String ?? "",
                                     name: name,
                                     summary: summarize(tool: name, input: input))
                default:
                    return nil
                }
            }

        case "user":
            let blocks = (object["message"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
            return blocks.compactMap { block in
                guard block["type"] as? String == "tool_result" else { return nil }
                return .toolResult(id: block["tool_use_id"] as? String ?? "",
                                   output: clip(resultText(block["content"])),
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
