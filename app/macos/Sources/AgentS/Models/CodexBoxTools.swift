import Foundation
import AgentSCore

/// Only the tools advertised by the selected Box are available to Codex's dynamic tool dispatcher.
@MainActor
struct CodexBoxTools {
    let definitions: [[String: Any]]
    private let names: Set<String>
    private let boxName: String?

    static func load(boxName: String?) async throws -> Self {
        let tools = try await BoxMCP.shared.agentTools(box: boxName)
        var names = Set<String>()
        let definitions = tools.compactMap { tool -> [String: Any]? in
            guard let name = tool["name"] as? String,
                  let inputSchema = tool["inputSchema"] as? [String: Any], names.insert(name).inserted else { return nil }
            return ["type": "function", "name": name, "description": tool["description"] as? String ?? name,
                    "inputSchema": inputSchema]
        }
        guard !definitions.isEmpty else {
            throw BoxAccount.Failure(message: "No tools were available for the selected Simular Box.")
        }
        return Self(definitions: definitions, names: names, boxName: boxName)
    }

    var instructions: String { Box.agentInstructions(boxName: boxName) }

    func call(name: String, id: String, arguments: [String: Any], onEvent: (ClaudeEvent) -> Void) async -> [String: Any] {
        guard names.contains(name) else {
            return ["success": false, "contentItems": [["type": "inputText", "text": "That Box tool is unavailable."]]]
        }
        onEvent(.toolCall(id: id, name: name, summary: ClaudeStream.summarize(tool: name, input: arguments),
                          detail: ToolActivity.detail(input: arguments)))
        let items: [[String: String]]
        let success: Bool
        do {
            let result = try await BoxMCP.shared.callAgentTool(name, arguments: arguments, box: boxName)
            items = Self.contentItems(result)
            success = result["isError"] as? Bool != true
        } catch {
            items = [["type": "inputText", "text": error.localizedDescription]]
            success = false
        }
        let text = items.compactMap { $0["text"] }.joined(separator: "\n")
        let preview = text.isEmpty ? "Image attached." : text
        let limit = ClaudeStream.maxToolOutput
        onEvent(.toolResult(id: id, output: preview.count > limit ? String(preview.prefix(limit)) + "\n…" : preview,
                            isError: !success))
        return ["success": success, "contentItems": items]
    }

    private static func contentItems(_ result: [String: Any]) -> [[String: String]] {
        let content = result["content"] as? [[String: Any]] ?? []
        var items: [[String: String]] = []
        for item in content {
            switch item["type"] as? String {
            case "text":
                if let text = item["text"] as? String { items.append(["type": "inputText", "text": text]) }
            case "image":
                if let mime = item["mimeType"] as? String, ["image/png", "image/jpeg", "image/webp", "image/gif"].contains(mime),
                   let data = item["data"] as? String, data.utf8.count <= 12 * 1024 * 1024 {
                    items.append(["type": "inputImage", "imageUrl": "data:\(mime);base64,\(data)"])
                }
            default: break
            }
        }
        if items.isEmpty, let structured = result["structuredContent"],
           let data = try? JSONSerialization.data(withJSONObject: structured, options: [.fragmentsAllowed]),
           let text = String(data: data, encoding: .utf8) {
            items.append(["type": "inputText", "text": text])
        }
        return items.isEmpty ? [["type": "inputText", "text": "The Box tool returned no output."]] : items
    }
}
