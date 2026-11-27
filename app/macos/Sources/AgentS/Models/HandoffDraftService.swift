import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Generates handoff text without opening an agent session or supplying tools.
enum HandoffDraftService {
    enum Backend: Sendable {
        case onDevice
        case ollama(model: String)
        case openRouter(model: String, token: String)
    }

    enum Failure: LocalizedError {
        case unavailable, invalidResponse

        var errorDescription: String? {
            switch self {
            case .unavailable: "The drafting model is unavailable."
            case .invalidResponse: "The drafting model didn’t return a complete message."
            }
        }
    }

    private static let maxOutputTokens = 2048
    private static let maxDraftBytes = 16 * 1024

    static var onDeviceAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) { return SystemLanguageModel.default.isAvailable }
        #endif
        return false
    }

    static func draft(instructions: String, prompt: String, using backend: Backend) async throws -> String {
        try Task.checkCancellation()
        switch backend {
        case .onDevice:
            #if canImport(FoundationModels)
            if #available(macOS 26.0, *), onDeviceAvailable {
                let session = LanguageModelSession(model: .default, tools: []) { instructions }
                let text: String
                do {
                    text = try await session.respond(to: prompt).content
                } catch {
                    try Task.checkCancellation()
                    throw Failure.unavailable
                }
                try Task.checkCancellation()
                return try checkedText(text)
            }
            #endif
            throw Failure.unavailable
        case .ollama, .openRouter:
            let request = try request(instructions: instructions, prompt: prompt, using: backend)
            let data: Data
            let response: URLResponse
            do {
                (data, response) = try await session.data(for: request)
            } catch {
                try Task.checkCancellation()
                throw Failure.unavailable
            }
            try Task.checkCancellation()
            guard let response = response as? HTTPURLResponse, response.url == request.url,
                  (200..<300).contains(response.statusCode) else { throw Failure.unavailable }
            return try responseText(data, using: backend)
        }
    }

    static func request(instructions: String, prompt: String, using backend: Backend) throws -> URLRequest {
        let url: URL
        let model: String
        let token: String?
        let limits: [String: Any]
        switch backend {
        case .onDevice: throw Failure.unavailable
        case let .ollama(selectedModel):
            url = Ollama.base.appendingPathComponent("api/chat")
            model = selectedModel
            token = nil
            limits = ["options": ["num_predict": maxOutputTokens]]
        case let .openRouter(selectedModel, proxyToken):
            // These model selectors can enable server-side tools independently of the request's tool list.
            let modelID = selectedModel.lowercased()
            let variants = modelID.split(separator: ":")
            guard !modelID.contains("@preset/"), !variants.dropFirst().contains("online"),
                  variants.first != "openrouter/fusion" else { throw Failure.unavailable }
            url = URL(string: "http://127.0.0.1:\(OpenRouter.proxyPort)/v1/chat/completions")!
            model = selectedModel
            token = proxyToken
            limits = [
                "max_completion_tokens": maxOutputTokens,
                "modalities": ["text"],
                "tool_choice": "none",
                "plugins": [["id": "web", "enabled": false]],
            ]
        }
        let messages = [
            ["role": "system", "content": instructions],
            ["role": "user", "content": prompt],
        ]
        var body: [String: Any] = ["model": model, "stream": false, "messages": messages]
        body.merge(limits) { _, limit in limit }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return request
    }

    static func responseText(_ data: Data, using backend: Backend) throws -> String {
        guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              body["error"] == nil else { throw Failure.invalidResponse }
        let message: [String: Any]
        switch backend {
        case .onDevice: throw Failure.invalidResponse
        case .ollama:
            guard body["done"] as? Bool == true, body["done_reason"] as? String == "stop",
                  let reply = body["message"] as? [String: Any] else { throw Failure.invalidResponse }
            message = reply
        case .openRouter:
            guard let choices = body["choices"] as? [[String: Any]], choices.count == 1,
                  let choice = choices.first, choice["finish_reason"] as? String == "stop",
                  let reply = choice["message"] as? [String: Any] else { throw Failure.invalidResponse }
            message = reply
        }
        guard message["role"] as? String == "assistant", let text = message["content"] as? String,
              absentOrEmptyCalls(message["tool_calls"]),
              message["function_call"] == nil || message["function_call"] is NSNull,
              message["refusal"] == nil || message["refusal"] is NSNull else {
            throw Failure.invalidResponse
        }
        return try checkedText(text)
    }

    private static func absentOrEmptyCalls(_ value: Any?) -> Bool {
        value == nil || value is NSNull || (value as? [Any])?.isEmpty == true
    }

    private static func checkedText(_ text: String) throws -> String {
        let draft = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard draft.utf8.count <= maxDraftBytes, draft.unicodeScalars.contains(where: {
            CharacterSet.alphanumerics.contains($0) || $0.properties.isEmojiPresentation
        }) else {
            throw Failure.invalidResponse
        }
        return draft
    }

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 90
        configuration.timeoutIntervalForResource = 120
        configuration.connectionProxyDictionary = [:]
        return URLSession(configuration: configuration, delegate: NoRedirects(), delegateQueue: nil)
    }()

    private final class NoRedirects: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
            completionHandler(nil)
        }
    }
}
