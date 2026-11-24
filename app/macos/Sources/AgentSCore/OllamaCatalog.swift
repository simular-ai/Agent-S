import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// The public model library. The local Ollama service lists installed models separately.
public actor OllamaCatalog {
    public static let shared = OllamaCatalog()
    public static let libraryURL = URL(string: "https://ollama.com/library")!

    public struct Variant: Identifiable, Hashable, Sendable {
        public let name: String
        public let size: String?
        public var id: String { name }
    }

    enum Failure: LocalizedError {
        case invalidFamily
        case unavailable
        case unexpectedLayout

        var errorDescription: String? {
            switch self {
            case .invalidFamily: "That isn’t a valid Ollama library model name."
            case .unavailable: "The Ollama model library is unavailable."
            case .unexpectedLayout: "Couldn’t read the Ollama model library."
            }
        }
    }

    private struct Cached<Value> {
        let value: Value
        private let expires = ContinuousClock.now.advanced(by: .seconds(600))

        var freshValue: Value? { ContinuousClock.now < expires ? value : nil }
    }

    /// The model pages are addressed directly. Do not follow redirects to other pages or hosts.
    private final class NoRedirects: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest,
                        completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
            completionHandler(nil)
        }
    }

    private let session: URLSession
    private var library: Cached<[String]>?
    private var tags: [String: Cached<[Variant]>] = [:]

    private init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 20
        configuration.timeoutIntervalForResource = 30
        session = URLSession(configuration: configuration, delegate: NoRedirects(), delegateQueue: nil)
    }

    public func families() async throws -> [String] {
        if let cached = library?.freshValue { return cached }
        let families = try await Self.parseFamilies(html(at: Self.libraryURL))
        library = Cached(value: families)
        return families
    }

    public func variants(for family: String) async throws -> [Variant] {
        let url = try Self.tagsURL(for: family)
        if let cached = tags[family]?.freshValue { return cached }
        let variants = try await Self.parseVariants(html(at: url), family: family)
        tags[family] = Cached(value: variants)
        return variants
    }

    private func html(at url: URL) async throws -> String {
        let (data, response) = try await session.data(from: url)
        guard let response = response as? HTTPURLResponse,
              response.statusCode == 200, response.url == url else { throw Failure.unavailable }
        guard response.mimeType == "text/html", let html = String(data: data, encoding: .utf8) else {
            throw Failure.unexpectedLayout
        }
        return html
    }

    static func tagsURL(for family: String) throws -> URL {
        guard family.wholeMatch(of: identifier) != nil else { throw Failure.invalidFamily }
        return libraryURL.appendingPathComponent(family).appendingPathComponent("tags")
    }

    private static var identifier: Regex<Substring> { #/[A-Za-z0-9][A-Za-z0-9._-]*/# }

    static func parseFamilies(_ html: String) throws -> [String] {
        // The library is a single result list. Exclude links elsewhere in the page, such as navigation.
        guard let listing = html.firstMatch(of: #/<div\b[^>]*\sid=["']repo["'][^>]*>\s*<ul\b[^>]*>([\s\S]*?)<\/ul>/#)?.1 else {
            throw Failure.unexpectedLayout
        }
        let links = listing.matches(of: #/<a\b[^>]*\shref=["']\/library\/([A-Za-z0-9][A-Za-z0-9._-]*)["']/#)
        var seen: Set<String> = []
        let families = links.map { String($0.1) }.filter { seen.insert($0).inserted }
        guard !families.isEmpty else { throw Failure.unexpectedLayout }
        return families
    }

    static func parseVariants(_ html: String, family: String) throws -> [Variant] {
        guard let countText = html.firstMatch(of: #/<p\b[^>]*>\s*(\d+)\s+models?\s*<\/p>/#)?.1,
              let expectedCount = Int(countText), expectedCount > 0 else { throw Failure.unexpectedLayout }
        let libraryPath = "/library/\(family)"
        var seen: Set<String> = []
        var variants: [Variant] = []

        // Each model has duplicate mobile and desktop markup. The mobile link contains the name,
        // cloud marker and download size for one row, so read it once and keep the page order.
        for link in html.matches(of: #/<a\b([^>]*)>([\s\S]*?)<\/a\s*>/#) {
            let attributes = link.1
            guard let classes = attributes.firstMatch(of: #/\sclass=["']([^"']+)["']/#)?.1,
                  classes.split(whereSeparator: \.isWhitespace).contains("md:hidden"),
                  let path = attributes.firstMatch(of: #/\shref=["']([^"']+)["']/#)?.1 else { continue }
            let name: String
            let tag: Substring
            if path == libraryPath {
                name = family
                tag = ""
            } else {
                let prefix = libraryPath + ":"
                guard path.hasPrefix(prefix) else { continue }
                tag = path.dropFirst(prefix.count)
                guard tag.wholeMatch(of: identifier) != nil else { continue }
                name = "\(family):\(tag)"
            }
            guard seen.insert(name).inserted else { continue }
            let body = link.2
            guard !tag.lowercased().split(separator: "-").contains("cloud"), !isHosted(body) else { continue }
            variants.append(Variant(name: name, size: modelSize(in: body)))
        }

        // A verified page with only hosted entries legitimately has no downloadable variants.
        guard seen.count == expectedCount else { throw Failure.unexpectedLayout }
        return variants
    }

    private static func isHosted(_ body: Substring) -> Bool {
        body.contains(#/•\s*[A-Za-z ]+\s+Usage\s*•/#)
            || body.contains(#/\s(?:title|aria-label)=["'][^"']*[Cc]loud[^"']*["']/#)
            || body.contains(#/<title>\s*[Cc]loud\s*<\/title>/#)
    }

    private static func modelSize(in body: Substring) -> String? {
        let text = body.replacingOccurrences(of: "&nbsp;", with: " ")
        guard let match = text.firstMatch(of: #/•\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)(?:\s*[-–]\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B))?\s*•/#) else {
            return nil
        }
        let first = "\(match.1) \(match.2)"
        guard let last = match.3, let unit = match.4 else { return first }
        return "\(first) – \(last) \(unit)"
    }
}
