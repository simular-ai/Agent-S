import Foundation

public enum AgentMemoryKind: String, CaseIterable, Codable, Sendable {
    case activeWork
    case preference
    case decision
    case workflow

    public var displayName: String {
        switch self {
        case .activeWork: "Active work"
        case .preference: "Preferences"
        case .decision: "Decisions"
        case .workflow: "Workflows"
        }
    }
}

public enum AgentMemoryStatus: String, Codable, Sendable {
    case open
    case resolved
}

/// The computer and inference route that may receive a note.
public struct AgentMemoryRoute: Equatable, Codable, Sendable {
    public enum OllamaInference: String, Codable, Sendable {
        case local
        case hosted
        case unconfirmed
    }

    public let privacy: PrivacyMode
    public let harness: Harness
    public let provider: ModelProvider?
    public let computer: CloudComputer?
    public let target: String?
    public let model: String?
    public let ollamaInference: OllamaInference?

    public init(privacy: PrivacyMode, harness: Harness, provider: ModelProvider?, computer: CloudComputer? = nil,
                boxName: String? = nil, saiMachine: String? = nil, ollamaInference: OllamaInference = .unconfirmed,
                model: String? = nil) {
        self.privacy = privacy
        self.harness = harness
        self.provider = harness == .sai ? nil : provider
        self.computer = privacy == .remote ? (computer ?? (harness == .sai ? .sai : .box)) : nil
        switch self.computer {
        case .box: target = Self.nonempty(boxName)
        case .sai: target = Self.nonempty(saiMachine)
        case nil: target = nil
        }
        self.model = self.provider?.usesAccount == true ? nil : Self.nonempty(model)
        self.ollamaInference = self.provider == .ollama ? ollamaInference : nil
    }

    /// Stable across app launches and distinct for each computer, provider, and model.
    public var continuationKey: String {
        let parts = [privacy.rawValue, harness.rawValue, provider?.rawValue, computer?.rawValue,
                     target, model, ollamaInference?.rawValue]
        return parts.map { part in
            part.map { "\($0.utf8.count):\($0)" } ?? "-"
        }.joined(separator: "/")
    }

    public var allowsMemory: Bool {
        if harness == .sai { return privacy == .remote && computer == .sai && target != nil }
        guard let provider else { return false }
        return ModelProvider.allowed(in: privacy, harness: harness, computer: computer).contains(provider)
    }

    public var allowsAutomaticRecall: Bool {
        allowsMemory && (provider != .ollama || ollamaInference == .local || ollamaInference == .hosted)
    }

    public var displayName: String {
        let location = switch computer {
        case .box: target.map { "Simular Box · \($0)" } ?? "Simular Box"
        case .sai: "Sai computer"
        case nil: privacy.location.displayName
        }
        let inference = switch ollamaInference {
        case .local: "Local Ollama"
        case .hosted: "Hosted Ollama"
        case .unconfirmed: "Ollama"
        case nil: provider?.displayName
        }
        return [location, harness.displayName, inference, model].compactMap { $0 }.joined(separator: " · ")
    }

    private static func nonempty(_ value: String?) -> String? {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        // The host uses exact identifiers for selected computers and model requests.
        return value
    }
}

public enum AgentMemoryScope: Equatable, Codable, Sendable {
    case `private`
    case route(AgentMemoryRoute)

    public var displayName: String {
        switch self {
        case .private: "Private"
        case let .route(route): route.displayName
        }
    }
}

public struct AgentMemoryNote: Equatable, Codable, Identifiable, Sendable {
    public let id: UUID
    public fileprivate(set) var kind: AgentMemoryKind
    public fileprivate(set) var text: String
    public fileprivate(set) var status: AgentMemoryStatus
    public fileprivate(set) var sourceChatItemID: String?
    public fileprivate(set) var updatedAt: Date
    public fileprivate(set) var isPinned: Bool
    public fileprivate(set) var scope: AgentMemoryScope

    fileprivate init(kind: AgentMemoryKind, text: String, sourceChatItemID: String?, now: Date, scope: AgentMemoryScope) {
        id = UUID()
        self.kind = kind
        self.text = text
        status = .open
        self.sourceChatItemID = sourceChatItemID
        updatedAt = now
        isPinned = false
        self.scope = scope
    }
}

public enum AgentMemoryOperation: Equatable, Sendable {
    case add(kind: AgentMemoryKind, text: String)
    case update(id: UUID, kind: AgentMemoryKind?, text: String)
    case resolve(id: UUID)
    case supersede(id: UUID, kind: AgentMemoryKind, text: String)
}

public enum AgentMemoryError: LocalizedError, Equatable {
    case staleRevision
    case tooManyOperations
    case invalidNote
    case duplicateNote
    case invalidSource
    case noteLimit
    case unavailableNote
    case unavailableRoute

    public var errorDescription: String? {
        switch self {
        case .staleRevision: "Memory changed during the turn."
        case .tooManyOperations: "Too many memory changes in one turn."
        case .invalidNote: "Enter a memory note of up to \(AgentMemory.maxNoteBytes.formatted()) bytes."
        case .duplicateNote: "A matching memory note already exists."
        case .invalidSource: "The memory note has an invalid source."
        case .noteLimit: "Memory holds up to \(AgentMemory.maxNoteCount.formatted()) notes."
        case .unavailableNote: "The memory note is unavailable for this change."
        case .unavailableRoute: "Memory sharing requires a known model and computer route."
        }
    }
}

/// A separate document for one agent. A turn may commit proposed changes only against its input revision.
public struct AgentMemory: Equatable, Codable, Sendable {
    public static let currentVersion = 1
    public static let maxNoteCount = 256
    public static let maxNoteBytes = 1_024
    public static let maxOperationCount = 8
    public static let maxContextBytes = 4_096

    public let version: Int
    public private(set) var revision: UUID
    public private(set) var notes: [AgentMemoryNote]

    public init() {
        version = Self.currentVersion
        revision = UUID()
        notes = []
    }

    private enum CodingKeys: String, CodingKey { case version, revision, notes }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        version = try values.decode(Int.self, forKey: .version)
        guard version == Self.currentVersion else {
            throw DecodingError.dataCorruptedError(forKey: .version, in: values, debugDescription: "Unsupported memory version.")
        }
        revision = try values.decode(UUID.self, forKey: .revision)
        notes = try values.decode([AgentMemoryNote].self, forKey: .notes)
        guard notes.count <= Self.maxNoteCount, Set(notes.map(\.id)).count == notes.count,
              notes.allSatisfy({ (try? Self.validatedText($0.text)) == $0.text && Self.isValidSource($0.sourceChatItemID) }) else {
            throw DecodingError.dataCorruptedError(forKey: .notes, in: values, debugDescription: "Invalid memory notes.")
        }
    }

    public func notes(for route: AgentMemoryRoute) -> [AgentMemoryNote] {
        guard route.allowsMemory else { return [] }
        return notes.filter { $0.scope == .route(route) }.sorted(by: Self.precedes)
    }

    public func notes(in kind: AgentMemoryKind) -> [AgentMemoryNote] {
        notes.filter { $0.kind == kind }.sorted(by: Self.precedes)
    }

    private static func precedes(_ first: AgentMemoryNote, _ second: AgentMemoryNote) -> Bool {
        if first.isPinned != second.isPinned { return first.isPinned }
        if first.updatedAt != second.updatedAt { return first.updatedAt > second.updatedAt }
        return first.id.uuidString < second.id.uuidString
    }

    /// JSON references for the model, bounded after encoding and scoped to its current route.
    public func context(for route: AgentMemoryRoute, matching message: String = "") -> String? {
        guard route.allowsAutomaticRecall else { return nil }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let terms = Self.searchTerms(String(message.prefix(4_096)))
        let candidates = notes(for: route).filter { $0.status == .open }
        let ranked = candidates.map { note in
            (note, Self.searchTerms(note.text).intersection(terms).count)
        }.sorted { first, second in
            if first.0.isPinned != second.0.isPinned { return first.0.isPinned }
            if first.1 != second.1 { return first.1 > second.1 }
            return Self.precedes(first.0, second.0)
        }
        var entries: [String] = []
        var byteCount = 2 // array brackets
        for (note, _) in ranked {
            let reference = ["id": note.id.uuidString, "kind": note.kind.rawValue, "text": note.text]
            guard let item = try? encoder.encode(reference) else { continue }
            let length = item.count + (entries.isEmpty ? 0 : 1)
            guard byteCount + length <= Self.maxContextBytes else { continue }
            entries.append(String(decoding: item, as: UTF8.self))
            byteCount += length
        }
        return entries.isEmpty ? nil : "[" + entries.joined(separator: ",") + "]"
    }

    private static func searchTerms(_ text: String) -> Set<Substring> {
        Set(text.lowercased().split { !$0.isLetter && !$0.isNumber }.filter { $0.count >= 3 })
    }

    @discardableResult
    public mutating func apply(_ operations: [AgentMemoryOperation], expecting revision: UUID, route: AgentMemoryRoute,
                               sourceChatItemID: String? = nil, now: Date = .now) throws -> Bool {
        guard revision == self.revision else { throw AgentMemoryError.staleRevision }
        guard route.allowsMemory else { throw AgentMemoryError.unavailableRoute }
        guard operations.count <= Self.maxOperationCount else { throw AgentMemoryError.tooManyOperations }
        guard Self.isValidSource(sourceChatItemID) else { throw AgentMemoryError.invalidSource }
        var candidate = notes
        let scope = AgentMemoryScope.route(route)
        for operation in operations {
            switch operation {
            case let .add(kind, text):
                try Self.add(kind: kind, text: text, with: sourceChatItemID, scope: scope, now: now, to: &candidate)
            case let .update(id, kind, text):
                let index = try Self.scopedIndex(id: id, in: candidate, scope: scope)
                let text = try Self.validatedText(text)
                let kind = kind ?? candidate[index].kind
                guard candidate[index].text != text || candidate[index].kind != kind else { continue }
                guard !candidate.contains(where: { $0.id != id && Self.matches($0, kind: kind, text: text, scope: scope) }) else {
                    throw AgentMemoryError.duplicateNote
                }
                candidate[index].kind = kind
                candidate[index].text = text
                candidate[index].sourceChatItemID = sourceChatItemID
                candidate[index].updatedAt = now
            case let .resolve(id):
                let index = try Self.scopedIndex(id: id, in: candidate, scope: scope)
                Self.resolve(&candidate[index], sourceChatItemID: sourceChatItemID, now: now)
            case let .supersede(id, kind, text):
                let index = try Self.scopedIndex(id: id, in: candidate, scope: scope)
                let text = try Self.validatedText(text)
                guard candidate[index].text != text || candidate[index].kind != kind else { continue }
                try Self.add(kind: kind, text: text, with: sourceChatItemID, scope: scope, now: now, to: &candidate)
                Self.resolve(&candidate[index], sourceChatItemID: sourceChatItemID, now: now)
            }
        }
        return commit(candidate)
    }

    @discardableResult
    public mutating func saveUserNote(id: UUID? = nil, kind: AgentMemoryKind, text: String,
                                      route: AgentMemoryRoute? = nil, now: Date = .now) throws -> Bool {
        let text = try Self.validatedText(text)
        if let route, !route.allowsMemory { throw AgentMemoryError.unavailableRoute }
        var candidate = notes
        if let id {
            guard let index = candidate.firstIndex(where: { $0.id == id }) else { throw AgentMemoryError.unavailableNote }
            let scope = route.map(AgentMemoryScope.route) ?? candidate[index].scope
            let changedText = candidate[index].text != text || candidate[index].kind != kind
            guard changedText || candidate[index].scope != scope else { return false }
            if changedText { candidate[index].sourceChatItemID = nil }
            candidate[index].kind = kind
            candidate[index].text = text
            candidate[index].scope = scope
            candidate[index].updatedAt = now
        } else {
            try Self.add(kind: kind, text: text, with: nil, scope: route.map(AgentMemoryScope.route) ?? .private,
                         now: now, to: &candidate)
        }
        return commit(candidate)
    }

    @discardableResult
    public mutating func setScope(_ scope: AgentMemoryScope, id: UUID, now: Date = .now) throws -> Bool {
        if case let .route(route) = scope, !route.allowsMemory { throw AgentMemoryError.unavailableRoute }
        guard let index = notes.firstIndex(where: { $0.id == id }) else { throw AgentMemoryError.unavailableNote }
        guard notes[index].scope != scope else { return false }
        notes[index].scope = scope
        notes[index].updatedAt = now
        revision = UUID()
        return true
    }

    @discardableResult
    public mutating func setPinned(_ isPinned: Bool, id: UUID, now: Date = .now) -> Bool {
        guard let index = notes.firstIndex(where: { $0.id == id }), notes[index].isPinned != isPinned else { return false }
        notes[index].isPinned = isPinned
        notes[index].updatedAt = now
        revision = UUID()
        return true
    }

    @discardableResult
    public mutating func setStatus(_ status: AgentMemoryStatus, id: UUID, now: Date = .now) -> Bool {
        guard let index = notes.firstIndex(where: { $0.id == id }), notes[index].status != status else { return false }
        notes[index].status = status
        notes[index].updatedAt = now
        revision = UUID()
        return true
    }

    @discardableResult
    public mutating func forget(id: UUID) -> Bool {
        guard let index = notes.firstIndex(where: { $0.id == id }) else { return false }
        notes.remove(at: index)
        revision = UUID()
        return true
    }

    private mutating func commit(_ candidate: [AgentMemoryNote]) -> Bool {
        guard notes != candidate else { return false }
        notes = candidate
        revision = UUID()
        return true
    }

    private static func scopedIndex(id: UUID, in notes: [AgentMemoryNote], scope: AgentMemoryScope) throws -> Int {
        guard let index = notes.firstIndex(where: { $0.id == id && $0.scope == scope && $0.status == .open }) else {
            throw AgentMemoryError.unavailableNote
        }
        return index
    }

    private static func add(kind: AgentMemoryKind, text: String, with sourceChatItemID: String?, scope: AgentMemoryScope,
                            now: Date, to notes: inout [AgentMemoryNote]) throws {
        let text = try validatedText(text)
        guard !notes.contains(where: { matches($0, kind: kind, text: text, scope: scope) }) else { return }
        guard notes.count < maxNoteCount else { throw AgentMemoryError.noteLimit }
        notes.append(AgentMemoryNote(kind: kind, text: text, sourceChatItemID: sourceChatItemID, now: now, scope: scope))
    }

    private static func matches(_ note: AgentMemoryNote, kind: AgentMemoryKind, text: String, scope: AgentMemoryScope) -> Bool {
        note.status == .open && note.kind == kind && note.text == text && note.scope == scope
    }

    private static func resolve(_ note: inout AgentMemoryNote, sourceChatItemID: String?, now: Date) {
        note.status = .resolved
        note.updatedAt = now
        note.sourceChatItemID = sourceChatItemID
    }

    private static func isValidSource(_ id: String?) -> Bool {
        guard let id else { return true }
        return !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && id.utf8.count <= 256
    }

    private static func validatedText(_ text: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.utf8.count <= maxNoteBytes else { throw AgentMemoryError.invalidNote }
        return trimmed
    }
}
