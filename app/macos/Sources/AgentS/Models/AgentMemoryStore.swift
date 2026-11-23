import Foundation
import AgentSCore

/// Agent S owns the memory file on the Mac, outside the folder shared with the guest.
struct AgentMemoryStore {
    struct Document: Codable {
        var notes = AgentMemory()
        var sessions: [String: String] = [:]
    }

    private let file: URL
    private static let maxFileSize = 2 * 1024 * 1024

    init(dataFolder: URL, agentID: UUID) {
        file = dataFolder.appendingPathComponent("memory", isDirectory: true)
            .appendingPathComponent(agentID.uuidString).appendingPathExtension("json")
    }

    func load() throws -> Document {
        guard FileManager.default.fileExists(atPath: file.path) else { return Document() }
        let reader = try FileHandle(forReadingFrom: file)
        defer { try? reader.close() }
        let data = try reader.read(upToCount: Self.maxFileSize + 1) ?? Data()
        guard data.count <= Self.maxFileSize else { throw CocoaError(.fileReadTooLarge) }
        return try JSONDecoder().decode(Document.self, from: data)
    }

    func save(_ document: Document) throws {
        let folder = file.deletingLastPathComponent()
        let manager = FileManager.default
        try manager.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: folder.path)
        let data = try JSONEncoder().encode(document)
        guard data.count <= Self.maxFileSize else { throw CocoaError(.fileWriteOutOfSpace) }
        try data.write(to: file, options: .atomic)
        try? manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
    }

    func delete() throws {
        if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
    }
}
