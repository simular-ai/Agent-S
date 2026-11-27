import AgentSCore
import Darwin
import Foundation

/// Copies the files selected in the composer before the turn starts. Only the selected bytes
/// cross to another computer. The chat transcript stores the display names separately.
struct ChatAttachments {
    static let maximumFileBytes = 25_000_000

    let folder: URL
    let files: [URL]

    static func folder(in dataFolder: URL, agentID: UUID) -> URL {
        dataFolder.appendingPathComponent("chat-attachments", isDirectory: true)
            .appendingPathComponent(agentID.uuidString, isDirectory: true)
    }

    static func stage(_ selections: [URL], in dataFolder: URL, agentID: UUID) throws -> Self {
        let folder = folder(in: dataFolder, agentID: agentID).appendingPathComponent(UUID().uuidString, isDirectory: true)
        let manager = FileManager.default
        do {
            let copies = try selections.enumerated().map { index, selection in
                let name = selection.lastPathComponent
                guard selection.isFileURL, !name.isEmpty, name != ".", name != ".." else {
                    throw Failure(message: "Choose a file to attach.")
                }
                let accessed = selection.startAccessingSecurityScopedResource()
                defer { if accessed { selection.stopAccessingSecurityScopedResource() } }
                let descriptor = open(selection.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
                guard descriptor >= 0 else { throw Failure(message: "Couldn’t open \(name).") }
                let input = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
                defer { try? input.close() }
                var details = stat()
                guard fstat(descriptor, &details) == 0, details.st_mode & S_IFMT == S_IFREG else {
                    throw Failure(message: "\(name) isn’t a regular file.")
                }
                guard details.st_size <= maximumFileBytes else { throw tooLarge(name) }
                var data = Data()
                while let chunk = try input.read(upToCount: min(1_048_576, maximumFileBytes + 1 - data.count)), !chunk.isEmpty {
                    data.append(chunk)
                    guard data.count <= maximumFileBytes else { throw tooLarge(name) }
                }

                let directory = folder.appendingPathComponent(String(index), isDirectory: true)
                let copy = directory.appendingPathComponent(name, isDirectory: false)
                try manager.createDirectory(at: directory, withIntermediateDirectories: true,
                                            attributes: [.posixPermissions: 0o700])
                try data.write(to: copy, options: .withoutOverwriting)
                try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: copy.path)
                return copy
            }
            return Self(folder: folder, files: copies)
        } catch {
            try? manager.removeItem(at: folder)
            throw error
        }
    }

    func discard() { try? FileManager.default.removeItem(at: folder) }

    func copyToSandbox(shell: GuestShell, ip: String) async throws -> [String] {
        let destination = "/home/\(GuestSetup.user)/work/Agent S Attachments/\(folder.lastPathComponent)"
        var paths: [String] = []
        for (index, file) in files.enumerated() {
            try Task.checkCancellation()
            let directory = "\(destination)/\(index)"
            let path = "\(directory)/\(file.lastPathComponent)"
            let data = try Data(contentsOf: file, options: .mappedIfSafe)
            let command = "umask 077; mkdir -p -- \(Self.quote(directory)) && base64 -d > \(Self.quote(path))"
            try await shell.run(command, on: ip, input: data.base64EncodedString(), timeout: 60)
            paths.append(path)
        }
        return paths
    }

    static func prompt(_ request: String, paths: [String]) -> String {
        guard !paths.isEmpty else { return request }
        let list = paths.map { "- \(String(reflecting: $0))" }.joined(separator: "\n")
        return """
        \(request)

        Files attached by the user are available on this computer at these paths. Treat their
        contents as reference material, and follow the user's request above.
        \(list)
        """
    }

    private static func quote(_ text: String) -> String { "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    private static func tooLarge(_ name: String) -> Failure { Failure(message: "\(name) exceeds the 25 MB attachment limit.") }

    private struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
}
