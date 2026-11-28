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
        guard !files.isEmpty else { return [] }
        try Task.checkCancellation()
        let destination = "/home/\(GuestSetup.user)/.agent-s-attachments-\(UUID().uuidString)"
        let batch = Self.quote(destination)
        let owner = ".agent-s-owner"
        let token = Self.quote(UUID().uuidString)
        let enterBatch = "test ! -L \(batch) && cd -- \(batch) && test ! -L \(owner) && "
            + "test -f \(owner) && test \"$(cat -- \(owner))\" = \(token)"
        let create = """
        umask 077
        mkdir -m 700 -- \(batch) && cd -- \(batch) || exit $?
        set -C
        printf '%s\\n' \(token) > \(owner)
        """
        do {
            try await shell.run(create, on: ip, timeout: 15)
            var paths: [String] = []
            for (index, file) in files.enumerated() {
                try Task.checkCancellation()
                let relativePath = "\(index)/\(file.lastPathComponent)"
                let path = Self.quote(relativePath)
                let data = try Data(contentsOf: file, options: .mappedIfSafe)
                let command = """
                umask 077
                \(enterBatch) || exit 1
                set -C
                mkdir -m 700 -- \(index) || exit $?
                stat -c '%d:%i' -- \(index) > .directory-\(index) || exit $?
                test ! -e \(path) && test ! -L \(path) || exit 1
                exec 3> \(path) || exit $?
                stat -Lc '%d:%i' -- /proc/self/fd/3 > .file-\(index) || exit $?
                base64 -d >&3
                """
                try await shell.run(command, on: ip, input: data.base64EncodedString(), timeout: 60)
                paths.append("\(destination)/\(relativePath)")
            }
            try Task.checkCancellation()
            return paths
        } catch {
            // A cancellation or lost response can occur after mkdir. Verify this batch’s marker
            // before removing anything, including when the creation call did not return success.
            let cleanupFiles = files.enumerated().map { index, file in
                let path = Self.quote("\(index)/\(file.lastPathComponent)")
                return """
                if test ! -L \(index) && test -d \(index) &&
                   test ! -L .directory-\(index) && test -s .directory-\(index) &&
                   test "$(stat -c '%d:%i' -- \(index))" = "$(cat -- .directory-\(index))"; then
                    if test ! -L \(path) && test -f \(path) &&
                       test ! -L .file-\(index) && test -s .file-\(index) &&
                       test "$(stat -c '%d:%i' -- \(path))" = "$(cat -- .file-\(index))"; then
                        rm -f -- \(path)
                    fi
                    if rmdir -- \(index); then rm -f -- .directory-\(index) .file-\(index); fi
                fi
                """
            }.joined(separator: "\n")
            let cleanup = """
            \(enterBatch) || exit 0
            \(cleanupFiles)
            test . -ef \(batch) && rm -- \(owner) && rmdir -- \(batch)
            """
            // The transfer task may be canceled. Give cleanup its own bounded SSH request.
            await Task.detached {
                _ = try? await shell.run(cleanup, on: ip, timeout: 10)
            }.value
            throw error
        }
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
