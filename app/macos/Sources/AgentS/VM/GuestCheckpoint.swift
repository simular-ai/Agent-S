import Darwin
import Foundation

/// Holds exclusive use of the sandbox disk while this process is starting or running its VM.
final class GuestDiskLock {
    private let descriptor: Int32

    init(url: URL) throws {
        let fd = open(url.path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { throw GuestCheckpoint.fileError() }
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            let error: Error = errno == EWOULDBLOCK ? GuestCheckpoint.Failure.alreadyOpen : GuestCheckpoint.fileError()
            close(fd)
            throw error
        }
        descriptor = fd
    }

    deinit { close(descriptor) }
}

/// A single-use runtime checkpoint. The guest must stay paused while it is published.
@MainActor
struct GuestCheckpoint {
    private static let version = 1
    let files: GuestFiles

    enum Failure: LocalizedError {
        case alreadyOpen, incompatible, changedBackingFiles, notRunning, unsupported, alreadySaved
        case differentNetwork(SandboxNetworkMode)

        var errorDescription: String? {
            switch self {
            case .alreadyOpen: "The sandbox disk is already in use by another copy of Agent S."
            case .incompatible: "This saved sandbox state cannot be used with the current configuration."
            case .changedBackingFiles: "The sandbox disk or configuration files changed after the sandbox was saved."
            case .notRunning: "The sandbox must finish starting or stopping before it can be saved."
            case .unsupported: "This Mac cannot save and restore the sandbox's running state."
            case .alreadySaved: "A saved sandbox state already exists."
            case let .differentNetwork(mode):
                "The saved sandbox uses \(mode.title). Select it in Settings to resume, or discard the saved state to start with your new setting."
            }
        }
    }

    struct Metadata: Codable {
        let version: Int
        let memoryGB: Int
        let networkMode: SandboxNetworkMode?
        let backingFiles: [FileStamp]
    }

    struct FileStamp: Codable, Equatable {
        let device: Int32
        let inode: UInt64
        let size: Int64
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        let changedSeconds: Int?
        let changedNanoseconds: Int?

        init(url: URL, includeStatusChange: Bool = true) throws {
            var attributes = stat()
            guard lstat(url.path, &attributes) == 0 else { throw GuestCheckpoint.fileError() }
            guard (attributes.st_mode & S_IFMT) == S_IFREG else { throw Failure.changedBackingFiles }
            device = attributes.st_dev
            inode = attributes.st_ino
            size = attributes.st_size
            modifiedSeconds = attributes.st_mtimespec.tv_sec
            modifiedNanoseconds = attributes.st_mtimespec.tv_nsec
            changedSeconds = includeStatusChange ? attributes.st_ctimespec.tv_sec : nil
            changedNanoseconds = includeStatusChange ? attributes.st_ctimespec.tv_nsec : nil
        }
    }

    private var manager: FileManager { .default }
    var exists: Bool { manager.fileExists(atPath: files.checkpoint.path) }
    var stateURL: URL { files.checkpoint.appendingPathComponent("state.vzvmsave") }

    func metadata() throws -> Metadata {
        let url = files.checkpoint.appendingPathComponent("metadata.json")
        let metadata = try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: url))
        guard metadata.version == Self.version else { throw Failure.incompatible }
        guard try metadata.backingFiles == backingFiles() else { throw Failure.changedBackingFiles }
        return metadata
    }

    func save(memoryGB: Int, networkMode: SandboxNetworkMode, writeState: (URL) async throws -> Void) async throws {
        guard !exists else { throw Failure.alreadySaved }
        try clearStagingIfPresent()
        let staging = files.checkpointStaging
        let state = staging.appendingPathComponent("state.vzvmsave")
        try manager.createDirectory(at: staging, withIntermediateDirectories: false,
                                    attributes: [.posixPermissions: 0o700])
        defer { cleanStaging() }

        try await writeState(state)
        try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: state.path)
        let metadata = Metadata(version: Self.version, memoryGB: memoryGB, networkMode: networkMode,
                                backingFiles: try backingFiles())
        let metadataURL = staging.appendingPathComponent("metadata.json")
        try JSONEncoder().encode(metadata).write(to: metadataURL)
        try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: metadataURL.path)
        try move(staging, to: files.checkpoint)
    }

    /// Remove the checkpoint from the restore path atomically before a VM can write to its disk.
    func consume() throws {
        try clearStagingIfPresent()
        if exists { try move(files.checkpoint, to: files.checkpointStaging) }
    }

    /// A failed resume can keep the checkpoint if the VM stayed paused and the backing files still match.
    func retainAfterFailedResume(_ metadata: Metadata) throws {
        guard try metadata.backingFiles == backingFiles() else { throw Failure.changedBackingFiles }
        try move(files.checkpointStaging, to: files.checkpoint)
    }

    func cleanStaging() { try? clearStagingIfPresent() }

    private func backingFiles() throws -> [FileStamp] {
        // Virtualization's EFI attachment changes the variable store's status time on teardown.
        // Its identity, size, and content-modification time remain stable.
        try [FileStamp(url: files.disk), FileStamp(url: files.efiVariables, includeStatusChange: false),
             FileStamp(url: files.seed), FileStamp(url: files.machineIdentifier), FileStamp(url: files.macAddress)]
    }

    private func clearStagingIfPresent() throws {
        let staging = files.checkpointStaging
        if manager.fileExists(atPath: staging.path) { try manager.removeItem(at: staging) }
    }

    private func move(_ from: URL, to: URL) throws {
        guard rename(from.path, to.path) == 0 else { throw Self.fileError() }
    }

    nonisolated static func fileError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
