import Foundation
import AgentSCore

/// First-time setup of the guest's disk, key and cloud-init seed.
///
/// Debian publishes a ready-to-boot generic ARM64 cloud image (EFI, cloud-init); it is downloaded
/// once (~300 MB compressed), unpacked to a raw disk and grown; cloud-init resizes the root
/// partition to fill it on first boot.
enum GuestImage {
    static let imageURL = URL(string: "https://cloud.debian.org/images/cloud/trixie/latest/debian-13-generic-arm64.tar.xz")!
    static let diskSize: UInt64 = 32 * 1024 * 1024 * 1024

    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Prepares whatever is missing, reporting each step and the download's progress (0…1).
    static func prepare(files: GuestFiles,
                        step: @escaping @MainActor (String) -> Void,
                        progress: @escaping @MainActor (Double) -> Void) async throws {
        try files.ensureFolders()

        if !FileManager.default.fileExists(atPath: files.sshKey.path) {
            await step("Creating the app's key for the computer…")
            try run("/usr/bin/ssh-keygen", ["-t", "ed25519", "-N", "", "-C", "Agent S", "-f", files.sshKey.path])
        }

        if !FileManager.default.fileExists(atPath: files.disk.path) {
            await step("Downloading Debian (about 300 MB)…")
            let archive = try await download(imageURL, progress: progress)
            defer { try? FileManager.default.removeItem(at: archive) }
            await step("Unpacking the disk…")
            let work = files.root.appendingPathComponent("unpack", isDirectory: true)
            try? FileManager.default.removeItem(at: work)
            try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: work) }
            try run("/usr/bin/tar", ["-xJf", archive.path, "-C", work.path])
            guard let raw = try FileManager.default.contentsOfDirectory(atPath: work.path)
                .first(where: { $0.hasSuffix(".raw") }) else {
                throw Failure(message: "The Debian archive had no disk image in it.")
            }
            try FileManager.default.moveItem(at: work.appendingPathComponent(raw), to: files.disk)
            // Sparse: the file grows on disk only as the guest writes.
            let handle = try FileHandle(forWritingTo: files.disk)
            try handle.truncate(atOffset: diskSize)
            try handle.close()
        }

        if !FileManager.default.fileExists(atPath: files.seed.path) {
            await step("Writing the computer's first-boot setup…")
            try writeSeed(files: files)
        }
    }

    /// The cloud-init NoCloud seed: a small ISO labelled `cidata` with `user-data` and `meta-data`.
    static func writeSeed(files: GuestFiles) throws {
        let key = try String(contentsOf: files.sshPublicKey, encoding: .utf8)
        let folder = files.root.appendingPathComponent("seed", isDirectory: true)
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try GuestSetup.userData(sshPublicKey: key)
            .write(to: folder.appendingPathComponent("user-data"), atomically: true, encoding: .utf8)
        try GuestSetup.metaData(instanceId: UUID().uuidString)
            .write(to: folder.appendingPathComponent("meta-data"), atomically: true, encoding: .utf8)
        try GuestSetup.networkConfig
            .write(to: folder.appendingPathComponent("network-config"), atomically: true, encoding: .utf8)
        try? FileManager.default.removeItem(at: files.seed)
        try run("/usr/bin/hdiutil", ["makehybrid", "-quiet", "-iso", "-joliet",
                                     "-default-volume-name", "cidata",
                                     "-o", files.seed.path, folder.path])
    }

    private static func download(_ url: URL, progress: @escaping @MainActor (Double) -> Void) async throws -> URL {
        let (bytes, response) = try await URLSession.shared.bytes(from: url)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw Failure(message: "Downloading Debian failed (\((response as? HTTPURLResponse)?.statusCode ?? 0)).")
        }
        let total = max(response.expectedContentLength, 1)
        let target = FileManager.default.temporaryDirectory.appendingPathComponent("debian-\(UUID().uuidString).tar.xz")
        FileManager.default.createFile(atPath: target.path, contents: nil)
        let handle = try FileHandle(forWritingTo: target)
        defer { try? handle.close() }
        var buffer = Data()
        var written: Int64 = 0
        buffer.reserveCapacity(1 << 20)
        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 1 << 20 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                let fraction = Double(written) / Double(total)
                await progress(fraction)
            }
        }
        try handle.write(contentsOf: buffer)
        await progress(1)
        return target
    }

    @discardableResult
    static func run(_ tool: String, _ arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        let out = Pipe()
        process.standardOutput = out
        process.standardError = out
        try process.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let text = String(decoding: data, as: UTF8.self)
        guard process.terminationStatus == 0 else {
            throw Failure(message: "\((tool as NSString).lastPathComponent) failed: \(text.prefix(300))")
        }
        return text
    }
}
