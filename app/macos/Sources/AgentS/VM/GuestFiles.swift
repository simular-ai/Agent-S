import AppKit

/// Where the guest's files live: `~/Library/Application Support/Agent S`.
struct GuestFiles {
    let root: URL

    /// The app's earlier identities (bundle ID, data folder), newest first.
    static let legacy = [("ai.simular.saige", "Saige"), ("ai.simular.sai-local", "SaiLocal")]

    /// An earlier build of the app is still running (and may have the computer's disk open).
    static var legacyAppRunning: Bool {
        legacy.contains { !NSRunningApplication.runningApplications(withBundleIdentifier: $0.0).isEmpty }
    }

    static let standard: GuestFiles = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let root = base.appendingPathComponent("Agent S", isDirectory: true)
        // Move the computer (disk, key, agents, shared folder) over from an earlier name, once,
        // and never while that app is running: its VM may be writing the disk.
        let fm = FileManager.default
        let hasComputer = { (folder: URL) in fm.fileExists(atPath: folder.appendingPathComponent("disk.img").path) }
        if !hasComputer(root), !legacyAppRunning,
           let old = legacy.map({ base.appendingPathComponent($0.1, isDirectory: true) })
            .first(where: { fm.fileExists(atPath: $0.path) }) {
            // A folder made before the move holds only defaults; the earlier one wins.
            if fm.fileExists(atPath: root.path), hasComputer(old) { try? fm.removeItem(at: root) }
            if !fm.fileExists(atPath: root.path) { try? fm.moveItem(at: old, to: root) }
        }
        return GuestFiles(root: root)
    }()

    var disk: URL { root.appendingPathComponent("disk.img") }
    /// Agents' headshots, copied in when chosen.
    var avatars: URL { root.appendingPathComponent("avatars", isDirectory: true) }
    var efiVariables: URL { root.appendingPathComponent("efi-variables") }
    var machineIdentifier: URL { root.appendingPathComponent("machine-id") }
    var macAddress: URL { root.appendingPathComponent("mac-address") }
    var seed: URL { root.appendingPathComponent("seed.iso") }
    var checkpoint: URL { root.appendingPathComponent("vm-checkpoint", isDirectory: true) }
    var checkpointStaging: URL { root.appendingPathComponent(".vm-checkpoint-staging", isDirectory: true) }
    var diskLock: URL { root.appendingPathComponent(".vm-disk-lock") }
    var sshKey: URL { root.appendingPathComponent("id_ed25519") }
    var sshPublicKey: URL { root.appendingPathComponent("id_ed25519.pub") }
    /// Shared with the guest at /mnt/sai.
    var share: URL { root.appendingPathComponent("share", isDirectory: true) }

    var isPrepared: Bool {
        [disk, seed, sshKey].allSatisfy { FileManager.default.fileExists(atPath: $0.path) }
    }

    func ensureFolders() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        // Chats, the SSH key and the disk live here: this user only.
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        try FileManager.default.createDirectory(at: share, withIntermediateDirectories: true)
    }
}
