import Foundation

/// Executes only commands already approved by Agent S. One instance belongs to one turn.
@MainActor
final class HarnessMacCommand {
    private var process: Process?
    private var stopped = false

    func run(_ command: String) async throws -> (text: String, success: Bool, image: Data?) {
        try checkRunning()
        guard process == nil else { throw HarnessMacCommandBusy() }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("agent-s-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: temporary) }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", command]
        process.environment = ["HOME": home.path, "PATH": MacAccess.path, "TMPDIR": temporary.path]
        process.currentDirectoryURL = home
        process.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        let capture = HarnessMacOutput()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                capture.finish()
            } else { capture.append(data) }
        }
        process.standardOutput = pipe
        process.standardError = pipe
        self.process = process
        defer {
            pipe.fileHandleForReading.readabilityHandler = nil
            self.process = nil
        }
        let status = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Int32, Error>) in
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
        try checkRunning()
        let screenshot = temporary.appendingPathComponent("agent-s-screen.png")
        if status == 0, command.trimmingCharacters(in: .whitespaces) == "\(MacAccess.macTool) screenshot",
           let size = try? screenshot.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 8 * 1024 * 1024,
           let image = try? Data(contentsOf: screenshot) {
            return ("Screenshot captured.", true, image)
        }
        let text = await capture.read()
        try checkRunning()
        return (text.isEmpty ? "Command exited with status \(status)." : text, status == 0, nil)
    }

    func stop() {
        stopped = true
        if let process, process.isRunning { process.terminate() }
    }

    private func checkRunning() throws {
        try Task.checkCancellation()
        if stopped { throw CancellationError() }
    }
}

private struct HarnessMacCommandBusy: LocalizedError {
    var errorDescription: String? { "This agent is already running a command on the Mac." }
}

/// Drain the approved command’s output without retaining more than the tool can use.
private final class HarnessMacOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes = Data()
    private var truncated = false
    private let limit = 64 * 1024
    private var done = false
    private var waiter: CheckedContinuation<String, Never>?

    func append(_ data: Data) {
        lock.lock()
        defer { lock.unlock() }
        let remaining = limit - bytes.count
        bytes.append(data.prefix(remaining))
        if data.count > remaining { truncated = true }
    }

    func read() async -> String {
        await withCheckedContinuation { continuation in
            lock.lock()
            if done {
                let text = self.text
                lock.unlock()
                continuation.resume(returning: text)
            } else {
                waiter = continuation
                lock.unlock()
                // A command may have started a background process that still holds the pipe.
                DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(1)) { self.finish() }
            }
        }
    }

    func finish() {
        lock.lock()
        guard !done else { lock.unlock(); return }
        done = true
        let waiter = self.waiter
        self.waiter = nil
        let text = self.text
        lock.unlock()
        waiter?.resume(returning: text)
    }

    private var text: String {
        let result = String(decoding: bytes, as: UTF8.self)
        return truncated ? result + "\n[Output truncated]" : result
    }
}
