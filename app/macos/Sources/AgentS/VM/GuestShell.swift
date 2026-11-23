import Foundation
import AgentSCore

/// Talking to the guest over SSH with the app's own key.
///
/// The guest is found by its DHCP lease on the NAT network. Host keys are not pinned: the guest is
/// the app's own throwaway machine on a host-only network, and its key changes when it is rebuilt.
struct GuestShell {
    let files: GuestFiles
    let mac: String

    /// The guest's address from its DHCP lease: by MAC, else by the hostname cloud-init gave it
    /// (a guest that sends a DUID client ID is not listed under its MAC). The ARP table is no
    /// fallback: macOS hides it from this app.
    func address() -> String? {
        guard let leases = try? String(contentsOfFile: GuestNetwork.leasesPath, encoding: .utf8) else { return nil }
        return GuestNetwork.address(for: mac, hostname: GuestSetup.hostname, inLeases: leases)
    }

    /// Waits for the guest to have an address and answer SSH.
    func waitUntilReachable(timeout: TimeInterval = 300) async -> String? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let ip = address() {
                do {
                    try await run("true", on: ip)
                    return ip
                } catch {
                    NSLog("Agent S: SSH to the guest at %@ failed: %@", ip, "\(error)")
                }
            }
            try? await Task.sleep(nanoseconds: 2_000_000_000)
        }
        return nil
    }

    /// Whether first-boot setup (packages, Node, simulang, Claude Code) has finished.
    func isReady(on ip: String) async -> Bool {
        (try? await run("test -f \(GuestSetup.readyMarker)", on: ip)) != nil
    }

    /// `tunnelTo` forwards the same loopback port in the guest and on the Mac through SSH.
    /// Ollama and the OpenRouter proxy have separate ports so each route reaches its own provider.
    /// The app only connects out, so nothing on the Mac listens on the network (no firewall
    /// prompt, and only this session can use it).
    func sshArguments(ip: String, command: String, tunnelTo: UInt16? = nil) -> [String] {
        let tunnel = tunnelTo.map {
            ["-R", "127.0.0.1:\($0):127.0.0.1:\($0)", "-o", "ExitOnForwardFailure=yes"]
        } ?? []
        return tunnel + [
            "-i", files.sshKey.path,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=15",
            "\(GuestSetup.user)@\(ip)",
            command,
        ]
    }

    /// Runs a command to completion; throws on a non-zero exit.
    @discardableResult
    func run(_ command: String, on ip: String, input: String? = nil) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
            process.arguments = sshArguments(ip: ip, command: command)
            let out = Pipe()
            process.standardOutput = out
            process.standardError = out
            // Secrets travel on stdin, never on a command line the guest's process list shows.
            let stdin = Pipe()
            process.standardInput = input == nil ? FileHandle.nullDevice : stdin
            process.terminationHandler = { process in
                let text = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                if process.terminationStatus == 0 { continuation.resume(returning: text) }
                else { continuation.resume(throwing: GuestImage.Failure(message: text)) }
            }
            do {
                try process.run()
                if let input {
                    stdin.fileHandleForWriting.write(Data(input.utf8))
                    try? stdin.fileHandleForWriting.close()
                }
            } catch { continuation.resume(throwing: error) }
        }
    }

    /// Runs a command and returns its output and the process the caller can stop.
    func stream(_ command: String, on ip: String, tunnelTo: UInt16? = nil,
                standardInput: Pipe? = nil) throws -> (Process, TurnProcessOutput) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
        process.arguments = sshArguments(ip: ip, command: command, tunnelTo: tunnelTo)
        if let standardInput { process.standardInput = standardInput }
        return (process, try TurnProcessOutput(starting: process))
    }
}

/// Reads stdout until EOF independently of the exit notification so the last text delta is delivered.
struct TurnProcessOutput {
    enum Line: Sendable {
        case output(String)
        case rejected(String)
    }

    let lines: AsyncStream<Line>
    private let errors: AsyncStream<Data>
    private let statuses: AsyncStream<Int32>

    init(starting process: Process, inspect: @escaping @Sendable (String) -> Line = { .output($0) }) throws {
        let (lines, stdout) = AsyncStream<Line>.makeStream()
        let (errors, stderr) = AsyncStream<Data>.makeStream()
        let (statuses, exit) = AsyncStream<Int32>.makeStream()
        self.lines = lines
        self.errors = errors
        self.statuses = statuses

        let out = Pipe()
        let err = Pipe()
        process.standardOutput = out
        process.standardError = err
        let reader = LineReader { stdout.yield(inspect($0)) }
        out.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                reader.flush()
                stdout.finish()
            } else {
                reader.feed(data)
            }
        }
        err.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                stderr.finish()
            } else {
                stderr.yield(data)
            }
        }
        process.terminationHandler = { process in
            exit.yield(process.terminationStatus)
            exit.finish()
        }
        do {
            try process.run()
        } catch {
            out.fileHandleForReading.readabilityHandler = nil
            err.fileHandleForReading.readabilityHandler = nil
            throw error
        }
    }

    func completion() async -> (Int32, String) {
        var stderr = Data()
        for await data in errors { stderr.append(data) }
        var status = statuses.makeAsyncIterator()
        return (await status.next() ?? 1, String(decoding: stderr, as: UTF8.self))
    }
}

/// Splits a byte stream into lines, holding a partial line until its newline arrives.
final class LineReader: @unchecked Sendable {
    private var pending = Data()
    private let lock = NSLock()
    private let line: @Sendable (String) -> Void

    init(line: @escaping @Sendable (String) -> Void) { self.line = line }

    func feed(_ data: Data) {
        lock.lock()
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            lines.append(String(decoding: pending[pending.startIndex..<newline], as: UTF8.self))
            pending.removeSubrange(pending.startIndex...newline)
        }
        lock.unlock()
        lines.forEach(line)
    }

    func flush() {
        lock.lock()
        let rest = pending.isEmpty ? nil : String(decoding: pending, as: UTF8.self)
        pending.removeAll()
        lock.unlock()
        if let rest { line(rest) }
    }
}
