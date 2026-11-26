import Foundation
import Darwin
import AgentSCore

/// Talking to the guest over SSH with the app's own key.
///
/// The guest is found by its DHCP lease on the NAT network. Host keys are not pinned: the guest is
/// the app's own throwaway machine on a host-only network, and its key changes when it is rebuilt.
struct GuestShell {
    let files: GuestFiles
    let mac: String
    let forwardedSSHPort: UInt16?

    /// The guest's address from its DHCP lease: by MAC, else by the hostname cloud-init gave it
    /// (a guest that sends a DUID client ID is not listed under its MAC). The ARP table is no
    /// fallback: macOS hides it from this app.
    func address() -> String? {
        if forwardedSSHPort != nil { return "127.0.0.1" }
        guard let leases = try? String(contentsOfFile: GuestNetwork.leasesPath, encoding: .utf8) else { return nil }
        return GuestNetwork.address(for: mac, hostname: GuestSetup.hostname, inLeases: leases)
    }

    /// Waits for the guest to have an address and answer SSH.
    func waitUntilReachable(timeout: TimeInterval = 300) async -> String? {
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        while !Task.isCancelled {
            let remaining = deadline - ProcessInfo.processInfo.systemUptime
            guard remaining > 0 else { break }
            if let ip = address() {
                do {
                    try await run("true", on: ip, timeout: min(8, remaining))
                    return ip
                } catch is CancellationError {
                    return nil
                } catch {
                    NSLog("Agent S: SSH to the guest at %@ failed: %@", ip, "\(error)")
                }
            }
            let pause = min(2, deadline - ProcessInfo.processInfo.systemUptime)
            guard pause > 0 else { break }
            do { try await Task.sleep(for: .seconds(pause)) } catch { return nil }
        }
        return nil
    }

    /// Whether first-boot setup (packages, Node, simulang, Claude Code) has finished.
    func isReady(on ip: String) async throws -> Bool {
        do {
            try await run("test -f \(GuestSetup.readyMarker)", on: ip, timeout: 10)
            return true
        } catch let error as CommandFailure where error.status == 1 {
            return false
        }
    }

    /// `tunnelTo` forwards the same loopback port in the guest and on the Mac through SSH.
    /// Ollama and the OpenRouter proxy have separate ports so each route reaches its own provider.
    /// The app only connects out, so nothing on the Mac listens on the network (no firewall
    /// prompt, and only this session can use it).
    func sshArguments(ip: String, command: String, tunnelTo: UInt16? = nil) -> [String] {
        let tunnel = tunnelTo.map {
            ["-R", "127.0.0.1:\($0):127.0.0.1:\($0)", "-o", "ExitOnForwardFailure=yes"]
        } ?? []
        let port = forwardedSSHPort.map { ["-p", String($0)] } ?? []
        return tunnel + port + [
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

    struct CommandFailure: LocalizedError {
        let status: Int32
        let output: String

        var errorDescription: String? {
            output.isEmpty ? "The sandbox command exited with status \(status)." : output
        }
    }

    struct CommandTimeout: LocalizedError {
        var errorDescription: String? { "The sandbox command timed out." }
    }

    /// Recognizes the host SSH deadline and the exit statuses from GNU timeout in the guest.
    static func commandTimedOut(_ error: Error) -> Bool {
        let status = (error as? CommandFailure)?.status
        return error is CommandTimeout || status == 124 || status == 137
    }

    /// Runs a command to completion, with an optional wall deadline on the Mac.
    @discardableResult
    func run(_ command: String, on ip: String, input: String? = nil, timeout: TimeInterval? = nil) async throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
        process.arguments = sshArguments(ip: ip, command: command)
        return try await GuestCommand(process).run(input: input, timeout: timeout)
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

/// The subprocess, pipe callbacks, and deadline are all coordinated on one queue.
private final class GuestCommand: @unchecked Sendable {
    private let queue = DispatchQueue(label: "ai.simular.agent-s.guest-command", qos: .utility)
    private let process: Process
    private var continuation: CheckedContinuation<String, any Error>?
    private var reader: DispatchIO?
    private var writer: DispatchIO?
    private var deadline: DispatchWorkItem?
    private var escalation: DispatchWorkItem?
    private var output = Data()
    private var reachedEOF = false
    private var exitStatus: Int32?
    private var stopReason: (any Error)?

    init(_ process: Process) {
        self.process = process
    }

    func run(input: String?, timeout: TimeInterval?) async throws -> String {
        let deadline = timeout.flatMap { $0.isFinite ? DispatchTime.now() + max(0, $0) : nil }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async { self.start(continuation, input: input, deadline: deadline) }
            }
        } onCancel: {
            queue.async { self.stop(CancellationError()) }
        }
    }

    private func start(_ continuation: CheckedContinuation<String, any Error>, input: String?, deadline: DispatchTime?) {
        self.continuation = continuation
        if let stopReason {
            finish(.failure(stopReason))
            return
        }
        if let deadline, deadline <= .now() {
            finish(.failure(GuestShell.CommandTimeout()))
            return
        }

        let out = Pipe()
        process.standardOutput = out
        process.standardError = out
        // Secrets travel on stdin, never on a command line the guest's process list shows.
        let stdin = input.map { _ in Pipe() }
        process.standardInput = stdin ?? FileHandle.nullDevice
        if let handle = stdin?.fileHandleForWriting, fcntl(handle.fileDescriptor, F_SETNOSIGPIPE, 1) == -1 {
            finish(.failure(Self.posixError(errno)))
            return
        }
        process.terminationHandler = { [weak self] process in
            let status = process.terminationStatus
            self?.queue.async { [weak self] in self?.exited(status) }
        }
        do { try process.run() } catch {
            finish(.failure(error))
            return
        }

        reader = channel(out.fileHandleForReading)
        reader?.read(offset: 0, length: Int.max, queue: queue) { [weak self] done, data, error in
            self?.receivedOutput(data, done: done, error: error)
        }
        if let input, let handle = stdin?.fileHandleForWriting {
            writer = channel(handle)
            let data = Data(input.utf8).withUnsafeBytes { DispatchData(bytes: $0) }
            writer?.write(offset: 0, data: data, queue: queue) { [weak self] done, _, _ in
                if done { self?.closeInput() }
            }
        }
        if let deadline {
            let work = DispatchWorkItem { [weak self] in self?.stop(GuestShell.CommandTimeout()) }
            self.deadline = work
            queue.asyncAfter(deadline: deadline, execute: work)
        }
    }

    private func channel(_ handle: FileHandle) -> DispatchIO {
        DispatchIO(type: .stream, fileDescriptor: handle.fileDescriptor, queue: queue) { _ in
            try? handle.close()
        }
    }

    private func receivedOutput(_ data: DispatchData?, done: Bool, error: Int32) {
        guard continuation != nil, stopReason == nil else { return }
        if let data { output.append(contentsOf: data) }
        if error != 0 {
            stop(Self.posixError(error))
        } else if done {
            reachedEOF = true
            reader?.close()
            reader = nil
            completeIfReady()
        }
    }

    private func exited(_ status: Int32) {
        exitStatus = status
        closeInput()
        if let stopReason {
            finish(.failure(stopReason))
        } else {
            completeIfReady()
        }
    }

    private func completeIfReady() {
        guard let exitStatus, reachedEOF else { return }
        let text = String(decoding: output, as: UTF8.self)
        if exitStatus == 0 {
            finish(.success(text))
        } else {
            finish(.failure(GuestShell.CommandFailure(status: exitStatus, output: text)))
        }
    }

    private func stop(_ reason: any Error) {
        guard stopReason == nil else { return }
        stopReason = reason
        guard continuation != nil else { return }
        reader?.close(flags: .stop)
        reader = nil
        closeInput()
        guard exitStatus == nil else {
            finish(.failure(reason))
            return
        }
        if process.isRunning { process.terminate() }
        let work = DispatchWorkItem { [weak self] in self?.forceStop(reason) }
        escalation = work
        queue.asyncAfter(deadline: .now() + .milliseconds(250), execute: work)
    }

    private func forceStop(_ reason: any Error) {
        if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
        // Foundation still reaps the process if the kernel has delayed its exit.
        let work = DispatchWorkItem { [weak self] in self?.finish(.failure(reason)) }
        escalation = work
        queue.asyncAfter(deadline: .now() + .milliseconds(250), execute: work)
    }

    private func closeInput() {
        writer?.close(flags: .stop)
        writer = nil
    }

    private func finish(_ result: Result<String, any Error>) {
        guard let continuation else { return }
        self.continuation = nil
        deadline?.cancel()
        escalation?.cancel()
        continuation.resume(with: result)
    }

    private static func posixError(_ code: Int32) -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
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
