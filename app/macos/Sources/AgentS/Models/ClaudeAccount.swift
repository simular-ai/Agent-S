import AppKit
import Darwin
import AgentSCore

/// The installed CLI owns authentication and stores its sign-in in Agent S's Claude Code profile.
enum ClaudeAccount {
    private static let path = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    private static let sandboxSSH = URL(fileURLWithPath: "/usr/bin/ssh")

    static func profile(in dataFolder: URL) -> URL {
        dataFolder.appendingPathComponent("claude-account", isDirectory: true)
    }

    static func accountPlan(claude: URL, profile: URL) async -> HarnessAccountPlan? {
        guard FileManager.default.fileExists(atPath: profile.path) else { return nil }
        let process = Process()
        let output = Pipe()
        process.executableURL = claude
        process.currentDirectoryURL = profile
        process.arguments = ["auth", "status", "--json"]
        process.environment = environment(profile: profile)
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        return await withCheckedContinuation { continuation in
            process.terminationHandler = { finished in
                let data = output.fileHandleForReading.readDataToEndOfFile()
                guard finished.terminationStatus == 0,
                      let status = try? JSONDecoder().decode(AuthStatus.self, from: data),
                      status.authMethod == "claude.ai" else { continuation.resume(returning: nil); return }
                continuation.resume(returning: HarnessAccountPlan(provider: .claudeAccount, reportedPlan: status.subscriptionType))
            }
            do { try process.run() }
            catch { continuation.resume(returning: nil) }
        }
    }

    @MainActor
    static func signIn(claude: URL, profile: URL) async throws {
        let variables = environment(profile: profile).sorted { $0.key < $1.key }
            .map { quote("\($0.key)=\($0.value)") }.joined(separator: " ")
        let command = #"exec /usr/bin/env -i \#(variables) TERM="${TERM:-xterm-256color}" \#(quote(claude.path)) auth login"#
        try await openTerminal(command: command, in: profile)
    }

    @MainActor
    static func signInInSandbox(shell: GuestShell, ip: String,
                                onBrowserApproval: @escaping @MainActor (URL) throws -> Void) async throws -> HarnessAccountPlan {
        try Task.checkCancellation()
        // Keep the control socket private and short enough for Darwin's Unix socket paths.
        var template = Array("/tmp/agent-s-claude.XXXXXX".utf8CString)
        guard let temporary = mkdtemp(&template) else { throw SandboxSignInError.failed }
        let directory = URL(fileURLWithPath: String(cString: temporary), isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let socket = directory.appendingPathComponent("ssh").path

        let command = shell.sshArguments(ip: ip, command: SandboxAccount.claudeBrowserSignInCommand)
        let destination = "\(GuestSetup.user)@\(ip)"
        let options = ["-F", "/dev/null", "-S", socket, "-o", "BatchMode=yes"]
        let session = Process()
        session.executableURL = sandboxSSH
        session.arguments = options + ["-T", "-M", "-o", "ControlPersist=no", "-o", "IdentitiesOnly=yes"] + command
        do {
            // The guest watches stdin EOF so closing this pipe cancels its login and releases its lock.
            let result = try await runSandboxSSH(session, timeout: .seconds(310), input: Pipe(), verifySignIn: {
                guard let plan = try await sandboxSignInPlan(shell: shell, ip: ip, requireFinishedLogin: false) else {
                    throw SandboxSignInError.statusUnavailable
                }
                return plan
            }) { approval in
                let port = approval.callbackPort
                // Claude listens on guest IPv4. Mac browsers can resolve localhost to either family.
                var listeners = ["-L", "127.0.0.1:\(port):127.0.0.1:\(port)"]
                if approval.callbackHost == "localhost" {
                    listeners += ["-L", "[::1]:\(port):127.0.0.1:\(port)"]
                }
                let forwarding = Process()
                forwarding.executableURL = sandboxSSH
                forwarding.arguments = options + ["-O", "forward", "-o", "ExitOnForwardFailure=yes"] + listeners + [destination]
                do {
                    guard try await runSandboxSSH(forwarding, timeout: .seconds(5)) == .exited(0), session.isRunning else {
                        throw SandboxSignInError.callbackUnavailable
                    }
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    throw SandboxSignInError.callbackUnavailable
                }
                try Task.checkCancellation()
                try onBrowserApproval(approval.authorizationURL)
            }
            switch result {
            case let .verified(plan): return plan
            case .exited(0): throw SandboxSignInError.statusUnavailable
            case .exited(75): throw SandboxSignInError.alreadyOpen
            case .exited(124), .exited(137): throw SandboxSignInError.expired
            case .exited: throw SandboxSignInError.failed
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // A closed or lingering transport can outlast a finished login. Check the saved
            // profile once more, without accepting credentials a running login may roll back.
            try Task.checkCancellation()
            if let plan = try await sandboxSignInPlan(shell: shell, ip: ip, requireFinishedLogin: true) { return plan }
            throw error
        }
    }

    private static func sandboxSignInPlan(shell: GuestShell, ip: String, requireFinishedLogin: Bool) async throws -> HarnessAccountPlan? {
        let command = requireFinishedLogin
            ? SandboxAccount.claudeCompletedSignInStatusCommand
            : SandboxAccount.statusCommand(for: .claudeAccount)
        let output = try? await shell.run(command, on: ip, timeout: 10)
        try Task.checkCancellation()
        return output.flatMap { SandboxAccount.plan(for: .claudeAccount, output: $0) }
    }

    private enum SandboxSSHResult: Equatable {
        case exited(Int32)
        case verified(HarnessAccountPlan)
    }

    private enum SandboxSSHEvent: Sendable {
        case approval(ClaudeBrowserSignIn)
        case completed
        case exit(Int32)
        case handoffDeadline
        case deadline
    }

    @MainActor
    private static func runSandboxSSH(
        _ process: Process, timeout: Duration, input: Pipe? = nil,
        verifySignIn: (@MainActor () async throws -> HarnessAccountPlan)? = nil,
        onApproval: (@MainActor (ClaudeBrowserSignIn) async throws -> Void)? = nil
    ) async throws -> SandboxSSHResult {
        try Task.checkCancellation()
        let (events, continuation) = AsyncStream<SandboxSSHEvent>.makeStream(bufferingPolicy: .bufferingNewest(2))
        // Raw CLI output can contain authorization material. Only consume the validated announcement.
        process.standardInput = input ?? FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        let output = onApproval == nil ? nil : Pipe()
        process.standardOutput = output ?? FileHandle.nullDevice
        if let output {
            let reader = LineReader { line in
                if line == ClaudeBrowserSignIn.completionAnnouncement {
                    continuation.yield(.completed)
                } else if let approval = ClaudeBrowserSignIn.parseAnnouncement(line) {
                    continuation.yield(.approval(approval))
                }
            }
            output.fileHandleForReading.readabilityHandler = { handle in
                let bytes = handle.availableData
                if bytes.isEmpty {
                    handle.readabilityHandler = nil
                    reader.flush()
                } else {
                    reader.feed(bytes)
                }
            }
        }
        process.terminationHandler = { finished in
            continuation.yield(.exit(finished.terminationStatus))
            continuation.finish()
        }
        func schedule(_ event: SandboxSSHEvent, after delay: Duration) -> Task<Void, Never> {
            Task { @MainActor in
                do { try await Task.sleep(for: delay) } catch { return }
                continuation.yield(event)
                continuation.finish()
            }
        }
        let deadline = schedule(.deadline, after: timeout)
        let handoffDeadline = onApproval == nil ? nil : schedule(.handoffDeadline, after: .seconds(20))
        defer {
            deadline.cancel()
            handoffDeadline?.cancel()
            continuation.finish()
            output?.fileHandleForReading.readabilityHandler = nil
            try? input?.fileHandleForWriting.close()
            if process.isRunning {
                // Let SSH deliver EOF before terminating the master when the guest is still reachable.
                let grace = input == nil ? 0 : 250
                DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + .milliseconds(grace)) {
                    if process.isRunning { process.terminate() }
                    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + .milliseconds(250)) {
                        if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) }
                    }
                }
            }
        }
        do { try process.run() } catch { throw SandboxSignInError.failed }

        var opened = false
        for await event in events {
            try Task.checkCancellation()
            switch event {
            case let .approval(approval):
                guard !opened, process.isRunning else { continue }
                opened = true
                handoffDeadline?.cancel()
                try await onApproval?(approval)
            case .completed:
                if let verifySignIn { return .verified(try await verifySignIn()) }
            case let .exit(status): return .exited(status)
            case .handoffDeadline: throw SandboxSignInError.handoffUnavailable
            case .deadline: throw SandboxSignInError.expired
            }
        }
        try Task.checkCancellation()
        throw SandboxSignInError.failed
    }

    private enum SandboxSignInError: LocalizedError {
        case failed, alreadyOpen, handoffUnavailable, callbackUnavailable, expired, statusUnavailable

        var errorDescription: String? {
            switch self {
            case .failed: "Claude Code sign-in didn’t complete. Try again."
            case .alreadyOpen: "Claude Code sign-in is already running. Close any previous sign-in, then try again."
            case .handoffUnavailable: "Claude Code didn’t provide a browser sign-in link. Try again."
            case .callbackUnavailable: "Couldn’t connect your browser to Claude Code in the sandbox. Try again."
            case .expired: "Claude Code sign-in expired. Try again."
            case .statusUnavailable: "Couldn’t verify Claude Code sign-in in the sandbox. Try again."
            }
        }
    }

    @MainActor
    private static func openTerminal(command: String, in directory: URL) async throws {
        try Task.checkCancellation()
        let workspace = NSWorkspace.shared
        guard let terminal = workspace.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else {
            throw CocoaError(.fileNoSuchFile)
        }

        let fileManager = FileManager.default
        let privatePermissions: [FileAttributeKey: Any] = [.posixPermissions: 0o700]
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: privatePermissions)
        try fileManager.setAttributes(privatePermissions, ofItemAtPath: directory.path)
        let launcher = directory.appendingPathComponent("Sign in to Claude Code.command")
        let script = """
        #!/bin/sh
        cd \(quote(directory.path)) || exit 1
        \(command)

        """
        try Data(script.utf8).write(to: launcher, options: .atomic)
        try fileManager.setAttributes(privatePermissions, ofItemAtPath: launcher.path)
        try Task.checkCancellation()
        _ = try await workspace.open([launcher], withApplicationAt: terminal, configuration: .init())
    }

    private struct AuthStatus: Decodable {
        let authMethod: String
        let subscriptionType: String?
    }

    private static func environment(profile: URL) -> [String: String] {
        [
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            "PATH": path,
            "CLAUDE_CONFIG_DIR": profile.path,
        ]
    }

    private static func quote(_ argument: String) -> String {
        "'" + argument.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
