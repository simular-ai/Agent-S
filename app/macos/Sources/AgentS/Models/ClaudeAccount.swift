import AppKit
import AgentSCore

/// The installed CLI owns authentication and stores its sign-in in Agent S's Claude Code profile.
enum ClaudeAccount {
    private static let path = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

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
    static func signInInSandbox(shell: GuestShell, ip: String, dataFolder: URL) async throws {
        // Terminal carries the vendor's sign-in directly over SSH. Claude Code owns the guest profile.
        let arguments = ["-tt"] + shell.sshArguments(ip: ip, command: SandboxAccount.claudeTerminalSignInCommand)
        let command = "exec /usr/bin/ssh " + arguments.map(quote).joined(separator: " ")
        let directory = dataFolder.appendingPathComponent("claude-sandbox-sign-in", isDirectory: true)
        try await openTerminal(command: command, in: directory)
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
