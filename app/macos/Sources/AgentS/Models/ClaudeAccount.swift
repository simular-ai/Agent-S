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
        try Task.checkCancellation()
        let fileManager = FileManager.default
        let privateDirectory: [FileAttributeKey: Any] = [.posixPermissions: 0o700]
        try fileManager.createDirectory(at: profile, withIntermediateDirectories: true, attributes: privateDirectory)
        try fileManager.setAttributes(privateDirectory, ofItemAtPath: profile.path)

        // Terminal supports the CLI's browser flow and its fallback prompt for a code.
        let launcher = profile.appendingPathComponent("Sign in to Claude Code.command")
        let variables = environment(profile: profile).sorted { $0.key < $1.key }
            .map { quote("\($0.key)=\($0.value)") }.joined(separator: " ")
        let script = """
        #!/bin/sh
        cd \(quote(profile.path)) || exit 1
        exec /usr/bin/env -i \(variables) TERM="${TERM:-xterm-256color}" \(quote(claude.path)) auth login

        """
        try Data(script.utf8).write(to: launcher, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: launcher.path)

        let workspace = NSWorkspace.shared
        guard let terminal = workspace.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else {
            throw CocoaError(.fileNoSuchFile)
        }
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
