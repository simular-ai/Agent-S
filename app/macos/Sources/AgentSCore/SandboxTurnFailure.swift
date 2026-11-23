public enum SandboxTurnFailure {
    public static let codexWriterRecoveryMessage = "Codex still has this conversation open. Wait a moment and check what the previous turn completed before sending again. If it stays busy, turn the sandbox off and back on."

    /// Codex reports the same writer conflict with or without a thread-store prefix.
    public static func codexWriterMessage(for diagnostic: String) -> String? {
        diagnostic.lowercased().contains("already has an active writer") ? codexWriterRecoveryMessage : nil
    }
}
