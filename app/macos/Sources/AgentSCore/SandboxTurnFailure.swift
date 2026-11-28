public enum SandboxTurnFailure {
    public static let codexWriterRecoveryMessage = "Codex still has this conversation open. Wait a moment and check what the previous turn completed before sending again. If it stays busy, turn the sandbox off and back on."
    static let outOfMemoryMarker = "AGENT_S_SANDBOX_TURN_OOM"

    public static func outOfMemoryMessage(for diagnostic: String) -> String? {
        guard diagnostic.split(whereSeparator: \.isNewline).contains(Substring(outOfMemoryMarker)) else { return nil }
        return "The sandbox ran out of memory. Increase Memory in Settings → Computer, then turn the sandbox off and back on. Check what the previous turn completed before trying again."
    }

    /// Codex reports the same writer conflict with or without a thread-store prefix.
    public static func codexWriterMessage(for diagnostic: String) -> String? {
        diagnostic.lowercased().contains("already has an active writer") ? codexWriterRecoveryMessage : nil
    }
}
