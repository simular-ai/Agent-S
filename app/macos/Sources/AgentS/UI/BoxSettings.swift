import SwiftUI
import AgentSCore

/// The Simular Box connection and the Box this agent works on.
struct BoxSettings: View {
    @EnvironmentObject private var model: AppModel
    /// Keys are set in the app's Settings; an agent's settings only show one that is missing.
    var showKey = false

    var body: some View {
        Section("Simular Box") {
            if showKey || !model.hasBoxKey { KeySetup.box(model) }
            TextField("Box", text: $model.boxName, prompt: Text("Default Simular Box"))
            Text("The harness uses only your Box’s browser, shell, and files.")
                .font(.caption).foregroundStyle(.secondary)
            if model.harness == .deepSeek, DeepSeekBoxTurn.executable == nil {
                Text("Install DeepSeek Harness using Node.js 24.9 or later.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("npm install -g @deepseek-ai/dsh@\(DeepSeekCommand.version)")
                    .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
        }
    }
}
