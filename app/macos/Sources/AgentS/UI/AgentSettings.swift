import SwiftUI
import AgentSCore

/// One agent's settings: its name, where it runs, its harness and model, and which computer it
/// uses. Opened by right-clicking the agent in the sidebar; it edits the selected agent.
struct AgentSettings: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                AgentAvatar(config: model.agent.config, size: 40)
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.agent.config.name).font(.headline)
                    Text(model.privacy.displayName).font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(16)
            Divider()
            Form {
                Section("Agent") {
                    TextField("Name", text: $name)
                        .onSubmit(saveName)
                    LabeledContent("Works on", value: model.agent.config.usesLocalComputer
                                   ? "A Linux VM on this Mac"
                                   : model.privacy == .thisMac ? "This Mac itself (asks before each step)"
                                   : "The cloud: \(model.computer?.displayName ?? "a cloud computer")")
                }

                Section("Model") {
                    Picker("Harness", selection: $model.harness) {
                        ForEach(Harness.allowed(in: model.privacy, computer: model.computer)) { Text($0.displayName).tag($0) }
                    }
                    if model.harness != .sai, model.availableProviders.count > 1 {
                        Picker("Provider", selection: $model.provider) {
                            ForEach(model.availableProviders) { Text($0.displayName).tag($0) }
                        }
                    } else if model.harness != .sai {
                        LabeledContent("Provider", value: model.provider.displayName)
                    }
                    if model.usesOllama {
                        Picker("Model", selection: $model.agentModel) {
                            ForEach(model.modelChoices, id: \.self) { Text($0).tag($0) }
                        }
                        if !model.hasAgentModel {
                            if let status = model.pullStatus {
                                Text(status).font(.caption)
                            } else {
                                Button("Download \(model.agentModel)") { Task { await model.pullAgentModel() } }
                                    .disabled(!model.ollamaReady)
                            }
                        }
                    }
                    if model.harness != .sai, model.provider.usesAccount {
                        LabeledContent("Model", value: "Automatic")
                        HarnessAccountSetup()
                    }
                }

                if model.computer == .box { BoxSettings() }
                if model.harness == .sai {
                    SaiSettings()
                } else if model.provider == .openRouter {
                    OpenRouterSettings()
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack {
                Button("New conversation") { model.agent.reset() }
                Spacer()
                Button("Done") { saveName(); dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(12)
        }
        .frame(width: 520, height: 560)
        .onAppear { name = model.agent.config.name }
        .task(id: model.usesOllama ? model.selectedAgentID : nil) {
            if model.usesOllama { await model.refreshOllama() }
        }
    }

    private func saveName() {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, trimmed != model.agent.config.name { model.rename(model.agent, to: trimmed) }
    }
}
