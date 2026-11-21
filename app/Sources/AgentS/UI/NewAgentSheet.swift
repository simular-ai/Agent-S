import SwiftUI
import AgentSCore

/// The + button: a new agent with its own context. Pick where it works (a private computer, this
/// Mac itself, or the cloud),
/// its harness, and its provider and model (or, for Sai, its computer and model).
struct NewAgentSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = "Sai"
    @State private var side: PrivacyMode
    @State private var harness: Harness = .claudeCode
    @State private var provider: ModelProvider = .ollama
    @State private var ollamaModel = AppModel.defaultAgentModel
    @State private var openRouterModel = OpenRouter.suggested[0]
    @State private var saiMachine = ""
    @State private var saiModel = "auto"
    /// Sai first: it's all in one (agent, model and computer with one key); Simular Box is the
    /// configurable option.
    @State private var computer: CloudComputer = .sai
    @State private var boxName = ""

    init(initialSide: PrivacyMode = .fullyPrivate) {
        _side = State(initialValue: initialSide)
        _harness = State(initialValue: initialSide == .remote ? .sai : .claudeCode)
    }

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    TextField("Name", text: $name)
                    Picker("Runs", selection: $side) {
                        ForEach(PrivacyMode.allCases) { mode in
                            Label(mode.displayName, systemImage: mode.symbol).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text(side.summary).font(.caption).foregroundStyle(.secondary)
                    if side == .thisMac {
                        Label("It acts on your real Mac. Read each step before you allow it.", systemImage: "exclamationmark.shield")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                if side == .remote {
                    Section("Computer") {
                        Picker("Computer", selection: $computer) {
                            ForEach(CloudComputer.allCases) { Text($0.displayName).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        Text(computer == .sai
                             ? "Connect your Sai account to use its agent, model, and computer."
                             : "Choose the harness and model provider. The harness runs in Agent S and uses only your Box’s tools.")
                            .font(.caption).foregroundStyle(.secondary)
                        if computer == .box {
                            TextField("Box", text: $boxName, prompt: Text("Default Simular Box"))
                            KeySetup.box(model)
                        }
                    }
                }
                if !(side == .remote && computer == .sai) {
                    Section("Harness") {
                        Picker("Harness", selection: $harness) {
                            ForEach(availableHarnesses) { Text($0.displayName).tag($0) }
                        }
                    }
                }
                if harness == .sai {
                    Section("Sai") {
                        KeySetup.sai(model)
                        Picker("Computer", selection: $saiMachine) {
                            ForEach(model.saiMachines) { Text($0.name + ($0.online ? "" : " (offline)")).tag($0.id) }
                            Text("This Mac's computer (coming)").tag(SaiAPI.localMachine)
                            if saiMachine.isEmpty { Text("None").tag("") }
                        }
                        Picker("Model", selection: $saiModel) {
                            ForEach(model.saiModels.filter(\.allowed)) { Text($0.id == "auto" ? "\($0.name) (recommended)" : $0.name).tag($0.id) }
                            if !model.saiModels.contains(where: { $0.id == saiModel }) { Text(saiModel).tag(saiModel) }
                        }
                    }
                } else {
                    Section("Model") {
                        if availableProviders.count > 1 {
                            Picker("Provider", selection: $provider) {
                                ForEach(availableProviders) { Text($0.displayName).tag($0) }
                            }
                        } else if let onlyProvider = availableProviders.first {
                            LabeledContent("Provider", value: onlyProvider.displayName)
                        }
                        if selectedProvider == .ollama {
                            Picker("Model", selection: $ollamaModel) {
                                ForEach(model.modelChoices, id: \.self) { Text($0).tag($0) }
                                if !model.modelChoices.contains(ollamaModel) { Text(ollamaModel).tag(ollamaModel) }
                            }
                            Button("Refresh model list") { Task { await model.refreshOllama() } }
                        } else if selectedProvider == .openRouter {
                            Picker("Model", selection: $openRouterModel) {
                                ForEach(OpenRouter.suggested, id: \.self) { Text($0).tag($0) }
                            }
                            TextField("Or any OpenRouter model id", text: $openRouterModel)
                            KeySetup.openRouter(model)
                        } else if let notice = selectedProvider.accountDataNotice {
                            LabeledContent("Model", value: "Automatic")
                            Text("Sign in after creating the agent. \(notice)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                Section {
                    Text("A new agent starts with its own chat and harness session.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Create") { create() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(12)
        }
        .frame(width: 480, height: 600)
        .onChange(of: computer) { _, _ in fitHarnessAndProvider() }
        .onChange(of: harness) { _, _ in fitProvider() }
        .onChange(of: model.saiMachines) { _, machines in
            if saiMachine.isEmpty { saiMachine = machines.first(where: \.online)?.id ?? machines.first?.id ?? "" }
        }
        .onChange(of: side, initial: true) { _, side in
            provider = side == .fullyPrivate ? .ollama : (model.hasOpenRouterKey ? .openRouter : .ollama)
            fitHarnessAndProvider()
        }
        .task {
            ollamaModel = model.modelChoices.first ?? AppModel.defaultAgentModel
        }
        // Sai's computers are looked up (which reads the Sai key) only once a Sai agent is chosen.
        .task(id: harness == .sai && model.hasSaiKey) {
            guard harness == .sai else { return }
            await model.refreshSai()
            if saiMachine.isEmpty {
                saiMachine = model.saiMachines.first(where: \.online)?.id ?? model.saiMachines.first?.id ?? ""
            }
        }
    }

    private var cloudComputer: CloudComputer? { side == .remote ? computer : nil }

    private var availableHarnesses: [Harness] { Harness.allowed(in: side, computer: cloudComputer) }

    private var availableProviders: [ModelProvider] { ModelProvider.allowed(in: side, harness: harness, computer: cloudComputer) }

    private var selectedProvider: ModelProvider {
        ModelProvider.fitting(provider, in: side, harness: harness, computer: cloudComputer) ?? provider
    }

    private func fitHarnessAndProvider() {
        let allowed = availableHarnesses
        if !allowed.contains(harness) { harness = allowed[0] }
        fitProvider()
    }

    private func fitProvider() {
        provider = selectedProvider
    }

    private func create() {
        model.createAgent(AgentConfig(
            name: name.trimmingCharacters(in: .whitespaces), privacy: side, harness: harness,
            provider: selectedProvider,
            ollamaModel: ollamaModel, openRouterModel: openRouterModel,
            saiMachine: saiMachine, saiModel: saiModel,
            cloudComputer: cloudComputer,
            boxName: boxName.trimmingCharacters(in: .whitespaces).isEmpty ? nil : boxName.trimmingCharacters(in: .whitespaces)))
        dismiss()
    }
}
