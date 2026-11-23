import SwiftUI
import AgentSCore

/// The + button: a new agent with its own context, computer, harness, and model.
struct NewAgentSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = "Sai"
    @State private var location: AgentLocation
    @State private var harness: Harness = .claudeCode
    @State private var provider: ModelProvider = .ollama
    @State private var ollamaModel = AppModel.defaultAgentModel
    @State private var openRouterModel = OpenRouter.suggested[0]
    @State private var openRouterZDR: Bool?
    @State private var saiMachine = ""
    @State private var saiModel = "auto"
    /// Sai first: it's all in one (agent, model and computer with one key); Simular Box is the
    /// configurable option.
    @State private var computer: CloudComputer = .sai
    @State private var boxName = ""

    init(initialLocation: AgentLocation = .linuxVM) {
        _location = State(initialValue: initialLocation)
        _harness = State(initialValue: initialLocation == .cloud ? .sai : .claudeCode)
    }

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    TextField("Name", text: $name)
                    Picker("Runs", selection: $location) {
                        ForEach(AgentLocation.allCases) { location in
                            Label(location.displayName, systemImage: location.symbol).tag(location)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text(location.summary).font(.caption).foregroundStyle(.secondary)
                    if location == .thisMac {
                        Label("Review each action before approving it.", systemImage: "exclamationmark.shield")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                if location == .cloud {
                    Section("Computer") {
                        Picker("Computer", selection: $computer) {
                            ForEach(CloudComputer.allCases) { Text($0.displayName).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        Text(computer == .sai
                             ? "Connect your Sai account to use its agent, model, and computer."
                             : "Choose the harness and model provider. The harness runs on your Mac and uses only your Box’s tools.")
                            .font(.caption).foregroundStyle(.secondary)
                        if computer == .box {
                            TextField("Box", text: $boxName, prompt: Text("Default Simular Box"))
                            KeySetup.box(model)
                        }
                    }
                }
                if cloudComputer != .sai {
                    Section("Harness") {
                        Picker("Harness", selection: $harness) {
                            ForEach(availableHarnesses) { Text($0.displayName).tag($0) }
                        }
                        if location != .linuxVM, harness == .deepSeek, DeepSeekTurn.executable == nil {
                            DeepSeekInstallHelp()
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
                            Text("Choose a local Ollama model for private inference on this Mac.")
                                .font(.caption).foregroundStyle(.secondary)
                            Picker("Model", selection: $ollamaModel) {
                                ForEach(model.modelChoices, id: \.self) { Text($0).tag($0) }
                                if !model.modelChoices.contains(ollamaModel) { Text(ollamaModel).tag(ollamaModel) }
                            }
                            Button("Refresh model list") { Task { await model.refreshOllama() } }
                        } else if selectedProvider == .openRouter {
                            Text("OpenRouter and the selected model provider receive your messages and content the agent reads.")
                                .font(.caption).foregroundStyle(.secondary)
                            OpenRouterZDRSetting(isOn: Binding(
                                get: { openRouterZDR ?? model.openRouterRequiresZDR }, set: { openRouterZDR = $0 }
                            ), appliesOnCreate: true)
                            OpenRouterModelSearch(selection: $openRouterModel)
                            Picker("Suggested models", selection: $openRouterModel) {
                                ForEach(OpenRouter.suggested, id: \.self) { Text($0).tag($0) }
                                if !OpenRouter.suggested.contains(openRouterModel) { Text(openRouterModel).tag(openRouterModel) }
                            }
                            KeySetup.openRouter(model)
                        } else if let notice = selectedProvider.accountDataNotice {
                            LabeledContent("Model", value: "Automatic")
                            Text("Sign in \(location == .linuxVM ? "inside the sandbox " : "")after creating the agent. \(notice)")
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
        .frame(width: 520, height: 600)
        .onChange(of: computer) { _, _ in fitHarnessAndProvider() }
        .onChange(of: harness) { _, _ in fitProvider() }
        .onChange(of: model.saiMachines) { _, machines in
            if saiMachine.isEmpty { saiMachine = machines.first(where: \.online)?.id ?? machines.first?.id ?? "" }
        }
        .onChange(of: location, initial: true) { previous, location in
            if location == .cloud {
                provider = model.hasOpenRouterKey ? .openRouter : .ollama
            } else if previous == .cloud {
                provider = .ollama
            }
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

    private var cloudComputer: CloudComputer? { location == .cloud ? computer : nil }

    private var availableHarnesses: [Harness] { Harness.allowed(in: location.mode(for: .ollama), computer: cloudComputer) }

    private var availableProviders: [ModelProvider] { ModelProvider.allowed(at: location, harness: harness, computer: cloudComputer) }

    private var selectedProvider: ModelProvider {
        ModelProvider.fitting(provider, at: location, harness: harness, computer: cloudComputer) ?? provider
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
        if selectedProvider == .openRouter, harness != .sai, let openRouterZDR {
            model.openRouterRequiresZDR = openRouterZDR
        }
        model.createAgent(AgentConfig(
            name: name.trimmingCharacters(in: .whitespaces), privacy: location.mode(for: selectedProvider), harness: harness,
            provider: selectedProvider,
            ollamaModel: ollamaModel, openRouterModel: openRouterModel,
            saiMachine: saiMachine, saiModel: saiModel,
            cloudComputer: cloudComputer,
            boxName: boxName.trimmingCharacters(in: .whitespaces).isEmpty ? nil : boxName.trimmingCharacters(in: .whitespaces)))
        dismiss()
    }
}
