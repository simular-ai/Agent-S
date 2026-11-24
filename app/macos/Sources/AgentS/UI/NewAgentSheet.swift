import SwiftUI
import AgentSCore

/// Agent configuration shared by first launch and the + button.
struct NewAgentSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    private let isFirstAgent: Bool
    @State private var name: String
    @State private var setupError: String?
    @State private var location: AgentLocation
    @State private var harness: Harness = .claudeCode
    @State private var provider: ModelProvider = .ollama
    @State private var selectedOllamaModel: String?
    @State private var openRouterModel = OpenRouter.suggested[0]
    @State private var openRouterZDR: Bool?
    @State private var saiMachine = ""
    @State private var saiModel = "auto"
    /// Sai first: it's all in one (agent, model and computer with one key); Simular Box is the
    /// configurable option.
    @State private var computer: CloudComputer = .sai
    @State private var boxName = ""

    init(initialLocation: AgentLocation = .linuxVM, isFirstAgent: Bool = false) {
        self.isFirstAgent = isFirstAgent
        let location: AgentLocation = isFirstAgent ? .linuxVM : initialLocation
        _location = State(initialValue: location)
        _harness = State(initialValue: location == .cloud ? .sai : .claudeCode)
        _name = State(initialValue: isFirstAgent ? "Agent" : "Sai")
    }

    var body: some View {
        VStack(spacing: 0) {
            if isFirstAgent {
                VStack(spacing: 8) {
                    SaiAvatar(size: 40)
                    Text("Set up your first agent")
                        .font(.title2.weight(.semibold))
                    Text("Choose how your agent runs in its own Linux sandbox.")
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 20)
                .padding(.bottom, 4)
            }
            Form {
                Section {
                    if isFirstAgent {
                        Label(location.displayName, systemImage: location.symbol)
                    } else {
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
                            Picker("Model", selection: Binding(
                                get: { ollamaModel }, set: { selectedOllamaModel = $0 }
                            )) {
                                ForEach(ollamaChoices, id: \.self) { name in
                                    Text(OllamaModelSetup.title(for: name)).tag(name)
                                }
                            }
                            OllamaModelSetup(name: ollamaModel)
                            if model.ollamaReady {
                                Button("Refresh model list") { Task { await model.refreshOllama() } }
                            }
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
                            Text("Sign in after creating the agent. \(notice)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                if !isFirstAgent {
                    Section {
                        Text("A new agent starts with its own chat and harness session.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            if let setupError {
                Text(setupError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding([.horizontal, .bottom], 12)
            }
            Divider()
            HStack {
                if isFirstAgent {
                    Text("Downloads the sandbox and its tools.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if !isFirstAgent {
                    Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                }
                Button(isFirstAgent ? "Set up agent" : "Create") { create() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(12)
        }
        .frame(width: 520)
        .frame(minHeight: isFirstAgent ? nil : 600, idealHeight: 600, maxHeight: isFirstAgent ? 620 : 600)
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
            if !isFirstAgent, selectedOllamaModel == nil {
                selectedOllamaModel = model.modelChoices.first ?? AppModel.defaultAgentModel
            }
        }
        .task(id: selectedProvider == .ollama) {
            if selectedProvider == .ollama { await model.refreshOllama() }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            if selectedProvider == .ollama, !model.ollamaReady {
                Task { await model.refreshOllama() }
            }
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

    private var localOllamaNames: [String] {
        model.ollamaModels.filter { !$0.usesRemoteInference && !Ollama.isCloudReference($0.name) }
            .map { $0.name.hasSuffix(":latest") ? String($0.name.dropLast(7)) : $0.name }
    }

    private var ollamaModel: String {
        selectedOllamaModel ?? (isFirstAgent ? localOllamaNames.first : model.modelChoices.first) ?? AppModel.defaultAgentModel
    }

    private var ollamaChoices: [String] {
        var choices = isFirstAgent ? localOllamaNames : model.modelChoices
        for name in [AppModel.defaultAgentModel, OllamaModelSetup.nemotron] where !choices.contains(name) {
            choices.append(name)
        }
        if !choices.contains(ollamaModel) { choices.append(ollamaModel) }
        return choices
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
        let config = AgentConfig(
            name: name.trimmingCharacters(in: .whitespaces), privacy: location.mode(for: selectedProvider), harness: harness,
            provider: selectedProvider,
            ollamaModel: ollamaModel, openRouterModel: openRouterModel,
            saiMachine: saiMachine, saiModel: saiModel,
            cloudComputer: cloudComputer,
            boxName: boxName.trimmingCharacters(in: .whitespaces).isEmpty ? nil : boxName.trimmingCharacters(in: .whitespaces))
        if isFirstAgent {
            do {
                try model.completeFirstAgentSetup(config)
            } catch {
                setupError = "Couldn’t save your agent. \(error.localizedDescription)"
                return
            }
        }
        if selectedProvider == .openRouter, harness != .sai, let openRouterZDR {
            model.openRouterRequiresZDR = openRouterZDR
        }
        if isFirstAgent {
            Task { await model.prepareAndStartSandbox() }
        } else {
            model.createAgent(config)
            dismiss()
        }
    }
}
