import SwiftUI
import AgentSCore

/// Agent configuration for first launch, Box onboarding, and the + button.
struct NewAgentSheet: View {
    enum Mode: Equatable {
        case firstAgent
        case boxOnboarding
        case newAgent(AgentLocation)
    }

    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    private let mode: Mode
    private var isFirstAgent: Bool { mode == .firstAgent }
    private var isBoxOnboarding: Bool { mode == .boxOnboarding }
    @State private var name: String
    @State private var setupError: String?
    @State private var location: AgentLocation
    @State private var harness: Harness
    @State private var providerOverride: ModelProvider?
    @State private var selectedOllamaModel: String?
    @State private var openRouterModel = OpenRouter.suggested[0]
    @State private var openRouterZDR: Bool?
    @State private var saiMachine = ""
    @State private var saiModel = "auto"
    @State private var computer: CloudComputer
    @State private var boxName = ""

    init(_ mode: Mode) {
        self.mode = mode
        switch mode {
        case .firstAgent:
            _location = State(initialValue: .linuxVM)
            _computer = State(initialValue: .sai)
            _harness = State(initialValue: .deepSeek)
            _name = State(initialValue: "Agent")
        case .boxOnboarding:
            _location = State(initialValue: .cloud)
            _computer = State(initialValue: .box)
            _harness = State(initialValue: .claudeCode)
            _name = State(initialValue: "Box Agent")
        case let .newAgent(location):
            _location = State(initialValue: location)
            _computer = State(initialValue: .sai)
            _harness = State(initialValue: location == .cloud ? .sai : .claudeCode)
            _name = State(initialValue: "Sai")
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            if isFirstAgent || isBoxOnboarding {
                VStack(spacing: 8) {
                    SaiAvatar(size: 40)
                    Text(isFirstAgent ? "Set up your first agent" : "Set up Simular Box")
                        .font(.title2.weight(.semibold))
                        .accessibilityAddTraits(.isHeader)
                    Text(isFirstAgent
                         ? "The open-source agent that uses apps and websites on its own private desktop."
                         : "Your local sandbox keeps setting up while you add a Box agent.")
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 4)
            }
            Form {
                Section {
                    if isFirstAgent {
                        Label(location.displayName, systemImage: location.symbol)
                        Text("Your agent gets a separate desktop that runs on this Mac.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        TextField("Name", text: $name)
                        if !isBoxOnboarding {
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
                }
                if location == .cloud {
                    Section("Computer") {
                        if isBoxOnboarding {
                            Link("Open Box dashboard", destination: Box.portal)
                            Text("Sign in to choose a Box. Leave the name blank to use your default Box.")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Picker("Computer", selection: $computer) {
                                ForEach(CloudComputer.allCases) { Text($0.displayName).tag($0) }
                            }
                            .pickerStyle(.segmented)
                            Text(computer == .sai
                                 ? "Connect your Sai account to use its agent, model, and computer."
                                 : "Choose the harness and model provider. The harness runs on your Mac and uses only your Box’s tools.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if computer == .box {
                            TextField("Box name", text: $boxName, prompt: Text("Default Simular Box"))
                            if isBoxOnboarding, model.hasBoxKey {
                                Label("Box key saved", systemImage: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            } else {
                                KeySetup.box(model)
                            }
                        }
                    }
                }
                if cloudComputer != .sai {
                    Section("Harness") {
                        Picker("Harness", selection: Binding(get: { harness }, set: selectHarness)) {
                            ForEach(availableHarnesses) { Text($0.displayName).tag($0) }
                        }
                        if isBoxOnboarding {
                            Text("The harness runs on this Mac and controls your Box remotely.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if location != .linuxVM, harness == .deepSeek, DeepSeekTurn.executable == nil {
                            DeepSeekInstallHelp()
                        } else if isBoxOnboarding, harness == .claudeCode, BoxAccount.claude == nil {
                            Link("Install Claude Code on this Mac", destination: URL(string: "https://code.claude.com/docs/en/setup")!)
                        } else if isBoxOnboarding, harness == .codex, CodexAccount.executable == nil {
                            Link("Install Codex on this Mac", destination: URL(string: "https://learn.chatgpt.com/docs/cli")!)
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
                            Picker("Provider", selection: Binding(
                                get: { selectedProvider }, set: { providerOverride = $0 }
                            )) {
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
                            if !isFirstAgent {
                                OllamaModelSetup(name: ollamaModel)
                                if model.ollamaReady {
                                    Button("Refresh model list") { Task { await model.refreshOllama() } }
                                }
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
                            if !isFirstAgent { KeySetup.openRouter(model) }
                        } else if let notice = selectedProvider.accountDataNotice {
                            LabeledContent("Model", value: "Automatic")
                            Text("Sign in after creating the agent. \(notice)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                if !isFirstAgent, !isBoxOnboarding {
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
                    Text("Next, set up the sandbox and connect your model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if !isFirstAgent {
                    Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                }
                Button(createTitle) { create() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || (isBoxOnboarding && !model.hasBoxKey))
            }
            .padding(12)
        }
        .frame(width: 520)
        .frame(minHeight: isFirstAgent ? nil : 600, idealHeight: 600, maxHeight: isFirstAgent ? 620 : 600)
        .onChange(of: computer) { _, _ in fitHarness() }
        .onChange(of: location) { _, _ in fitHarness() }
        .onChange(of: model.saiMachines) { _, machines in
            if saiMachine.isEmpty { saiMachine = machines.first(where: \.online)?.id ?? machines.first?.id ?? "" }
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

    private var createTitle: String {
        switch mode {
        case .firstAgent: "Set up agent"
        case .boxOnboarding: "Add Box agent"
        case .newAgent: "Create"
        }
    }

    private var localOllamaNames: [String] {
        model.ollamaModels.filter { !$0.usesRemoteInference && !Ollama.isCloudReference($0.name) }
            .map { $0.name.hasSuffix(":latest") ? String($0.name.dropLast(7)) : $0.name }
    }

    private var ollamaModel: String {
        selectedOllamaModel ?? (isFirstAgent ? OllamaModelSetup.nemotron : model.modelChoices.first) ?? AppModel.defaultAgentModel
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
        let preferred: ModelProvider = switch harness {
        case .claudeCode: .claudeAccount
        case .codex: .codexAccount
        case .deepSeek, .sai: .ollama
        }
        return ModelProvider.fitting(providerOverride ?? preferred, at: location, harness: harness,
                                     computer: cloudComputer) ?? preferred
    }

    private func fitHarness() {
        let choices = availableHarnesses
        if !choices.contains(harness), let first = choices.first { selectHarness(first) }
    }

    private func selectHarness(_ choice: Harness) {
        guard choice != harness else { return }
        harness = choice
        providerOverride = nil
    }

    private func create() {
        let config = AgentConfig(
            name: name.trimmingCharacters(in: .whitespaces), privacy: location.mode(for: selectedProvider), harness: harness,
            provider: selectedProvider,
            ollamaModel: ollamaModel, openRouterModel: openRouterModel,
            saiMachine: saiMachine, saiModel: saiModel,
            cloudComputer: cloudComputer,
            boxName: boxName.trimmingCharacters(in: .whitespaces).isEmpty ? nil : boxName.trimmingCharacters(in: .whitespaces))
        do {
            if isFirstAgent {
                try model.completeFirstAgentSetup(config)
            } else {
                try model.createAgent(config, selectAfterCreation: !isBoxOnboarding)
            }
        } catch {
            setupError = "Couldn’t save your agent. \(error.localizedDescription)"
            return
        }
        if selectedProvider == .openRouter, harness != .sai, let openRouterZDR {
            model.openRouterRequiresZDR = openRouterZDR
        }
        if isFirstAgent {
            Task { await model.prepareAndStartSandbox() }
        } else {
            dismiss()
        }
    }
}
