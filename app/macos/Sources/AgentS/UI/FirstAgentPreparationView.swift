import SwiftUI
import AgentSCore

/// The setup steps for the first sandbox and its model, before opening the chat.
struct FirstAgentPreparationView: View {
    private enum Page {
        case local
        case cloud
        case configureCloud(NewAgentSheet.CloudSetup)
    }

    @EnvironmentObject private var model: AppModel
    @State private var page: Page = .local
    @State private var service: FirstAgentCloudView.Service = .automatic
    @State private var finishedCloudStep = false
    @State private var finishError: String?
    let openComputer: () -> Void

    var body: some View {
        ZStack {
            switch page {
            case .local:
                centeredPage { localSetup }
            case .cloud:
                centeredPage {
                    FirstAgentCloudView(service: $service, back: { page = .local }, skip: finishCloudStep) {
                        page = .configureCloud($0)
                    }
                }
            case let .configureCloud(choice):
                NewAgentSheet(.cloudOnboarding(choice, back: { page = .cloud }, finished: finishCloudStep))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .tint(Theme.accent)
        .task {
            if model.usesOllama { await model.refreshOllama() }
        }
        .task(id: model.provider.usesAccount && model.accountComputerReady) {
            if model.provider.usesAccount, model.accountComputerReady {
                await model.refreshAccountStatus()
            }
        }
    }

    private func centeredPage<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        let pageContent = content()
        return GeometryReader { geometry in
            ScrollView {
                pageContent
                    .frame(maxWidth: 500)
                    .padding(28)
                    .frame(maxWidth: .infinity, minHeight: geometry.size.height)
            }
        }
    }

    private var localSetup: some View {
        VStack(spacing: 20) {
            VStack(spacing: 8) {
                SaiAvatar(size: 44)
                Text("Get your agent ready")
                    .font(.title2.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                if model.usesOllama, !modelIsReady {
                    Text("You can download the model while the sandbox sets up.")
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            VStack(spacing: 12) {
                setupCard("Sandbox on This Mac", symbol: "desktopcomputer") {
                    Text("A separate Linux desktop for your agent. The first setup downloads Debian and installs the tools your agent uses.")
                        .font(.callout).foregroundStyle(.secondary)
                    sandboxState
                }
                modelCard
            }
            if let finishError {
                Text(finishError).font(.callout).foregroundStyle(.red)
            }
            HStack(spacing: 12) {
                if offersCloudStep {
                    Button("Set up a cloud agent…") { page = .cloud }
                }
                if readyToChat {
                    Button("Open chat") { finish() }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button("Finish later") { finish() }
                }
            }
        }
    }

    private var offersCloudStep: Bool {
        !finishedCloudStep && !model.agents.contains { $0.config.privacy.location == .cloud }
    }

    private var modelIsReady: Bool {
        model.ollamaReady(for: model.agentModel) && model.hasAgentModel
    }

    private var readyToChat: Bool {
        model.canChat && (!model.usesOllama || modelIsReady)
    }

    private func finishCloudStep() {
        finishedCloudStep = true
        page = .local
        if readyToChat { finish() }
    }

    @ViewBuilder private var sandboxState: some View {
        switch model.phase {
        case let .preparing(step):
            if case let .downloadingDebian(progress) = step {
                DownloadProgressView(label: step.label, progress: progress)
            } else {
                ProgressView(step.label).controlSize(.small)
            }
        case let .booting(step):
            ProgressView(step).controlSize(.small)
        case .stopping:
            ProgressView(model.status.label).controlSize(.small)
        case .ready:
            Label("Sandbox ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case let .failed(message):
            Text(message).font(.callout).foregroundStyle(.red)
            if model.vm.machine != nil, model.vm.state != .running {
                Button("Turn off sandbox") { Task { await model.stop() } }
            } else {
                startButton("Try again")
            }
        case .stopped:
            startButton(model.hasSavedSandbox ? "Resume sandbox" : "Start sandbox")
        case .needsSetup:
            startButton("Set up sandbox")
        }
    }

    @ViewBuilder private var modelCard: some View {
        if model.usesOllama {
            let name = Ollama.displayName(for: model.agentModel)
            setupCard(name, symbol: "laptopcomputer") {
                Text("Ollama downloads and runs this model on your Mac.")
                    .font(.callout).foregroundStyle(.secondary)
                if modelIsReady {
                    Label("Model ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    OllamaModelDetails(name: model.agentModel, sandboxMemory: model.computerMemory.bytes)
                } else {
                    OllamaModelSetup(name: model.agentModel)
                }
            }
        } else if model.provider.usesAccount {
            setupCard(model.harness.displayName, symbol: "person.crop.circle") {
                HarnessAccountSetup {
                    finish(openSandbox: true)
                }
            }
        } else {
            setupCard(model.activeModel, symbol: "cloud") {
                Text("OpenRouter runs this model online.")
                    .font(.callout).foregroundStyle(.secondary)
                if model.hasOpenRouterKey {
                    Label("OpenRouter connected", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                } else {
                    KeySetup.openRouter(model)
                }
            }
        }
    }

    private func setupCard<Content: View>(_ title: String, symbol: String,
                                           @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            Label(title, systemImage: symbol).font(.headline)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
    }

    private func startButton(_ title: String) -> some View {
        Button(title) { Task { await model.prepareAndStartSandbox() } }
    }

    private func finish(openSandbox: Bool = false) {
        do {
            try model.finishFirstAgentPreparation()
            if openSandbox { openComputer() }
        } catch {
            finishError = "Couldn’t save your setup progress. Try opening chat again."
        }
    }
}
