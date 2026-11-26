import SwiftUI
import AgentSCore

/// Optional cloud setup while the first local agent is being prepared.
struct FirstAgentCloudView: View {
    enum Service: String, CaseIterable {
        case automatic
        case claudeCode
        case codex
        case openRouter
        case none

        var label: String {
            switch self {
            case .automatic: "Automatic"
            case .claudeCode: "Claude Code"
            case .codex: "Codex"
            case .openRouter: "OpenRouter"
            case .none: "None of these"
            }
        }
    }

    @EnvironmentObject private var model: AppModel
    @Binding var service: Service
    @State private var installedHarnesses: [Harness] = []
    let back: () -> Void
    let skip: () -> Void
    let configure: (NewAgentSheet.CloudSetup) -> Void

    var body: some View {
        VStack(spacing: 20) {
            VStack(spacing: 8) {
                Image(systemName: "cloud").font(.system(size: 30, weight: .medium))
                    .accessibilityHidden(true)
                Text("Set up a cloud agent")
                    .font(.title2.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                Text("Your local setup can continue while you connect a cloud computer.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            VStack(alignment: .leading, spacing: 6) {
                Picker("Which of these do you already use?", selection: $service) {
                    ForEach(Service.allCases, id: \.self) { choice in
                        Text(choice.label).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                if service == .automatic {
                    Text(detectionSummary).font(.caption).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .leading, spacing: 8) {
                Label(recommendsBox ? "Simular Box" : "Sai", systemImage: recommendsBox ? "desktopcomputer" : "sparkles")
                    .font(.headline)
                Text(recommendsBox
                     ? "A cloud computer built for agents and optimized for computer use. Multiple agents can share the same Box."
                     : "Sai includes the agent, model, and cloud computer. Connect your Sai account with one key.")
                    .font(.callout).foregroundStyle(.secondary)
                Button(recommendsBox ? "Set up Simular Box" : "Set up Sai") {
                    configure(recommendsBox ? boxSetup : .sai)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))

            Button(recommendsBox ? "Set up Sai instead" : "Set up Simular Box instead") {
                configure(recommendsBox ? .sai : boxSetup)
            }

            HStack(spacing: 12) {
                Button("Back", action: back).keyboardShortcut(.cancelAction)
                Button("Skip for now", action: skip)
            }
        }
        .onAppear(perform: detectHarnesses)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            detectHarnesses()
        }
    }

    private var detectedService: Service {
        if model.hasOpenRouterKey { return .openRouter }
        if installedHarnesses.contains(.claudeCode) { return .claudeCode }
        if installedHarnesses.contains(.codex) { return .codex }
        return .none
    }

    private var activeService: Service { service == .automatic ? detectedService : service }
    private var recommendsBox: Bool { activeService != .none }

    private var detectionSummary: String {
        switch detectedService {
        case .automatic, .none: "No supported setup detected on this Mac. Choose a service if you use it elsewhere."
        case .openRouter: "An OpenRouter key is saved in Agent S."
        case .claudeCode, .codex: "\(detectedService.label) is installed on this Mac."
        }
    }

    private var boxSetup: NewAgentSheet.CloudSetup {
        switch activeService {
        case .claudeCode: .box(harness: .claudeCode, provider: nil)
        case .codex: .box(harness: .codex, provider: nil)
        case .openRouter: .box(harness: installedHarnesses.first ?? .claudeCode, provider: .openRouter)
        case .automatic, .none: .box(harness: .claudeCode, provider: nil)
        }
    }

    private func detectHarnesses() {
        installedHarnesses = [
            BoxAccount.claude == nil ? nil : .claudeCode,
            CodexAccount.executable == nil ? nil : .codex,
            !DeepSeekTurn.isAvailable ? nil : .deepSeek,
        ].compactMap { $0 }
    }
}
