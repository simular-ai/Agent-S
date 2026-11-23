import SwiftUI
import AgentSCore

/// The selected agent's action items, memory, and session details, below the computer screen.
struct ContextView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
    @State private var showActionItems = false
    @State private var showSessionDetails = false
    @State private var showInstructions = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                actionItems
                Divider()
                AgentMemoryView()
                Divider()
                DisclosureGroup("Session and instructions", isExpanded: $showSessionDetails) {
                    sessionDetails.padding(.top, 10)
                }
                .font(.callout)
                .foregroundStyle(.secondary)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var actionItems: some View {
        let requests = agent.pendingActionItems
        let work = model.openWork(for: agent)

        return DisclosureGroup(isExpanded: $showActionItems) {
            VStack(alignment: .leading, spacing: 8) {
                if requests.isEmpty, work.isEmpty {
                    Text("Nothing needs your attention.").font(.callout).foregroundStyle(.secondary)
                }
                ForEach(requests) { item in
                    HStack(spacing: 8) {
                        Label(item.text, systemImage: "hand.raised")
                            .font(.callout)
                            .lineLimit(2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button("Review") { model.revealChatItem(item.id, in: agent) }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                    }
                    .padding(10)
                    .background(Theme.card, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.hairline).allowsHitTesting(false))
                }
                if !work.isEmpty {
                    section("Open work") {
                        OpenWorkChecklist(notes: work)
                            .id("\(agent.id)/\(model.memoryRoute(for: agent).continuationKey)")
                    }
                }
            }
            .padding(.top, 8)
        } label: {
            Text("Action items (\(requests.count + work.count))")
                .font(.headline)
                .accessibilityAddTraits(.isHeader)
        }
    }

    private var sessionDetails: some View {
        VStack(alignment: .leading, spacing: 12) {
            section("Session") {
                row("Harness", model.harness.displayName)
                row("Model", model.harness == .sai
                    ? "\(model.activeModel) · Sai API" : "\(model.activeModel) · \(model.provider.displayName)")
                if model.harness == .sai { row("Computer", model.saiMachineName) }
                if model.computer == .box { row("Computer", model.boxName.isEmpty ? "Simular Box (default)" : "Simular Box · \(model.boxName)") }
                row("Session", agent.currentSessionID(for: model.memoryRoute(for: agent)) ?? "New (starts with the next message)", mono: true)
                if let usage = agent.lastUsage {
                    row("Last turn", "\(usage.input.formatted()) tokens in · \(usage.output.formatted()) out")
                }
            }

            section("Instructions") {
                DisclosureGroup(isExpanded: $showInstructions) {
                    Text(instructions)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 6)
                } label: {
                    Text(instructionsLabel)
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
        }
        .foregroundStyle(.primary)
    }

    private var instructions: String {
        if model.computer == .box {
            switch model.harness {
            case .claudeCode: return Box.instructions(boxName: model.boxName)
            case .deepSeek: return DeepSeekTurn.instructions(boxName: model.boxName)
            default: return Box.agentInstructions(boxName: model.boxName)
            }
        }
        if model.privacy == .thisMac, model.harness == .deepSeek { return DeepSeekTurn.macInstructions }
        if model.provider == .codexAccount {
            return CodexAccountTurn.instructions
        }
        if model.privacy == .thisMac, model.provider == .claudeAccount {
            return MacAgent.instructions(macToolPath: MacAccess.macTool, viaMCP: true)
        }
        return AgentInstructions.text(for: model.harness)
    }

    private var instructionsLabel: String {
        if model.computer == .box { return "Simular Box tools" }
        if model.provider == .codexAccount || model.privacy == .thisMac && model.harness == .deepSeek {
            return "Agent S tools and approvals"
        }
        if model.harness == .codex { return "AGENTS.md, written before each turn" }
        return "System prompt (replaces Claude Code's default)"
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6, content: content)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 12).fill(Theme.card))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.hairline))
        }
    }

    private func row(_ label: String, _ value: String, mono: Bool = false) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.callout).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
            Text(value)
                .font(mono ? .system(.callout, design: .monospaced) : .callout)
                .textSelection(.enabled)
                .lineLimit(2)
        }
    }
}

private struct OpenWorkChecklist: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
    @State private var selection: Set<UUID> = []
    @State private var pendingDeletion: Set<UUID> = []
    @State private var startingWork = false
    @State private var actionError: String?

    let notes: [AgentMemoryNote]

    private var checkedIDs: Set<UUID> { selection.intersection(notes.map(\.id)) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !checkedIDs.isEmpty {
                HStack(spacing: 8) {
                    Spacer()
                    Button("Delete", role: .destructive) { pendingDeletion = checkedIDs }
                        .buttonStyle(.bordered)
                        .disabled(startingWork)
                    Button("Run", action: run)
                        .buttonStyle(.borderedProminent).tint(Theme.accent)
                        .disabled(startingWork || !model.canRunOpenWork(for: agent))
                        .help(model.chatBlocker ?? "Run checked items in one chat turn")
                }
                .controlSize(.small)
                Divider()
            }
            ForEach(notes) { note in
                Toggle(isOn: Binding(
                    get: { selection.contains(note.id) },
                    set: { checked in
                        if checked { selection.insert(note.id) }
                        else { selection.remove(note.id) }
                    }
                )) {
                    Text(note.text).font(.callout).lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .toggleStyle(.checkbox)
                .help(note.text)
                .disabled(startingWork)
            }
        }
        .onChange(of: notes.map(\.id)) { _, ids in selection.formIntersection(ids) }
        .confirmationDialog("Delete checked work?", isPresented: Binding(
            get: { !pendingDeletion.isEmpty }, set: { if !$0 { pendingDeletion.removeAll() } }
        ), presenting: pendingDeletion) { ids in
            Button("Delete", role: .destructive) { delete(ids) }
        }
        .alert("Open work", isPresented: Binding(
            get: { actionError != nil }, set: { if !$0 { actionError = nil } }
        )) {
            Button("OK", role: .cancel) { actionError = nil }
        } message: {
            Text(actionError ?? "")
        }
    }

    private func run() {
        let ids = checkedIDs
        startingWork = true
        Task {
            defer { startingWork = false }
            if await model.runOpenWork(ids: ids, for: agent) {
                selection.subtract(ids)
            } else {
                actionError = model.chatBlocker ?? "Couldn’t run the checked items."
            }
        }
    }

    private func delete(_ ids: Set<UUID>) {
        guard model.selectedAgentID == agent.id else { return }
        let current = ids.intersection(model.openWork(for: agent).map(\.id))
        guard !current.isEmpty else { return }
        do {
            try agent.forgetMemoryNotes(ids: current)
            selection.subtract(current)
        } catch {
            actionError = error.localizedDescription
        }
    }
}
