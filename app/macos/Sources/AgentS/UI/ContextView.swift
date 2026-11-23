import SwiftUI
import AgentSCore

/// The selected agent's action items, memory, and session details, below the computer screen.
struct ContextView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
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
        let work = agent.memory.notes(for: model.memoryRoute(for: agent))
            .filter { $0.kind == .activeWork && $0.status == .open }

        return VStack(alignment: .leading, spacing: 8) {
            Text("Action items").font(.headline).accessibilityAddTraits(.isHeader)
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
                    ForEach(work) { note in
                        Text(note.text)
                            .font(.callout)
                            .lineLimit(3)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .help(note.text)
                    }
                }
            }
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
            case .deepSeek: return DeepSeekBoxTurn.instructions(boxName: model.boxName)
            default: return Box.agentInstructions(boxName: model.boxName)
            }
        }
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
        if model.provider == .codexAccount { return "Agent S tools and approvals" }
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
