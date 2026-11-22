import SwiftUI
import AgentSCore

/// Everything the agent is working with in this conversation, under the computer's screen: the
/// session, what the last turn cost, the instructions it was given, and the full transcript with
/// every tool's output (which the chat folds away).
struct ContextView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
    @State private var showInstructions = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Context").font(.headline)

                section("Session") {
                    row("Harness", model.harness.displayName)
                    row("Model", model.harness == .sai
                        ? "\(model.activeModel) · Sai API" : "\(model.activeModel) · \(model.provider.displayName)")
                    if model.harness == .sai { row("Computer", model.saiMachineName) }
                    if model.computer == .box { row("Computer", model.boxName.isEmpty ? "Simular Box (default)" : "Simular Box · \(model.boxName)") }
                    row("Session", agent.currentSessionID ?? "New (starts with the next message)", mono: true)
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

                section("Conversation · \(agent.items.count) items") {
                    if agent.items.isEmpty {
                        Text("Nothing yet.").font(.callout).foregroundStyle(.secondary)
                    }
                    ForEach(agent.items) { ContextItem(item: $0) }
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
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

/// One transcript entry in full: who said it, and for a tool step the command and its output.
private struct ContextItem: View {
    let item: ChatItem

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption.weight(.semibold)).foregroundStyle(color)
            Text(item.toolInput ?? item.text).font(.callout).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if case let .step(_, done, _, output) = item.kind {
                Text(done ? (output.isEmpty ? "(no output)" : output) : "running…")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Theme.bubble))
            }
        }
        .padding(.vertical, 4)
    }

    private var label: String {
        switch item.kind {
        case .user: "You"
        case .agent: "Sai"
        case let .step(name, _, failed, _):
            ToolActivity.title(tool: name, summary: item.text, detail: item.toolInput) + (failed ? " · failed" : "")
        case let .notice(isError): isError ? "Error" : "Note"
        case .approval: "Approval"
        }
    }

    private var color: Color {
        switch item.kind {
        case let .step(_, _, failed, _): failed ? .red : .secondary
        case let .notice(isError): isError ? .red : .secondary
        case .approval: .orange
        default: .primary
        }
    }
}
