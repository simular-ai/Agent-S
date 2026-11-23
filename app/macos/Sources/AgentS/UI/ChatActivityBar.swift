import SwiftUI
import AgentSCore

struct ChatTranscriptRow: View {
    let row: ChatTranscript.Row
    let isRunning: Bool
    var showsWorking = false

    var body: some View {
        switch row.content {
        case let .message(item):
            ChatRow(item: item)
        case let .activity(steps):
            ChatActivityBar(steps: steps, isCurrentTurnRunning: isRunning && row.isCurrentTurn,
                            showsWorking: showsWorking)
        }
    }
}

/// One disclosure for an uninterrupted stretch of tool calls.
struct ChatActivityBar: View {
    let steps: ArraySlice<ChatItem>
    let isCurrentTurnRunning: Bool
    let showsWorking: Bool
    @State private var expanded = false

    var body: some View {
        let status = summary
        return VStack(alignment: .leading, spacing: 8) {
            Button { expanded.toggle() } label: {
                HStack(spacing: 8) {
                    statusIcon(status)
                    Text(status.title).font(.callout).lineLimit(1)
                    Spacer(minLength: 0)
                    if let count = status.count {
                        Text(count).font(.caption).foregroundStyle(.secondary).fixedSize()
                    }
                    if let issue = status.issue {
                        Text(issue).font(.caption).foregroundStyle(status.hasFailure ? Color.red : .secondary).fixedSize()
                    }
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.caption2.weight(.semibold)).foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 14).padding(.vertical, 9)
                .background(RoundedRectangle(cornerRadius: 14).stroke(Theme.hairline))
                .contentShape(RoundedRectangle(cornerRadius: 14))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(status.title)
            .accessibilityValue(status.accessibilityValue)
            .accessibilityHint(expanded ? "Hide tool steps" : "Show tool steps")

            if expanded {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(steps) { step in
                        ChatRow(item: step, isToolActive: isCurrentTurnRunning)
                    }
                }
                .padding(.leading, 10)
            }
        }
    }

    private struct Summary {
        let title: String
        let count: String?
        let issue: String?
        let isWorking: Bool
        let hasFailure: Bool

        var accessibilityValue: String {
            [count, issue, isWorking ? "Running" : (issue == nil ? "Completed" : nil)]
                .compactMap { $0 }.joined(separator: ", ")
        }
    }

    private var summary: Summary {
        var active: ChatItem?
        var failures = 0
        var unfinished = 0
        for step in steps {
            if case let .step(_, done, failed, _) = step.kind {
                if failed { failures += 1 }
                else if !done {
                    if isCurrentTurnRunning { active = step }
                    else { unfinished += 1 }
                }
            }
        }

        let count = steps.count == 1 ? "1 step" : "\(steps.count) steps"
        let titles = [failures > 0 ? "\(failures) failed" : nil, unfinished > 0 ? "\(unfinished) unfinished" : nil]
            .compactMap { $0 }
        let action = active.flatMap(stepTitle) ?? (showsWorking ? "Working…" : nil)
        let single = steps.count == 1 ? steps.first.flatMap(stepTitle) : nil
        return Summary(title: action ?? single ?? count, count: action == nil ? nil : count,
                       issue: titles.isEmpty ? nil : titles.joined(separator: ", "),
                       isWorking: active != nil || showsWorking, hasFailure: failures > 0)
    }

    @ViewBuilder private func statusIcon(_ status: Summary) -> some View {
        if status.isWorking {
            ProgressView().controlSize(.small)
        } else {
            let symbol = status.hasFailure ? "exclamationmark.circle" : status.issue == nil ? "checkmark.circle" : "clock"
            let color: Color = status.hasFailure ? .red : status.issue == nil ? .green : .secondary
            Image(systemName: symbol).foregroundStyle(color)
        }
    }

    private func stepTitle(_ step: ChatItem) -> String? {
        guard case let .step(name, _, _, _) = step.kind else { return nil }
        return ToolActivity.title(tool: name, summary: step.text, detail: step.toolInput)
    }
}
