import SwiftUI
import AgentSCore

struct ChatTranscriptRow: View {
    let row: ChatTranscript.Row
    let isRunning: Bool
    var showsWorking = false
    var onHandoffDecision: ((UUID, Bool) -> Void)?

    var body: some View {
        switch row.content {
        case let .message(item):
            ChatRow(item: item, onHandoffDecision: onHandoffDecision)
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
    @State private var expandedStepID: String?

    var body: some View {
        let status = summary
        return VStack(alignment: .leading, spacing: 0) {
            Button { expanded.toggle() } label: {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        statusIcon(status).frame(width: 14)
                        Text(status.title).font(.callout.weight(.medium)).lineLimit(1)
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
                    if !expanded, let preview = status.preview {
                        Text(preview).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            .padding(.leading, 22)
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(status.title)
            .accessibilityValue(status.accessibilityValue)
            .accessibilityHint(expanded ? "Hide activity steps" : "Show activity steps")

            if expanded {
                Divider().padding(.horizontal, 12)
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(steps) { step in stepRow(step) }
                }
                .padding(.horizontal, 8).padding(.vertical, 4)
            }
        }
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(Theme.hairline).allowsHitTesting(false) }
    }

    private struct Summary {
        let title: String
        let count: String?
        let issue: String?
        let preview: String?
        let isWorking: Bool
        let hasFailure: Bool

        var accessibilityValue: String {
            [count, issue, isWorking ? "Running" : (issue == nil ? "Completed" : nil), preview]
                .compactMap { $0 }.joined(separator: ", ")
        }
    }

    private var summary: Summary {
        var active: ChatItem?
        var failureCount = 0
        var lastFailure: ChatItem?
        var unfinished = 0
        for step in steps {
            if case let .step(_, done, failed, _) = step.kind {
                if failed {
                    failureCount += 1
                    lastFailure = step
                } else if !done {
                    if isCurrentTurnRunning { active = step }
                    else { unfinished += 1 }
                }
            }
        }

        let count = steps.count == 1 ? "1 step" : "\(steps.count) steps"
        let titles = [failureCount > 0 ? "\(failureCount) failed" : nil, unfinished > 0 ? "\(unfinished) unfinished" : nil]
            .compactMap { $0 }
        let action = active.flatMap(stepTitle)
        let isWorking = active != nil || showsWorking
        let single = steps.count == 1 ? steps.first.flatMap(stepTitle) : nil
        let preview: String? = if !isWorking, single == nil, let title = lastFailure.flatMap(stepTitle) {
            "Last failed step · \(title)"
        } else if showsWorking, action == nil, let title = steps.last.flatMap(stepTitle) {
            "Last step · \(title)"
        } else { nil }
        return Summary(title: action ?? (isWorking ? "Working…" : single ?? "Activity"),
                       count: single != nil && !isWorking ? nil : count,
                       issue: titles.isEmpty ? nil : titles.joined(separator: " · "), preview: preview,
                       isWorking: isWorking, hasFailure: failureCount > 0)
    }

    @ViewBuilder private func statusIcon(_ status: Summary) -> some View {
        if status.isWorking {
            ProgressView().controlSize(.mini)
        } else {
            let symbol = status.hasFailure ? "exclamationmark.circle" : status.issue == nil ? "checkmark.circle" : "clock"
            let color: Color = status.hasFailure ? .red : .secondary
            Image(systemName: symbol).foregroundStyle(color)
        }
    }

    @ViewBuilder private func stepRow(_ item: ChatItem) -> some View {
        if case let .step(name, done, failed, output) = item.kind, let title = stepTitle(item) {
            let open = expandedStepID == item.id
            let active = !done && isCurrentTurnRunning
            let status = failed ? "Failed" : done ? "Completed" : active ? "Running" : "Unfinished"
            let symbol = failed ? "exclamationmark.circle" : done ? "checkmark" : active ? "circle.dotted" : "clock"
            Button { expandedStepID = open ? nil : item.id } label: {
                HStack(spacing: 8) {
                    Image(systemName: symbol).font(.caption).frame(width: 14)
                        .foregroundStyle(failed ? Color.red : .secondary)
                    Text(title).font(.callout).lineLimit(1)
                    Spacer(minLength: 0)
                    if failed || !done {
                        Text(status).font(.caption).foregroundStyle(failed ? Color.red : .secondary).fixedSize()
                    }
                    Image(systemName: open ? "chevron.down" : "chevron.right")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 4).padding(.vertical, 5)
                .background(open ? Theme.bubble : .clear, in: RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(title)
            .accessibilityValue(status)
            .accessibilityHint(open ? "Hide step details" : "Show step details")

            if open {
                let generic = ToolActivity.title(tool: name, summary: item.text, detail: item.toolInput)
                let input = item.toolInput ?? (item.text == name || item.text == generic ? "" : item.text)
                let details = VStack(alignment: .leading, spacing: 8) {
                    stepDetail("Tool", name)
                    stepDetail("Input", input)
                    stepDetail("Output", output)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                ViewThatFits(in: .vertical) {
                    details
                    ScrollView { details }
                }
                .frame(maxHeight: 180)
                .padding(10)
                .background(Theme.bubble, in: RoundedRectangle(cornerRadius: 8))
                .padding(.leading, 18).padding(.vertical, 4)
            }
        }
    }

    @ViewBuilder private func stepDetail(_ label: String, _ value: String) -> some View {
        if !value.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                Text(label).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(verbatim: value).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
        }
    }

    private func stepTitle(_ step: ChatItem) -> String? {
        guard case let .step(name, _, _, _) = step.kind else { return nil }
        return ToolActivity.activityTitle(tool: name, summary: step.text, detail: step.toolInput)
    }
}
