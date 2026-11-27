import AppKit
import SwiftUI
import AgentSCore

/// A compact view of a This Mac conversation while the agent works in another app.
struct MacTurnPanelView: View {
    @ObservedObject var agent: AgentSession
    @Binding var draft: String
    let isPanelKey: Bool
    let showConversation: () -> Void
    let onSubmit: (String) -> Bool
    @State private var showsActivity = false
    @State private var sendFailed = false
    @FocusState private var composerFocused: Bool

    private enum ScrollTarget: Hashable {
        case approval(String)
        case history(String)
        case activity
    }

    private struct TranscriptRevision: Equatable {
        let firstApproval: String?
        let lastMessage: ChatItem?
        let showsActivity: Bool
    }

    var body: some View {
        let rows = recentHistory
        let activity = rows.filter { if case .activity = $0.content { true } else { false } }
        let messages = rows.filter { if case .message = $0.content { true } else { false } }
        VStack(spacing: 0) {
            header
            Divider()
            if !activity.isEmpty || agent.isRunning {
                activityBar(activity)
                Divider()
            }
            transcript(messages: messages, activity: activity)
            composer
        }
        .background(Theme.canvas)
        .environmentObject(agent)
        .onChange(of: isEditingDraft) { _, editing in agent.setMacComposerEditing(editing) }
        .onChange(of: draft) { sendFailed = false }
        .onChange(of: pendingApprovals.first?.id) { _, approval in
            if approval != nil { showsActivity = false }
        }
        .onDisappear { agent.setMacComposerEditing(false) }
    }

    private var header: some View {
        let status = controlStatus
        let paused = agent.macControlPaused
        return HStack(spacing: 8) {
            Button(action: showConversation) {
                HStack(spacing: 8) {
                    AgentAvatar(config: agent.config, size: 30)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 4) {
                            Text(agent.config.name).font(.headline).lineLimit(1)
                            Image(systemName: "arrow.up.right.square")
                                .font(.caption2).foregroundStyle(.tertiary)
                                .accessibilityHidden(true)
                        }
                        HStack(spacing: 5) {
                            Circle().fill(status.color).frame(width: 6, height: 6)
                                .accessibilityHidden(true)
                            Text(status.title).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        }
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Open chat with \(agent.config.name)")
            .accessibilityValue(status.title)
            .help("Open the full conversation")
            Spacer(minLength: 0)
            Button(paused ? "Resume" : "Take over") {
                if paused {
                    composerFocused = false
                    agent.setMacComposerEditing(false)
                    agent.setMacTakeover(false)
                } else {
                    agent.setMacTakeover(true)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(Theme.accent)
            .disabled(!agent.isRunning || agent.isStopping)
            .help(paused ? "Resume this agent’s Mac actions" : "Pause before this agent’s next Mac action")
            Button("Stop", role: .destructive) { agent.stop() }
                .buttonStyle(.bordered)
                .tint(.red)
                .disabled(!agent.isRunning || agent.isStopping)
        }
        .controlSize(.small)
        .padding(12)
    }

    private var controlStatus: (title: String, color: Color) {
        if agent.isStopping { return ("Stopping…", .secondary) }
        if !pendingApprovals.isEmpty { return ("Approval needed", .orange) }
        if agent.isRunning, agent.macControlPaused { return ("New actions paused", .green) }
        if agent.isRunning { return ("Controlling this Mac", .orange) }
        return ("Ready", .secondary)
    }

    private func activityBar(_ activity: [ChatTranscript.Row]) -> some View {
        let step = latestStep(in: activity)
        let title = if let step, case let .step(name, _, _, _) = step.kind {
            ToolActivity.activityTitle(tool: name, summary: step.text, detail: step.toolInput)
        } else {
            "Working…"
        }
        return HStack(spacing: 8) {
            activityIcon(step)
                .frame(width: 14)
                .accessibilityHidden(true)
            Text(verbatim: title)
                .font(.callout).lineLimit(1)
                .help(title)
            Spacer(minLength: 0)
            if !activity.isEmpty {
                Button { showsActivity.toggle() } label: {
                    HStack(spacing: 5) {
                        Text("Activity").font(.caption)
                        Image(systemName: showsActivity ? "chevron.down" : "chevron.right")
                            .font(.caption2.weight(.semibold))
                    }
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .fixedSize()
                .accessibilityLabel(showsActivity ? "Hide activity" : "Show activity")
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Theme.bubble)
    }

    @ViewBuilder private func activityIcon(_ step: ChatItem?) -> some View {
        if !pendingApprovals.isEmpty {
            Image(systemName: "hand.raised").foregroundStyle(.secondary)
        } else if agent.isRunning, agent.macControlPaused {
            Image(systemName: "pause.circle").foregroundStyle(.secondary)
        } else if agent.isRunning {
            ProgressView().controlSize(.mini)
        } else if let step, case .step(_, _, true, _) = step.kind {
            Image(systemName: "exclamationmark.circle").foregroundStyle(.red)
        } else if let step, case .step(_, false, _, _) = step.kind {
            Image(systemName: "clock").foregroundStyle(.secondary)
        } else {
            Image(systemName: "checkmark.circle").foregroundStyle(.secondary)
        }
    }

    private func latestStep(in activity: [ChatTranscript.Row]) -> ChatItem? {
        let steps = activity.flatMap { row -> ArraySlice<ChatItem> in
            guard !agent.isRunning || row.isCurrentTurn, case let .activity(steps) = row.content else { return [] }
            return steps
        }
        if agent.isRunning, let active = steps.last(where: {
            if case .step(_, false, false, _) = $0.kind { true } else { false }
        }) {
            return active
        }
        return steps.last
    }

    private func transcript(messages: [ChatTranscript.Row], activity: [ChatTranscript.Row]) -> some View {
        let approvals = pendingApprovals
        let lastMessage: ChatItem? = if let last = messages.last, case let .message(item) = last.content { item } else { nil }
        let revision = TranscriptRevision(firstApproval: approvals.first?.id,
                                          lastMessage: showsActivity ? nil : lastMessage,
                                          showsActivity: showsActivity)
        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 9) {
                    if showsActivity, !activity.isEmpty {
                        VStack(spacing: 6) {
                            ForEach(activity) { row in
                                ChatTranscriptRow(row: row, isRunning: agent.isRunning)
                            }
                        }
                        .id(ScrollTarget.activity)
                    }
                    ForEach(messages) { row in
                        ChatTranscriptRow(row: row, isRunning: agent.isRunning,
                                          onHandoffDecision: agent.decideHandoff, neutralUserMessages: true)
                            .id(ScrollTarget.history(row.id))
                    }
                    ForEach(approvals) { item in
                        ChatRow(item: item)
                            .controlSize(.small)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(ScrollTarget.approval(item.id))
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .task(id: revision) {
                await Task.yield()
                guard !Task.isCancelled else { return }
                if revision.showsActivity {
                    proxy.scrollTo(ScrollTarget.activity, anchor: .top)
                } else if let approval = revision.firstApproval {
                    proxy.scrollTo(ScrollTarget.approval(approval), anchor: .top)
                } else if let latest = revision.lastMessage {
                    proxy.scrollTo(ScrollTarget.history(latest.id), anchor: .bottom)
                }
            }
        }
    }

    private var composer: some View {
        let status = composerStatus
        return VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .bottom, spacing: 6) {
                TextField("Message \(agent.config.name)…", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.callout)
                    .lineLimit(1...4)
                    .focused($composerFocused)
                    .onSubmit(submit)
                    .onKeyPress(.return, phases: [.down, .repeat]) { press in
                        let modifiers = press.modifiers.intersection([.shift, .command, .control, .option])
                        let window = NSApp.currentEvent?.window ?? NSApp.keyWindow
                        guard modifiers == .shift, let editor = window?.firstResponder as? NSTextView,
                              !editor.hasMarkedText() else { return .ignored }
                        editor.insertNewlineIgnoringFieldEditor(nil)
                        return .handled
                    }
                    .padding(.vertical, 6)
                Button(action: submit) {
                    Image(systemName: "arrow.up")
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(canSend ? Theme.onAccent : Color.secondary)
                        .frame(width: 28, height: 28)
                        .background(canSend ? Theme.accent : Theme.bubble, in: RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!canSend)
                .accessibilityLabel("Send")
                .help("Send message")
            }
            .padding(.leading, 8).padding(.trailing, 4).padding(.vertical, 4)
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 9))
            .overlay {
                RoundedRectangle(cornerRadius: 9)
                    .strokeBorder(composerFocused && isPanelKey ? Color.accentColor : Theme.hairline)
                    .allowsHitTesting(false)
            }
            Text(status ?? " ")
                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                .accessibilityHidden(status == nil)
        }
        .padding(.horizontal, 12).padding(.bottom, 10)
    }

    private var trimmedDraft: String { draft.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var canSend: Bool { !trimmedDraft.isEmpty && !agent.isStopping }
    private var isEditingDraft: Bool { composerFocused && isPanelKey && agent.isRunning && !agent.isStopping }

    private var composerStatus: String? {
        if sendFailed { return "Couldn’t send the message. Try again." }
        if isEditingDraft { return "Mac actions pause while you type." }
        let queued = agent.queuedFollowUps.count
        guard queued > 0 else { return nil }
        return queued == 1 ? "Follow-up queued" : "\(queued) follow-ups queued"
    }

    private func submit() {
        guard canSend else { return }
        if onSubmit(trimmedDraft) {
            draft = ""
            composerFocused = false
            agent.setMacComposerEditing(false)
        } else {
            sendFailed = true
        }
    }

    private var pendingApprovals: [ChatItem] { agent.pendingMacApprovals }

    private var recentHistory: ArraySlice<ChatTranscript.Row> {
        ChatTranscript(agent.items).rows.filter { row in
            if case let .message(item) = row.content, case .approval(_, _, _, nil) = item.kind { return false }
            return true
        }.suffix(40)
    }
}
