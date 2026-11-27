import SwiftUI

/// A compact view of a This Mac conversation while the agent works in another app.
struct MacTurnPanelView: View {
    @ObservedObject var agent: AgentSession
    let showConversation: () -> Void

    private enum ScrollTarget: Hashable {
        case approval(String)
        case history(String)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
        }
        .background(Theme.canvas)
        .environmentObject(agent)
    }

    private var header: some View {
        HStack(spacing: 10) {
            AgentAvatar(config: agent.config, size: 30)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(agent.config.name).font(.headline).lineLimit(1)
                if pendingApprovals.isEmpty {
                    HStack(spacing: 5) {
                        if agent.isRunning { ProgressView().controlSize(.small) }
                        Text(agent.isRunning ? "Working…" : (agent.items.isEmpty ? "No activity" : "Turn ended"))
                    }
                    .font(.caption).foregroundStyle(.secondary)
                } else {
                    Label("Approval needed", systemImage: "hand.raised")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
            Button(action: showConversation) {
                Label("Open chat", systemImage: "arrow.up.right.square")
                    .labelStyle(.iconOnly)
            }
            .help("Open the full conversation")
            Button("Stop", role: .destructive) { agent.stop() }
                .disabled(!agent.isRunning)
        }
        .controlSize(.small)
        .padding(12)
    }

    private var transcript: some View {
        let history = recentHistory
        let latest = history.last?.id
        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 9) {
                    ForEach(history) { row in
                        ChatTranscriptRow(row: row, isRunning: agent.isRunning,
                                          onHandoffDecision: agent.decideHandoff)
                            .id(ScrollTarget.history(row.id))
                    }
                    ForEach(pendingApprovals) { item in
                        ChatRow(item: item)
                            .controlSize(.small)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(ScrollTarget.approval(item.id))
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onAppear { scrollToLatest(proxy, historyID: latest) }
            .onChange(of: pendingApprovals.map(\.id)) { _, _ in scrollToLatest(proxy, historyID: latest) }
            .onChange(of: agent.items.last?.id) { _, _ in
                if pendingApprovals.isEmpty { scrollToLatest(proxy, historyID: latest) }
            }
        }
    }

    private var pendingApprovals: [ChatItem] { agent.pendingMacApprovals }

    private var recentHistory: ArraySlice<ChatTranscript.Row> {
        ChatTranscript(agent.items).rows.filter { row in
            if case let .message(item) = row.content, case .approval(_, _, _, nil) = item.kind { return false }
            return true
        }.suffix(40)
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy, historyID: String?) {
        if let approval = pendingApprovals.first {
            proxy.scrollTo(ScrollTarget.approval(approval.id), anchor: .top)
        } else if let historyID {
            proxy.scrollTo(ScrollTarget.history(historyID), anchor: .bottom)
        }
    }
}
