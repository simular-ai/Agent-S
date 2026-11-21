import SwiftUI

/// A compact view of a This Mac conversation while the agent works in another app.
struct MacTurnPanelView: View {
    @ObservedObject var agent: AgentSession
    let showConversation: () -> Void

    private enum ScrollTarget: Hashable {
        case approval(String)
        case end
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
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 9) {
                    ForEach(recentHistory) { item in
                        ChatRow(item: item)
                    }
                    ForEach(pendingApprovals) { item in
                        ChatRow(item: item)
                            .controlSize(.small)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(ScrollTarget.approval(item.id))
                    }
                    Color.clear.frame(height: 1).id(ScrollTarget.end)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onAppear { scrollToLatest(proxy) }
            .onChange(of: pendingApprovals.map(\.id)) { _, _ in scrollToLatest(proxy) }
            .onChange(of: agent.items.last?.id) { _, _ in
                if pendingApprovals.isEmpty { scrollToLatest(proxy) }
            }
        }
    }

    private var pendingApprovals: [ChatItem] { agent.pendingMacApprovals }

    private var recentHistory: ArraySlice<ChatItem> {
        agent.items.filter { item in
            if case .approval(_, _, _, nil) = item.kind { return false }
            return true
        }.suffix(40)
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy) {
        if let approval = pendingApprovals.first {
            proxy.scrollTo(ScrollTarget.approval(approval.id), anchor: .top)
        } else {
            proxy.scrollTo(ScrollTarget.end, anchor: .bottom)
        }
    }
}
