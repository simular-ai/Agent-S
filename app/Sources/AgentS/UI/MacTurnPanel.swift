import AppKit
import Combine
import SwiftUI

/// Keeps This Mac turns visible independently of the main window and its selected conversation.
@MainActor
final class MacTurnPanelController {
    private let state = MacTurnPanelState()
    private var observations: [UUID: AnyCancellable] = [:]
    private var panel: NSPanel?
    private var openConversation: ((AgentSession) -> Void)?

    func show(_ session: AgentSession, openConversation: @escaping (AgentSession) -> Void) {
        self.openConversation = openConversation
        for previous in state.sessions where previous.id != session.id && !previous.isRunning {
            remove(previous)
        }
        if observations[session.id] == nil {
            state.sessions.append(session)
            state.approvals[session.id] = Set(session.pendingMacApprovals.map(\.id))
            observations[session.id] = session.$items.dropFirst().sink { [weak self, weak session] items in
                guard let self, let session else { return }
                self.approvalsChanged(session, items: items)
            }
        }
        state.selectedID = session.id
        reveal()
    }

    func remove(_ session: AgentSession) {
        observations[session.id] = nil
        state.sessions.removeAll { $0.id == session.id }
        state.approvals[session.id] = nil
        if state.selectedID == session.id { state.selectedID = state.sessions.last?.id }
        if state.sessions.isEmpty { panel?.orderOut(nil) }
    }

    private func approvalsChanged(_ session: AgentSession, items: [ChatItem]) {
        let current = Set(session.pendingMacApprovals(in: items).map(\.id))
        let previous = state.approvals[session.id, default: []]
        guard current != previous else { return }
        state.approvals[session.id] = current
        if !current.isSubset(of: previous) {
            state.selectedID = session.id
            reveal()
        }
    }

    private func reveal() {
        if panel == nil { panel = makePanel() }
        panel?.orderFrontRegardless()
    }

    private func makePanel() -> NSPanel {
        let panel = MacTurnWindow(contentRect: NSRect(x: 0, y: 0, width: 388, height: 354),
                                  styleMask: [.titled, .closable, .resizable, .nonactivatingPanel],
                                  backing: .buffered, defer: false)
        panel.title = "Agent Activity"
        panel.contentMinSize = NSSize(width: 354, height: 240)
        panel.isReleasedWhenClosed = false
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .canJoinAllApplications, .ignoresCycle]
        panel.contentView = MacTurnHostingView(rootView: MacTurnPanelContent(state: state) { [weak self] session in
            self?.openConversation?(session)
        })
        let frameName = "This Mac activity panel"
        if !panel.setFrameUsingName(frameName), let screen = NSApp.mainWindow?.screen ?? NSScreen.main {
            let visible = screen.visibleFrame
            panel.setFrameOrigin(NSPoint(x: max(visible.minX, visible.maxX - panel.frame.width - 18),
                                         y: visible.minY + 18))
        }
        panel.setFrameAutosaveName(frameName)
        return panel
    }
}

@MainActor
private final class MacTurnPanelState: ObservableObject {
    @Published var sessions: [AgentSession] = []
    @Published var selectedID: UUID?
    @Published var approvals: [UUID: Set<String>] = [:]
}

private struct MacTurnPanelContent: View {
    @ObservedObject var state: MacTurnPanelState
    let openConversation: (AgentSession) -> Void

    var body: some View {
        VStack(spacing: 0) {
            if state.sessions.count > 1 {
                Picker("Conversation", selection: $state.selectedID) {
                    ForEach(state.sessions) { session in
                        let waiting = !state.approvals[session.id, default: []].isEmpty
                        Text(waiting ? "\(session.config.name) · Approval needed" : session.config.name)
                            .tag(Optional(session.id))
                    }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .padding(.horizontal, 12)
                .padding(.top, 8)
            }
            if let session = state.sessions.first(where: { $0.id == state.selectedID }) {
                MacTurnPanelView(agent: session) { openConversation(session) }
                    .id(session.id)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.canvas)
    }
}

private final class MacTurnWindow: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

private final class MacTurnHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}
