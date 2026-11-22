import AppKit
import Combine
import SwiftUI

/// Keeps local UI activity visible independently of the main window and its selected conversation.
@MainActor
final class MacTurnPanelController: NSObject, NSWindowDelegate {
    enum Presentation {
        case manual
        case macUI(needsApproval: Bool)
    }

    private let state = MacTurnPanelState()
    private var approvalObservations: [UUID: AnyCancellable] = [:]
    private var turnObservations: [UUID: AnyCancellable] = [:]
    private var manualSessionID: UUID?
    private var panel: NSPanel?
    private var openConversation: ((AgentSession) -> Void)?

    func show(_ session: AgentSession, presentation: Presentation, openConversation: @escaping (AgentSession) -> Void) {
        let id = session.id
        switch presentation {
        case .manual:
            if let previous = manualSessionID, previous != id { clearManualSession() }
            manualSessionID = id
        case let .macUI(needsApproval):
            guard session.isRunning else { return }
            if turnObservations[id] == nil {
                turnObservations[id] = session.$isRunning.dropFirst().sink { [weak self, weak session] isRunning in
                    guard !isRunning, let self, let session else { return }
                    self.turnObservations[id] = nil
                    if self.manualSessionID != id { self.remove(session) }
                }
            } else if !needsApproval {
                return
            }
        }

        self.openConversation = openConversation
        if approvalObservations[id] == nil {
            state.sessions.append(session)
            state.approvals[id] = Set(session.pendingMacApprovals.map(\.id))
            approvalObservations[id] = session.$items.dropFirst().sink { [weak self, weak session] items in
                guard let self, let session else { return }
                let approvals = Set(session.pendingMacApprovals(in: items).map(\.id))
                if self.state.approvals[id] != approvals { self.state.approvals[id] = approvals }
            }
        }
        state.selectedID = id
        reveal()
    }

    func remove(_ session: AgentSession) {
        let id = session.id
        approvalObservations[id] = nil
        turnObservations[id] = nil
        if manualSessionID == id { manualSessionID = nil }
        state.sessions.removeAll { $0.id == id }
        state.approvals[id] = nil
        if state.selectedID == id { state.selectedID = state.sessions.last?.id }
        if state.sessions.isEmpty { panel?.orderOut(nil) }
    }

    func windowWillClose(_ notification: Notification) {
        clearManualSession()
    }

    private func clearManualSession() {
        guard let id = manualSessionID else { return }
        manualSessionID = nil
        if turnObservations[id] == nil, let session = state.sessions.first(where: { $0.id == id }) {
            remove(session)
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
        panel.delegate = self
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
    // The nonactivating panel can accept keyboard copy when a user clicks selectable reply text.
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

private final class MacTurnHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}
