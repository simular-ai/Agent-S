import SwiftUI
import AgentSCore

/// One window: conversations on the left, the chat in the middle, and the computer's screen in a
/// panel on the right that the Computer button opens and closes. Setup lives in a settings sheet.
struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var showComputer = false
    @State private var showSettings = false
    @State private var takeOverSandbox = false
    /// The computer's screen fills the window: sidebar, chat and context are hidden.
    @State private var maximized = false

    var body: some View {
        if model.needsFirstAgentSetup {
            NewAgentSheet(isFirstAgent: true)
                .frame(minWidth: 641, maxWidth: .infinity, maxHeight: .infinity)
                .background(Theme.canvas)
        } else if model.needsFirstAgentPreparation {
            FirstAgentPreparationView { showComputer = true }
                .frame(minWidth: 641)
                .background(Theme.canvas)
        } else {
            content.environmentObject(model.agent)
        }
    }

    private var content: some View {
        Group {
            if maximized, showComputer {
                ComputerPanel(showComputer: $showComputer, showSettings: $showSettings, maximized: $maximized,
                              takeOver: $takeOverSandbox)
            } else {
                layout
            }
        }
        .background(Theme.canvas)
        .ignoresSafeArea(.container, edges: .top)
        .onChange(of: showComputer) { _, open in
            if !open {
                maximized = false
                takeOverSandbox = false
            }
        }
        .onChange(of: model.agent.config.usesLocalComputer) { _, local in
            if !local { takeOverSandbox = false }
        }
        .onChange(of: model.vm.machine == nil) { _, stopped in
            if stopped { takeOverSandbox = false }
        }
        .onChange(of: model.sandboxSignInRequest) { showComputer = true }
        .sheet(isPresented: $showSettings) {
            VStack(spacing: 0) {
                SetupView()
                Divider()
                HStack {
                    Spacer()
                    Button("Done") { showSettings = false }.keyboardShortcut(.defaultAction)
                }
                .padding(12)
            }
            .frame(width: 560, height: 540)
        }
    }

    private var layout: some View {
        HStack(spacing: 0) {
            Sidebar(showSettings: $showSettings)
                .frame(width: 260)
                .layoutPriority(1)
            Divider()
            HSplitView {
                // Split panes host their own views and reapply the window's title bar inset.
                ChatView(showComputer: $showComputer)
                    .ignoresSafeArea(.container, edges: .top)
                    .frame(minWidth: 380, maxWidth: .infinity)
                // Keep a zero-width pane when closed so HSplitView updates the minimum width.
                // Its height must stay flexible so the empty pane doesn't constrain the chat.
                ZStack {
                    if showComputer {
                        ComputerPanel(showComputer: $showComputer, showSettings: $showSettings, maximized: $maximized,
                                      takeOver: $takeOverSandbox)
                    }
                }
                .ignoresSafeArea(.container, edges: .top)
                .frame(minWidth: showComputer ? 420 : 0,
                       idealWidth: showComputer ? Self.panelWidth : 0,
                       maxWidth: showComputer ? .infinity : 0,
                       minHeight: 0, maxHeight: .infinity)
            }
        }
    }
}

extension ContentView {
    static let panelWidth: CGFloat = 640
}

/// Colours and sizes shared by the views.
enum Theme {
    /// Black in light mode, white in dark mode: the app's one accent, no brand blue.
    static let accent = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? .white : .black
    })
    /// Text and marks drawn on `accent`.
    static let onAccent = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? .black : .white
    })
    static let canvas = Color(nsColor: .windowBackgroundColor)
    static let sidebar = Color.primary.opacity(0.035)
    static let bubble = Color.primary.opacity(0.06)
    static let card = Color(nsColor: .textBackgroundColor)
    static let hairline = Color.primary.opacity(0.1)
    /// Room for the window's traffic lights, which sit over the content.
    static let titleBarHeight: CGFloat = 52
}

/// The Agent S monogram from S.png, traced in the source image's coordinates.
struct SaiMark: Shape {
    private static let viewBox = CGRect(x: 359, y: 291, width: 345, height: 481)

    private static let logo = Path { path in
        path.move(to: CGPoint(x: 703, y: 291))
        path.addLine(to: CGPoint(x: 502, y: 291))
        path.addCurve(to: CGPoint(x: 359, y: 433.5), control1: CGPoint(x: 423, y: 291), control2: CGPoint(x: 359, y: 355))
        path.addCurve(to: CGPoint(x: 502, y: 576), control1: CGPoint(x: 359, y: 512), control2: CGPoint(x: 423, y: 576))
        path.addLine(to: CGPoint(x: 514, y: 576))
        path.addLine(to: CGPoint(x: 514, y: 487))
        path.addLine(to: CGPoint(x: 502, y: 487))
        path.addCurve(to: CGPoint(x: 448, y: 433.5), control1: CGPoint(x: 472, y: 487), control2: CGPoint(x: 448, y: 463))
        path.addCurve(to: CGPoint(x: 502, y: 380), control1: CGPoint(x: 448, y: 404), control2: CGPoint(x: 472, y: 380))
        path.addLine(to: CGPoint(x: 660, y: 380))
        path.closeSubpath()

        path.move(to: CGPoint(x: 550, y: 487))
        path.addLine(to: CGPoint(x: 561, y: 487))
        path.addCurve(to: CGPoint(x: 704, y: 629.5), control1: CGPoint(x: 640, y: 487), control2: CGPoint(x: 704, y: 551))
        path.addCurve(to: CGPoint(x: 561, y: 772), control1: CGPoint(x: 704, y: 708), control2: CGPoint(x: 640, y: 772))
        path.addLine(to: CGPoint(x: 403, y: 772))
        path.addLine(to: CGPoint(x: 360, y: 683))
        path.addLine(to: CGPoint(x: 561, y: 683))
        path.addCurve(to: CGPoint(x: 615, y: 629.5), control1: CGPoint(x: 591, y: 683), control2: CGPoint(x: 615, y: 659))
        path.addCurve(to: CGPoint(x: 561, y: 576), control1: CGPoint(x: 615, y: 600), control2: CGPoint(x: 591, y: 576))
        path.addLine(to: CGPoint(x: 550, y: 576))
        path.closeSubpath()
    }

    func path(in rect: CGRect) -> Path {
        let scale = min(rect.width / Self.viewBox.width, rect.height / Self.viewBox.height)
        let offset = CGPoint(x: rect.midX - Self.viewBox.midX * scale,
                             y: rect.midY - Self.viewBox.midY * scale)
        return Self.logo.applying(CGAffineTransform(translationX: offset.x, y: offset.y).scaledBy(x: scale, y: scale))
    }
}

/// Sai's headshot: the Sai mark in white on a round tile, with an optional status dot.
struct SaiAvatar: View {
    var size: CGFloat
    var status: Color? = nil

    var body: some View {
        Circle()
            .fill(Theme.accent)
            .frame(width: size, height: size)
            .overlay {
                SaiMark()
                    .fill(Theme.onAccent)
                    .frame(width: size * 0.56, height: size * 0.56)
            }
            .overlay(alignment: .bottomTrailing) {
                if let status {
                    Circle().fill(status)
                        .frame(width: size * 0.2, height: size * 0.2)
                        .overlay(Circle().stroke(Theme.canvas, lineWidth: max(2, size * 0.04)))
                }
            }
    }
}

/// The computer's button: a display icon with the computer's status dot. It sits at the top right
/// of the chat while the computer panel is closed, and at the top left of the panel while it is
/// open (gray), where it closes the panel.
struct ComputerToggle: View {
    @EnvironmentObject private var model: AppModel
    @Binding var showComputer: Bool

    var body: some View {
        Button { withAnimation(.easeInOut(duration: 0.2)) { showComputer.toggle() } } label: {
            Image(systemName: "display")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Color.primary)
                .frame(width: 38, height: 38)
                .background(Circle().fill(showComputer ? Theme.bubble.opacity(2) : Theme.card))
                .overlay(Circle().stroke(Theme.hairline))
                .overlay(alignment: .topTrailing) {
                    Circle().fill(dot)
                        .frame(width: 10, height: 10)
                        .overlay(Circle().stroke(Theme.canvas, lineWidth: 2))
                        .offset(x: 1, y: -1)
                }
        }
        .buttonStyle(.plain)
        .help("\(label) · \(showComputer ? "hide" : "show") the computer")
    }

    private var dot: Color {
        if model.agent.config.usesLocalComputer { return model.status.color }
        if model.harness == .sai {
            switch model.saiScreenState {
            case .running: return .green
            case .starting: return .orange
            case .unknown, .asleep, .offline: return .secondary
            }
        }
        return model.canChat ? .green : .secondary
    }

    private var label: String {
        if model.agent.config.usesLocalComputer { return model.status.label }
        if let blocker = model.chatBlocker { return blocker }
        if model.privacy == .thisMac { return "This Mac ready" }
        if model.harness == .sai {
            let name = model.saiMachineName
            switch model.saiScreenState {
            case .running: return "\(name) is running"
            case .starting: return "\(name) is starting"
            case .asleep: return "\(name) is asleep"
            case .offline: return "\(name) is offline"
            case .unknown: return "\(name) status unknown"
            }
        }
        return "Simular Box ready"
    }
}

/// A round icon button like the ones in the sidebar's corners.
struct CircleButton: View {
    let systemImage: String
    let help: String
    var filled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(filled ? Theme.onAccent : Color.primary)
                .frame(width: 34, height: 34)
                .background(Circle().fill(filled ? Theme.accent : Theme.card))
                .overlay(Circle().stroke(filled ? Color.clear : Theme.hairline))
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

struct Sidebar: View {
    @EnvironmentObject private var model: AppModel
    @Binding var showSettings: Bool
    @State private var creating: AgentLocation?
    @State private var renaming: AgentSession?
    @State private var newName = ""
    @State private var agentSettings = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                CircleButton(systemImage: "plus", help: "New agent") { creating = .linuxVM }
            }
            .padding(.horizontal, 16)
            .frame(height: Theme.titleBarHeight)


            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(AgentLocation.allCases) { location in
                        AgentSection(location: location, creating: $creating, renaming: $renaming, newName: $newName, agentSettings: $agentSettings)
                    }
                }
                .padding(.horizontal, 10)
            }

            HStack {
                Spacer()
                CircleButton(systemImage: "gearshape", help: "Settings") { showSettings = true }
            }
            .padding(16)
        }
        .background(Theme.sidebar)
        .task(id: model.hasOllamaAgents) {
            if model.hasOllamaAgents, !model.ollamaChecked { await model.refreshOllama() }
        }
        .sheet(item: $creating) { NewAgentSheet(initialLocation: $0) }
        .sheet(isPresented: $agentSettings) { AgentSettings() }
        .alert("Rename agent", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $newName)
            Button("Rename") { if let renaming { model.rename(renaming, to: newName) }; renaming = nil }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
    }
}

/// One side of the sidebar: its header and its agents, with Rename / New conversation / Delete.
struct AgentSection: View {
    @EnvironmentObject private var model: AppModel
    let location: AgentLocation
    @Binding var creating: AgentLocation?
    @Binding var renaming: AgentSession?
    @Binding var newName: String
    @Binding var agentSettings: Bool

    var body: some View {
        SideHeader(title: location.displayName, symbol: location.symbol)
            .padding(.top, location == .linuxVM ? 0 : 14)
        let list = model.agents(at: location)
        if list.isEmpty {
            Button(newAgentLabel) { creating = location }
                .buttonStyle(.plain).font(.callout).foregroundStyle(.secondary)
                .padding(.horizontal, 14).padding(.vertical, 6)
        }
        ForEach(list) { session in
            ConversationRow(session: session, selected: session.id == model.selectedAgentID) { model.select(session) }
                .contextMenu { menu(for: session) }
        }
    }

    private var newAgentLabel: String {
        switch location {
        case .linuxVM: "New sandbox agent…"
        case .thisMac: "New agent for this Mac…"
        case .cloud: "New cloud agent…"
        }
    }

    @ViewBuilder private func menu(for session: AgentSession) -> some View {
        Button("Agent Settings…") { model.select(session); agentSettings = true }
        Divider()
        Button("Rename…") { newName = session.config.name; renaming = session }
        Button("Choose picture…") { model.chooseAvatar(for: session) }
        Menu("Color") {
            ForEach(AgentAvatar.palette.indices, id: \.self) { index in
                Button(AgentAvatar.paletteNames[index]) { model.setAvatarColor(index, for: session) }
            }
        }
        if session.config.avatarImage != nil {
            Button("Use default picture") { model.clearAvatar(for: session) }
        }
        Divider()
        Button("New conversation") { session.reset() }
        Divider()
        Button("Delete", role: .destructive) { model.delete(session) }
            .disabled(model.agents.count < 2)
    }
}

struct SideHeader: View {
    let title: String
    let symbol: String

    var body: some View {
        Label(title, systemImage: symbol)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)
    }
}

/// An agent's headshot: its chosen picture, or the Sai mark on its own color.
struct AgentAvatar: View {
    let config: AgentConfig
    var size: CGFloat

    static let palette: [Color] = [
        Color(white: 0.08), Color(red: 0.20, green: 0.42, blue: 0.95), Color(red: 0.13, green: 0.60, blue: 0.36),
        Color(red: 0.93, green: 0.45, blue: 0.13), Color(red: 0.55, green: 0.30, blue: 0.85),
        Color(red: 0.86, green: 0.22, blue: 0.30), Color(red: 0.10, green: 0.58, blue: 0.62),
    ]
    static let paletteNames = ["Black", "Blue", "Green", "Orange", "Purple", "Red", "Teal"]

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Self.palette[(config.avatarColor ?? 0) % Self.palette.count]
                    .overlay {
                        Text(initial)
                            .font(.system(size: size * 0.46, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white)
                    }
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
    }

    /// The first letter of the agent's name ("S" for Sai), or "?" before it has one.
    private var initial: String {
        config.name.trimmingCharacters(in: .whitespacesAndNewlines).first.map { String($0).uppercased() } ?? "?"
    }

    private var image: NSImage? {
        config.avatarImage.flatMap { NSImage(contentsOf: GuestFiles.standard.avatars.appendingPathComponent($0)) }
    }
}

/// One agent's name, inference route, and latest message, with a spinner while it works.
struct ConversationRow: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var session: AgentSession
    let selected: Bool
    let select: () -> Void

    var body: some View {
        let inference = model.sidebarInference(for: session.config)
        let preview = session.lastLine ?? summary
        let detail = "\(inference.label) · \(inference.help)"
        let status = session.isRunning ? "\(detail), Working" : detail
        return Button(action: select) {
            HStack(spacing: 12) {
                AgentAvatar(config: session.config, size: 36)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(session.config.name).font(.body.weight(.medium)).lineLimit(1)
                        Image(systemName: inference.symbol)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 22, height: 16)
                            .background(Theme.bubble, in: Capsule())
                            .accessibilityHidden(true)
                    }
                    if !preview.isEmpty {
                        Text(preview).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                if session.isRunning { ProgressView().controlSize(.small) }
            }
            .padding(.horizontal, 10).padding(.vertical, 8)
            .background(RoundedRectangle(cornerRadius: 12).fill(selected ? Theme.bubble.opacity(1.6) : Color.clear))
            .contentShape(RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .help(detail)
        .accessibilityLabel(Text(session.config.name))
        .accessibilityValue(Text(preview.isEmpty ? status : "\(status), \(preview)"))
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    /// "Claude Code · gemma4", "Sai · auto": until there's a conversation.
    private var summary: String {
        let config = session.config
        let model: String
        if config.harness == .sai { model = config.saiModel }
        else if config.provider.usesAccount { model = "Automatic" }
        else { model = config.provider == .ollama ? config.ollamaModel : config.openRouterModel }
        return "\(config.harness.displayName) · \(model)"
    }
}

/// The computer's state in a few words and a colour.
struct ComputerStatus {
    let label: String
    let color: Color
}

extension AppModel {
    var status: ComputerStatus {
        switch phase {
        case .needsSetup: ComputerStatus(label: "Computer not set up", color: .secondary)
        case let .preparing(step): ComputerStatus(label: step.label, color: .orange)
        case .stopped: ComputerStatus(label: "Computer off", color: .secondary)
        case let .booting(step): ComputerStatus(label: step, color: .orange)
        case .ready: ComputerStatus(label: "Computer on", color: .green)
        case let .failed(message): ComputerStatus(label: message, color: .red)
        }
    }
}
