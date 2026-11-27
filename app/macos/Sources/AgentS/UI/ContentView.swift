import SwiftUI
import UniformTypeIdentifiers
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
            NewAgentSheet(.firstAgent)
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
                WindowHeaderSpace()
                    .padding(.leading, 64)
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
                Link(destination: URL(string: "https://discord.com/invite/Uru8geWNp4")!) {
                    DiscordMark()
                        .fill(.black)
                        .frame(width: 18, height: 14)
                        .frame(width: 34, height: 34)
                        .background(.white, in: Circle())
                        .overlay(Circle().stroke(Theme.hairline))
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Join Discord")
                .help("Join Discord")
                Spacer()
                CircleButton(systemImage: "gearshape", help: "Settings") { showSettings = true }
            }
            .padding(16)
        }
        .background(Theme.sidebar)
        .task(id: model.hasOllamaAgents) {
            if model.hasOllamaAgents, !model.ollamaChecked { await model.refreshOllama() }
        }
        .sheet(item: $creating) { NewAgentSheet(.newAgent($0)) }
        .sheet(isPresented: $agentSettings) { AgentSettings() }
        .alert("Rename agent", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $newName)
            Button("Rename") { if let renaming { model.rename(renaming, to: newName) }; renaming = nil }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
    }
}

/// Discord's Clyde symbol from https://discord.com/branding, in the original SVG coordinates.
private struct DiscordMark: Shape {
    private static let size = CGSize(width: 65, height: 48)

    private static let logo = Path { path in
        path.move(to: CGPoint(x: 41.2351, y: 0))
        path.addCurve(to: CGPoint(x: 39.5556, y: 3.397), control1: CGPoint(x: 40.6164, y: 1.09866), control2: CGPoint(x: 40.0607, y: 2.2352))
        path.addCurve(to: CGPoint(x: 25.0584, y: 3.397), control1: CGPoint(x: 34.7569, y: 2.67719), control2: CGPoint(x: 29.8697, y: 2.67719))
        path.addCurve(to: CGPoint(x: 23.3788, y: 0), control1: CGPoint(x: 24.5658, y: 2.2352), control2: CGPoint(x: 23.9976, y: 1.09866))
        path.addCurve(to: CGPoint(x: 10.3085, y: 4.02841), control1: CGPoint(x: 18.8705, y: 0.770324), control2: CGPoint(x: 14.4759, y: 2.12155))
        path.addCurve(to: CGPoint(x: 0.925755, y: 39.9432), control1: CGPoint(x: 2.04967, y: 16.2652), control2: CGPoint(x: -0.185531, y: 28.1863))
        path.addCurve(to: CGPoint(x: 16.951, y: 47.9874), control1: CGPoint(x: 5.76238, y: 43.517), control2: CGPoint(x: 11.1799, y: 46.2447))
        path.addCurve(to: CGPoint(x: 20.3859, y: 42.4562), control1: CGPoint(x: 18.2517, y: 46.2447), control2: CGPoint(x: 19.4009, y: 44.3883))
        path.addCurve(to: CGPoint(x: 14.981, y: 39.88), control1: CGPoint(x: 18.5169, y: 41.7616), control2: CGPoint(x: 16.7111, y: 40.8903))
        path.addCurve(to: CGPoint(x: 16.307, y: 38.8824), control1: CGPoint(x: 15.4356, y: 39.5517), control2: CGPoint(x: 15.8776, y: 39.2107))
        path.addCurve(to: CGPoint(x: 48.3449, y: 38.8824), control1: CGPoint(x: 26.4475, y: 43.6559), control2: CGPoint(x: 38.1917, y: 43.6559))
        path.addCurve(to: CGPoint(x: 49.6708, y: 39.88), control1: CGPoint(x: 48.7742, y: 39.236), control2: CGPoint(x: 49.2162, y: 39.577))
        path.addCurve(to: CGPoint(x: 44.2533, y: 42.4688), control1: CGPoint(x: 47.9408, y: 40.9029), control2: CGPoint(x: 46.1349, y: 41.7616))
        path.addCurve(to: CGPoint(x: 47.6882, y: 48), control1: CGPoint(x: 45.2383, y: 44.4009), control2: CGPoint(x: 46.3875, y: 46.2573))
        path.addCurve(to: CGPoint(x: 63.7134, y: 39.9684), control1: CGPoint(x: 53.4593, y: 46.2573), control2: CGPoint(x: 58.8768, y: 43.5422))
        path.addCurve(to: CGPoint(x: 54.3054, y: 4.04104), control1: CGPoint(x: 65.0268, y: 26.3299), control2: CGPoint(x: 61.4656, y: 14.5099))
        path.addCurve(to: CGPoint(x: 41.2478, y: 0.0252565), control1: CGPoint(x: 50.1507, y: 2.13418), control2: CGPoint(x: 45.7561, y: 0.782952))
        path.addLine(to: CGPoint(x: 41.2351, y: 0))
        path.closeSubpath()

        path.move(to: CGPoint(x: 21.8003, y: 32.7072))
        path.addCurve(to: CGPoint(x: 16.0923, y: 26.3804), control1: CGPoint(x: 18.6811, y: 32.7072), control2: CGPoint(x: 16.0923, y: 29.8785))
        path.addCurve(to: CGPoint(x: 21.7876, y: 20.041), control1: CGPoint(x: 16.0923, y: 22.8824), control2: CGPoint(x: 18.5801, y: 20.041))
        path.addCurve(to: CGPoint(x: 27.4956, y: 26.3804), control1: CGPoint(x: 24.9952, y: 20.041), control2: CGPoint(x: 27.5461, y: 22.895))
        path.addCurve(to: CGPoint(x: 21.8003, y: 32.7072), control1: CGPoint(x: 27.4451, y: 29.8658), control2: CGPoint(x: 24.9826, y: 32.7072))
        path.closeSubpath()

        path.move(to: CGPoint(x: 42.8389, y: 32.7072))
        path.addCurve(to: CGPoint(x: 37.1436, y: 26.3804), control1: CGPoint(x: 39.7071, y: 32.7072), control2: CGPoint(x: 37.1436, y: 29.8785))
        path.addCurve(to: CGPoint(x: 42.8389, y: 20.041), control1: CGPoint(x: 37.1436, y: 22.8824), control2: CGPoint(x: 39.6314, y: 20.041))
        path.addCurve(to: CGPoint(x: 48.5343, y: 26.3804), control1: CGPoint(x: 46.0465, y: 20.041), control2: CGPoint(x: 48.5848, y: 22.895))
        path.addCurve(to: CGPoint(x: 42.8389, y: 32.7072), control1: CGPoint(x: 48.4838, y: 29.8658), control2: CGPoint(x: 46.0213, y: 32.7072))
        path.closeSubpath()
    }

    func path(in rect: CGRect) -> Path {
        let size = Self.size
        let scale = min(rect.width / size.width, rect.height / size.height)
        return Self.logo.applying(CGAffineTransform(
            a: scale, b: 0, c: 0, d: scale,
            tx: rect.midX - size.width * scale / 2,
            ty: rect.midY - size.height * scale / 2
        ))
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
    /// The agent being dragged in this section; drags from other sections or apps leave it nil.
    @State private var dragging: UUID?

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
                .onDrag {
                    dragging = session.id
                    return NSItemProvider(object: session.id.uuidString as NSString)
                }
                .onDrop(of: [.text], delegate: AgentDrop(target: session.id, dragging: $dragging, model: model))
                .accessibilityAction(named: "Move up") { withAnimation { model.moveAgent(session, by: -1) } }
                .accessibilityAction(named: "Move down") { withAnimation { model.moveAgent(session, by: 1) } }
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

/// Reorders a section's agents live as a dragged row passes over the others, and saves on drop.
struct AgentDrop: DropDelegate {
    let target: UUID
    @Binding var dragging: UUID?
    let model: AppModel

    func validateDrop(info: DropInfo) -> Bool { dragging != nil }

    func dropEntered(info: DropInfo) {
        guard let dragging, dragging != target else { return }
        withAnimation(.easeInOut(duration: 0.15)) { model.moveAgent(dragging, to: target) }
    }

    func dropUpdated(info: DropInfo) -> DropProposal? {
        DropProposal(operation: dragging == nil ? .forbidden : .move)
    }

    func performDrop(info: DropInfo) -> Bool {
        guard dragging != nil else { return false }
        dragging = nil
        model.saveAgents()
        return true
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
    private let name: String
    private let avatarImage: String?
    private let avatarColor: Int?
    var size: CGFloat

    init(config: AgentConfig, size: CGFloat) {
        self.init(name: config.name, avatarImage: config.avatarImage, avatarColor: config.avatarColor, size: size)
    }

    init(name: String, avatarImage: String?, avatarColor: Int?, size: CGFloat) {
        self.name = name
        self.avatarImage = avatarImage
        self.avatarColor = avatarColor
        self.size = size
    }

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
                Self.palette[(avatarColor ?? 0) % Self.palette.count]
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
        name.trimmingCharacters(in: .whitespacesAndNewlines).first.map { String($0).uppercased() } ?? "?"
    }

    private var image: NSImage? {
        avatarImage.flatMap { NSImage(contentsOf: GuestFiles.standard.avatars.appendingPathComponent($0)) }
    }
}

/// One agent's name, inference route, and latest message, with a spinner while it works.
struct ConversationRow: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var session: AgentSession
    @State private var rowHeight: CGFloat = 0
    @State private var dropAfter: Bool?
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
        .onDrag { dragItem() }
        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { rowHeight = $0 }
        .onDrop(of: [dragType], delegate: AgentRowDrop(type: dragType, rowHeight: rowHeight, dropAfter: $dropAfter) { id, after in
            withAnimation(.easeInOut(duration: 0.18)) { model.moveAgent(id, relativeTo: session, after: after) }
        })
        .overlay(alignment: dropAfter == true ? .bottom : .top) {
            if dropAfter != nil {
                Capsule().fill(Theme.accent)
                    .frame(height: 2)
                    .padding(.horizontal, 6)
                    .allowsHitTesting(false)
            }
        }
    }

    private var dragType: UTType {
        // A separate drag type for each computer group limits which rows accept a drop.
        UTType(exportedAs: "ai.simular.agent-s.sidebar-agent.\(session.config.privacy.location.rawValue.lowercased())", conformingTo: .data)
    }

    private func dragItem() -> NSItemProvider {
        let provider = NSItemProvider()
        let data = Data(session.id.uuidString.utf8)
        provider.registerDataRepresentation(forTypeIdentifier: dragType.identifier, visibility: .ownProcess) { completion in
            completion(data, nil)
            return nil
        }
        return provider
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

private struct AgentRowDrop: DropDelegate {
    let type: UTType
    let rowHeight: CGFloat
    @Binding var dropAfter: Bool?
    let move: @MainActor (UUID, Bool) -> Void

    func dropUpdated(info: DropInfo) -> DropProposal? {
        dropAfter = info.location.y >= rowHeight / 2
        return DropProposal(operation: .move)
    }

    func dropExited(info: DropInfo) { dropAfter = nil }

    func performDrop(info: DropInfo) -> Bool {
        dropAfter = nil
        guard let item = info.itemProviders(for: [type]).first else { return false }
        let after = info.location.y >= rowHeight / 2
        let move = move
        _ = item.loadDataRepresentation(forTypeIdentifier: type.identifier) { data, _ in
            guard let data, let id = UUID(uuidString: String(decoding: data, as: UTF8.self)) else { return }
            Task { @MainActor in move(id, after) }
        }
        return true
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
        case .stopping: ComputerStatus(label: "Shutting down the sandbox…", color: .orange)
        case .ready: ComputerStatus(label: "Computer on", color: .green)
        case let .failed(message): ComputerStatus(label: message, color: .red)
        }
    }
}
