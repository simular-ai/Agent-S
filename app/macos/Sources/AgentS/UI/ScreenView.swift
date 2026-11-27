import SwiftUI
import AgentSCore
import Virtualization

/// The computer, opened beside the chat: its state, Turn on/off, and its screen, live. Watching by
/// default; "Take over" lets your keyboard and mouse reach it, so a press while watching never
/// lands in the middle of the agent's work.
struct ComputerPanel: View {
    @EnvironmentObject private var model: AppModel
    @Binding var showComputer: Bool
    @Binding var showSettings: Bool
    @Binding var maximized: Bool
    @Binding var takeOver: Bool
    @State private var showOnePassword = false
    @State private var openingChromePasswords = false
    @State private var chromePasswordError: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            if model.computer == .box {
                BoxLiveView()
                    .frame(maxWidth: .infinity, maxHeight: maximized ? .infinity : 420)
                if !maximized {
                    Divider()
                    ContextView().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else if model.privacy == .thisMac {
                VStack(spacing: 12) {
                    Label("Working on this Mac", systemImage: "laptopcomputer").font(.headline)
                    Text("The agent uses your own screen, apps and files: watch it here on your Mac. Each step waits for your Allow in the chat.")
                        .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    MacAccessList()
                }
                .padding(16)
                .frame(maxWidth: .infinity, maxHeight: maximized ? .infinity : nil)
                if !maximized {
                    Divider()
                    ContextView().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else if model.harness == .sai {
                SaiLiveView()
                    .id("\(model.saiKeyRevision):\(model.saiMachine)")
                    .frame(maxWidth: .infinity, maxHeight: maximized ? .infinity : 420)
                if !maximized {
                    Divider()
                    ContextView().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
            Group {
                if model.isSavingSandbox {
                    ProgressView("Saving sandbox…").frame(height: 260)
                } else if model.phase == .stopping {
                    ProgressView(model.status.label).frame(height: 260)
                } else if showsSavedSandbox {
                    savedSandbox.frame(height: 260)
                } else if let machine = model.vm.machine {
                    VMDisplay(machine: machine, takeOver: takeOver)
                        .aspectRatio(VMDisplay.aspectRatio(for: machine), contentMode: .fit)
                        .overlay(alignment: .topTrailing) {
                            if !takeOver {
                                Text("Watching")
                                    .font(.caption).padding(.horizontal, 8).padding(.vertical, 4)
                                    .background(.thinMaterial, in: Capsule())
                                    .padding(10)
                                    .allowsHitTesting(false)
                            }
                        }
                        .overlay {
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(takeOver ? Theme.accent : Theme.hairline, lineWidth: takeOver ? 2 : 1)
                                .allowsHitTesting(false)
                        }
                        .padding(.horizontal, 16).padding(.top, 4).padding(.bottom, 12)
                } else {
                    off.frame(height: 260)
                }
            }
            // The screen sits right under the header; the conversation's context fills the rest.
            // Maximized, the screen takes all the room and the context is hidden.
            .frame(maxWidth: .infinity, maxHeight: maximized ? .infinity : nil)
            .layoutPriority(1)
            if !maximized {
                Divider()
                ContextView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            }
        }
        .background(Theme.sidebar)
        .sheet(isPresented: $showOnePassword) { OnePasswordSheet() }
        .alert("Chrome Passwords", isPresented: Binding(
            get: { chromePasswordError != nil },
            set: { if !$0 { chromePasswordError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(chromePasswordError ?? "")
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            ComputerToggle(showComputer: $showComputer)
            // The dot on the icon says "on"; words only when there is more to say.
            if model.privacy == .thisMac {
                Text("This Mac").font(.callout).foregroundStyle(.secondary).lineLimit(1)
            } else if model.computer == .box {
                Text(model.boxName.isEmpty ? "Simular Box" : model.boxName).font(.callout).foregroundStyle(.secondary).lineLimit(1)
            } else if model.harness == .sai {
                Text(model.saiMachineName).font(.callout).foregroundStyle(.secondary).lineLimit(1)
            } else if model.phase != .ready, !model.isSavingSandbox, !showsSavedSandbox {
                Text(model.status.label).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                    .truncationMode(.tail)
            }
            WindowHeaderSpace(minWidth: 8)
            if model.agent.config.usesLocalComputer {
                ViewThatFits(in: .horizontal) {
                    sandboxControls(.titleAndIcon)
                    sandboxControls(.iconOnly)
                }
            }
            CircleButton(systemImage: maximized ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right",
                         help: maximized ? "Back to the chat" : "Fill the window with the computer") {
                withAnimation(.easeInOut(duration: 0.2)) { maximized.toggle() }
            }
            .keyboardShortcut("f", modifiers: [.command, .shift])
        }
        .padding(.horizontal, 16)
        // Maximized, the panel starts at the window's left edge: clear the traffic lights.
        .padding(.leading, maximized ? 64 : 0)
        .frame(height: Theme.titleBarHeight)
    }

    private func sandboxControls<Style: LabelStyle>(_ labelStyle: Style) -> some View {
        HStack(spacing: 10) {
            Toggle(isOn: $takeOver) { Label("Take over", systemImage: "hand.raised") }
                .toggleStyle(.button)
                .disabled(model.vm.machine == nil || model.isSavingSandbox)
                .accessibilityLabel("Take over the sandbox")
                .help(takeOver ? "Release control and return to watching" : "Control the sandbox with your keyboard and mouse")
            passwordMenu
            powerButton
        }
        .labelStyle(labelStyle)
        .fixedSize(horizontal: true, vertical: false)
    }

    private var passwordMenu: some View {
        Menu {
            Button("Fill from 1Password…") { showOnePassword = true }
            Button("Open Chrome Passwords") { Task { await openChromePasswords() } }
                .disabled(openingChromePasswords)
        } label: {
            Label("Passwords", systemImage: "key")
        }
        .disabled(model.phase != .ready || model.isSavingSandbox)
        .accessibilityLabel("Passwords")
        .help("Fill from 1Password or search saved logins in Chrome on this Mac")
    }

    private func openChromePasswords() async {
        guard !openingChromePasswords else { return }
        let workspace = NSWorkspace.shared
        guard let chrome = workspace.urlForApplication(withBundleIdentifier: "com.google.Chrome") else {
            chromePasswordError = "Install Google Chrome on this Mac to open its password manager."
            return
        }

        openingChromePasswords = true
        defer { openingChromePasswords = false }
        let site = await model.currentSite()
        guard model.phase == .ready, !model.isSavingSandbox, model.agent.config.usesLocalComputer else { return }
        guard let site, !site.isEmpty else {
            chromePasswordError = "Open a website in the sandbox browser first, then try again."
            return
        }

        var search = URLComponents()
        search.scheme = "chrome"
        search.host = "password-manager"
        search.path = "/passwords"
        search.queryItems = [URLQueryItem(name: "q", value: site)]
        guard let url = search.url else { return }
        do {
            _ = try await workspace.open([url], withApplicationAt: chrome, configuration: .init())
        } catch {
            chromePasswordError = "Couldn’t open Google Chrome. Open it on this Mac and try again."
        }
    }

    private var showsSavedSandbox: Bool {
        guard model.hasSavedSandbox, model.vm.machine == nil else { return false }
        switch model.phase {
        case .stopped, .failed: return true
        default: return false
        }
    }

    @ViewBuilder private var powerButton: some View {
        if !model.isSavingSandbox {
            switch model.phase {
            case .stopped, .failed:
                if case .failed = model.phase, model.vm.state == .running {
                    Button("Try again") { Task { await model.start() } }
                }
                if model.vm.machine != nil {
                    turnOffButton
                } else if model.files.isPrepared, !showsSavedSandbox {
                    Button("Turn on") { Task { await model.start() } }
                }
            case .ready, .booting:
                turnOffButton
            default:
                EmptyView()
            }
        }
    }

    private var turnOffButton: some View {
        Button {
            Task { await model.stop() }
        } label: {
            Label("Turn off the sandbox", systemImage: "power")
        }
        .labelStyle(.iconOnly)
        .accessibilityLabel("Turn off the sandbox")
        .help("Turn off the sandbox")
    }

    private var savedSandbox: some View {
        ContentUnavailableView {
            Label("Sandbox saved", systemImage: "pause.circle")
        } description: {
            if case let .failed(message) = model.phase {
                Text(message).lineLimit(3)
            } else {
                Text("Resume to return to your open apps and windows.")
            }
        } actions: {
            VStack(spacing: 8) {
                Button("Resume") { Task { await model.start() } }
                    .buttonStyle(.borderedProminent).tint(Theme.accent)
                if case .failed = model.phase { DiscardSavedSandboxButton() }
            }
        }
    }

    @ViewBuilder private var off: some View {
        if case .booting = model.phase {
            ProgressView().accessibilityLabel(model.status.label)
        } else if model.files.isPrepared {
            ContentUnavailableView {
                Label("The computer is off", systemImage: "display")
            } description: {
                Text("Turn it on to watch Sai work, or take over.")
            } actions: {
                Button("Turn on") { Task { await model.start() } }
                    .buttonStyle(.borderedProminent).tint(Theme.accent)
            }
        } else {
            ContentUnavailableView {
                Label("No computer yet", systemImage: "display")
            } description: {
                Text("Set it up once: it downloads Debian and makes its disk.")
            } actions: {
                Button("Set up the computer") { showSettings = true }
                    .buttonStyle(.borderedProminent).tint(Theme.accent)
            }
        }
    }
}

struct DiscardSavedSandboxButton: View {
    @EnvironmentObject private var model: AppModel
    @State private var confirmsDiscard = false

    var body: some View {
        Button("Discard saved state and start…") { confirmsDiscard = true }
            .buttonStyle(.borderless)
            .alert("Discard saved sandbox state?", isPresented: $confirmsDiscard) {
                Button("Cancel", role: .cancel) {}
                Button("Discard and start", role: .destructive) {
                    guard case .failed = model.phase, model.hasSavedSandbox, model.vm.machine == nil,
                          !model.isSavingSandbox else { return }
                    Task { await model.start(restoreSavedState: false) }
                }
            } message: {
                Text("Open apps, windows, and unsaved work from the saved session will be lost. Files saved to the sandbox disk will stay.")
            }
    }
}

/// Apple's view renders and maps input within the same native viewport.
struct VMDisplay: NSViewRepresentable {
    let machine: VZVirtualMachine
    let takeOver: Bool

    static func aspectRatio(for machine: VZVirtualMachine) -> CGFloat {
        if let size = machine.graphicsDevices.first?.displays.first?.sizeInPixels,
           size.width > 0, size.height > 0 {
            return size.width / size.height
        }
        return CGFloat(LinuxVM.screen.width) / CGFloat(LinuxVM.screen.height)
    }

    func makeNSView(context: Context) -> InputGate {
        let container = InputGate()
        let display = SandboxDisplayView()
        display.virtualMachine = machine
        display.automaticallyReconfiguresDisplay = false
        display.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(display)
        NSLayoutConstraint.activate([
            display.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            display.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            display.topAnchor.constraint(equalTo: container.topAnchor),
            display.bottomAnchor.constraint(equalTo: container.bottomAnchor),
        ])
        container.display = display
        display.setTakingOver(takeOver)
        container.wantsLayer = true
        container.layer?.cornerRadius = 12
        container.layer?.masksToBounds = true
        return container
    }

    func updateNSView(_ gate: InputGate, context: Context) {
        guard let display = gate.display else { return }
        if display.virtualMachine !== machine {
            display.setTakingOver(false)
            display.virtualMachine = machine
        }
        display.setTakingOver(takeOver)
    }

    static func dismantleNSView(_ gate: InputGate, coordinator: ()) {
        gate.display?.setTakingOver(false)
        gate.display?.virtualMachine = nil
    }
}

/// Hit-testing protects clicks. AppKit can deliver tracked movement to the native view directly.
final class InputGate: NSView {
    weak var display: SandboxDisplayView?

    override func hitTest(_ point: NSPoint) -> NSView? {
        display?.isTakingOver == true ? super.hitTest(point) : (frame.contains(point) ? self : nil)
    }
}

class SandboxDisplayView: VZVirtualMachineView {
    private(set) var isTakingOver = false

    func setTakingOver(_ enabled: Bool) {
        guard isTakingOver != enabled else { return }
        isTakingOver = enabled
        capturesSystemKeys = enabled
        if enabled {
            window?.makeFirstResponder(self)
        } else if let window, let responder = window.firstResponder as? NSView, responder.isDescendant(of: self) {
            window.makeFirstResponder(nil)
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if isTakingOver { window?.makeFirstResponder(self) }
    }

    override var acceptsFirstResponder: Bool { isTakingOver && super.acceptsFirstResponder }

    func containsPointerEvent(_ event: NSEvent) -> Bool {
        guard isTakingOver, let window, event.window === window else { return false }
        return bounds.intersection(visibleRect).contains(convert(event.locationInWindow, from: nil))
    }

    override func mouseEntered(with event: NSEvent) {
        if containsPointerEvent(event) { super.mouseEntered(with: event) }
    }

    override func mouseMoved(with event: NSEvent) {
        if containsPointerEvent(event) { super.mouseMoved(with: event) }
    }

    override func mouseExited(with event: NSEvent) {
        if isTakingOver { super.mouseExited(with: event) }
    }

    override func scrollWheel(with event: NSEvent) {
        if containsPointerEvent(event) {
            super.scrollWheel(with: event)
        } else {
            nextResponder?.scrollWheel(with: event)
        }
    }
}
