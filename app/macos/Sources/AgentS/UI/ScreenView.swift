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
    @State private var takeOver = false
    @State private var showOnePassword = false

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
                if let machine = model.vm.machine {
                    VMDisplay(machine: machine, takeOver: takeOver)
                        .aspectRatio(CGFloat(LinuxVM.screen.width) / CGFloat(LinuxVM.screen.height), contentMode: .fit)
                        .overlay(alignment: .topTrailing) {
                            if !takeOver {
                                Text("Watching")
                                    .font(.caption).padding(.horizontal, 8).padding(.vertical, 4)
                                    .background(.thinMaterial, in: Capsule())
                                    .padding(10)
                            }
                        }
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(takeOver ? Theme.accent : Theme.hairline, lineWidth: takeOver ? 2 : 1))
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
            } else if model.phase != .ready {
                Text(model.status.label).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 8)
            // Take over, 1Password and power belong to Agent S’s own computer.
            if model.agent.config.usesLocalComputer {
            Toggle(isOn: $takeOver) { Label("Take over", systemImage: "hand.raised") }
                .toggleStyle(.button)
                .labelStyle(.iconOnly)
                .disabled(model.vm.machine == nil)
                .help(takeOver ? "Your keyboard and mouse reach the computer" : "Watch only")
            Button { showOnePassword = true } label: { Label("Fill from 1Password", systemImage: "key") }
                .labelStyle(.iconOnly)
                .disabled(model.phase != .ready)
                .help("Type a login from your Mac's 1Password into the computer")
            powerButton
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

    @ViewBuilder private var powerButton: some View {
        switch model.phase {
        case .stopped, .failed:
            if model.files.isPrepared { Button("Turn on") { Task { await model.start() } } }
        case .ready, .booting:
            Button("Turn off") { Task { await model.stop() } }
        default:
            EmptyView()
        }
    }

    @ViewBuilder private var off: some View {
        if model.files.isPrepared {
            ContentUnavailableView {
                Label("The computer is off", systemImage: "display")
            } description: {
                Text("Turn it on to watch Sai work, or take over.")
            } actions: {
                if case .booting = model.phase {} else {
                    Button("Turn on") { Task { await model.start() } }
                        .buttonStyle(.borderedProminent).tint(Theme.accent)
                }
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

/// Apple's own view of the VM's display. Input passes through only while taking over; otherwise a
/// transparent layer above it swallows every event.
struct VMDisplay: NSViewRepresentable {
    let machine: VZVirtualMachine
    let takeOver: Bool

    func makeNSView(context: Context) -> NSView {
        let container = InputGate()
        let display = VZVirtualMachineView()
        display.virtualMachine = machine
        display.capturesSystemKeys = true
        if #available(macOS 14.0, *) { display.automaticallyReconfiguresDisplay = false }
        display.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(display)
        NSLayoutConstraint.activate([
            display.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            display.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            display.topAnchor.constraint(equalTo: container.topAnchor),
            display.bottomAnchor.constraint(equalTo: container.bottomAnchor),
        ])
        container.display = display
        container.passesInput = takeOver
        container.wantsLayer = true
        container.layer?.cornerRadius = 12
        container.layer?.masksToBounds = true
        return container
    }

    func updateNSView(_ view: NSView, context: Context) {
        guard let gate = view as? InputGate else { return }
        if gate.display?.virtualMachine !== machine { gate.display?.virtualMachine = machine }
        gate.passesInput = takeOver
        if takeOver { view.window?.makeFirstResponder(gate.display) }
    }
}

/// Lets events reach the VM view only while taking over.
final class InputGate: NSView {
    weak var display: VZVirtualMachineView?
    var passesInput = false

    override func hitTest(_ point: NSPoint) -> NSView? {
        passesInput ? super.hitTest(point) : (frame.contains(point) ? self : nil)
    }

    override var acceptsFirstResponder: Bool { !passesInput }
    override func keyDown(with event: NSEvent) { if passesInput { super.keyDown(with: event) } }
}
