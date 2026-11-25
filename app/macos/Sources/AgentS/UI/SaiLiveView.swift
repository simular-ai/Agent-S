import SwiftUI
import WebKit
import AgentSCore

/// The chosen Sai computer's screen, from the Sai API's live page. The page's address carries a
/// session token, so it goes to the web view only.
struct SaiLiveView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession

    private enum Phase {
        case opening, waiting, starting, live(URL), failed(String)
    }

    @State private var phase: Phase = .opening
    @State private var attempt = 0
    @State private var fresh = false
    @State private var wakeRequested = false
    @State private var lastNewLink: Date?

    private var isOwnComputer: Bool { model.selectedSaiComputer?.kind == "own" }

    var body: some View {
        Group {
            if !model.hasSaiKey || model.saiMachine.isEmpty || model.saiMachine == SaiAPI.localMachine {
                ContentUnavailableView("No Sai computer", systemImage: "display",
                                       description: Text(model.chatBlocker ?? ""))
            } else {
                switch phase {
                case .opening:
                    ProgressView("Opening \(model.saiMachineName)…")
                case .starting:
                    ProgressView("Starting \(model.saiMachineName)…")
                case .waiting:
                    sleepingComputer
                case .live(let url):
                    LiveWebView(url: url, key: "\(model.saiKeyRevision):\(model.saiMachine)",
                                onNeedsNewLink: newLink, onComputerStopped: computerStopped)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Theme.hairline))
                case .failed(let failure):
                    ContentUnavailableView {
                        Label("Can't show \(model.saiMachineName)", systemImage: "display")
                    } description: {
                        Text(failure)
                    } actions: {
                        Button("Try again") { attempt += 1 }
                    }
                }
            }
        }
        .padding(16)
        .task(id: attempt) { await load() }
        .onChange(of: agent.isRunning) { _, running in
            if running, case .waiting = phase { attempt += 1 }
        }
    }

    private var sleepingComputer: some View {
        ContentUnavailableView {
            if model.saiScreenState == .asleep {
                Label("\(model.saiMachineName) is asleep", systemImage: "moon.zzz")
            } else {
                Label("\(model.saiMachineName) is offline", systemImage: "display")
            }
        } description: {
            Text(isOwnComputer
                 ? "Open Sai on that computer, then try again."
                 : "Send Sai a message to start the computer. Its screen will connect when it’s ready.")
        } actions: {
            if !isOwnComputer, model.selectedSaiComputer?.canWake == true {
                Button("Wake computer") {
                    wakeRequested = true
                    phase = .starting
                    attempt += 1
                }
            }
            Button("Try again") { attempt += 1 }
        }
    }

    /// The page says its link is used up or expired: make a new one, at most every two minutes.
    private func newLink() {
        guard case .live = phase else { return }
        if let lastNewLink, Date().timeIntervalSince(lastNewLink) < 120 { return }
        lastNewLink = Date()
        fresh = true
        attempt += 1
    }

    private func computerStopped() {
        guard case .live = phase else { return }
        fresh = true
        attempt += 1
    }

    private func showWaiting(for machine: String) {
        let state = model.selectedSaiComputer?.screenState
        model.noteSaiScreenState(isOwnComputer || state == .offline ? .offline : .asleep, for: machine)
        phase = .waiting
    }

    private func load() async {
        let machine = model.saiMachine
        let revision = model.saiKeyRevision
        let wake = wakeRequested
        wakeRequested = false

        func isCurrent() -> Bool {
            !Task.isCancelled && model.hasSaiKey && model.saiMachine == machine && model.saiKeyRevision == revision
        }
        guard isCurrent(), !machine.isEmpty, machine != SaiAPI.localMachine else { return }

        let state = model.saiScreenState
        let forceFresh = fresh || wake || state == .asleep || state == .offline || state == .starting
        let deadline = ContinuousClock.now + .seconds(180)
        var delay = 2
        var checkedInventory = false
        var starting = wake || state == .starting
        phase = starting ? .starting : .opening

        do {
            if wake {
                model.noteSaiScreenState(.starting, for: machine)
                try await SaiAPI.wake(machine: machine)
            }
            while isCurrent() {
                do {
                    let next = try await SaiAPI.liveURL(machine: machine, fresh: forceFresh, keyRevision: revision)
                    guard isCurrent() else { return }
                    guard let next else {
                        phase = .failed("Sai didn't return a live view for this computer.")
                        return
                    }
                    fresh = false
                    model.noteSaiScreenState(.running, for: machine)
                    phase = .live(next)
                    return
                } catch let failure as SaiAPI.Failure where failure.isMachineNotRunning {
                    guard isCurrent() else { return }
                    fresh = true
                    if !checkedInventory {
                        checkedInventory = true
                        await model.refreshSaiComputers()
                        guard isCurrent() else { return }
                    }
                    starting = starting || agent.isRunning || model.selectedSaiComputer?.screenState == .starting
                    guard !isOwnComputer, starting, ContinuousClock.now < deadline else {
                        showWaiting(for: machine)
                        return
                    }
                    model.noteSaiScreenState(.starting, for: machine)
                    phase = .starting
                    try await Task.sleep(for: .seconds(delay))
                    delay = min(delay * 2, 12)
                }
            }
        } catch {
            guard isCurrent() else { return }
            if starting { model.noteSaiScreenState(.unknown, for: machine) }
            phase = .failed(error.localizedDescription)
        }
    }
}

/// A live screen page. One web view is kept per computer for the whole launch, so closing the
/// panel, switching agents or maximizing never reconnects. The page's own reconnects are paced:
/// a dropped connection retries after 3 s, 10 s, 30 s, then every minute; "too many requests"
/// waits a minute; a used-up or expired link asks the app for a new one.
struct LiveWebView: NSViewRepresentable {
    let url: URL
    /// Which computer this is; its web view is reused whenever this one is shown again.
    let key: String
    var onNeedsNewLink: () -> Void = {}
    var onComputerStopped: () -> Void = {}

    @MainActor private static var views: [String: (view: WKWebView, coordinator: Coordinator)] = [:]

    /// Centers the Guacamole display at its existing scale, which the page uses to map input.
    /// Set element styles directly so the hosted page's policy can keep blocking inline stylesheets.
    private static let previewLayout = """
    (() => {
      const screen = document.querySelector("main#screen.screen");
      const status = document.querySelector("header.bar #status");
      if (!screen || !status) return;

      Object.assign(screen.style, {
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      });
      const keepDisplaySize = () => {
        const display = screen.firstElementChild;
        if (display) display.style.flex = "none";
      };
      new MutationObserver(keepDisplaySize).observe(screen, { childList: true });
      keepDisplaySize();

      Object.assign(status.style, {
        minWidth: "0",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis"
      });
      const title = status.parentElement.querySelector(".title");
      if (title) {
        title.style.whiteSpace = "nowrap";
        const compact = window.matchMedia("(max-width: 500px)");
        const updateTitle = () => { title.hidden = compact.matches; };
        compact.addEventListener("change", updateTitle);
        updateTitle();
      }
    })();
    """

    /// Watches the Sai live page's notice and presses its own Try again on a schedule.
    private static let pacer = """
    (() => {
      const delays = [3000, 10000, 30000, 60000];
      let failures = 0, timer = null, asked = false;
      const tell = (what) => window.webkit.messageHandlers.saiLive.postMessage(what);
      const check = () => {
        const notice = document.getElementById("notice");
        const retry = document.getElementById("retry");
        if (!notice) return;
        if (notice.hidden) { failures = 0; asked = false; if (timer) { clearTimeout(timer); timer = null; } return; }
        if (timer) return;
        const text = (document.getElementById("notice-text") || notice).textContent || "";
        if (/computer is not running|machine[_ -]not[_ -]running|computer is asleep/i.test(text)) {
          if (!asked) { asked = true; tell("computer-stopped"); }
          return;
        }
        if (/expired|revoked|too many sessions|ask for a new one|incomplete/i.test(text)) {
          if (!asked) { asked = true; tell("new-link"); }
          return;
        }
        if (!retry || retry.hidden) return;
        const wait = /too many requests/i.test(text) ? 65000 : delays[Math.min(failures, delays.length - 1)];
        failures += 1;
        timer = setTimeout(() => { timer = null; retry.click(); }, wait);
      };
      new MutationObserver(check).observe(document.documentElement, { attributes: true, childList: true, subtree: true, characterData: true });
      check();
    })();
    """

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> LiveContainer {
        let container = LiveContainer()
        attach(to: container)
        return container
    }

    func updateNSView(_ container: LiveContainer, context: Context) {
        attach(to: container)
    }

    /// Puts this computer's web view in `container` (taking it from wherever it was shown before)
    /// and loads `url` if it isn't the page already showing.
    private func attach(to container: LiveContainer) {
        let entry: (view: WKWebView, coordinator: Coordinator)
        if let kept = Self.views[key] {
            entry = kept
        } else {
            let coordinator = Coordinator()
            let configuration = WKWebViewConfiguration()
            configuration.userContentController.addUserScript(
                WKUserScript(source: Self.previewLayout, injectionTime: .atDocumentEnd, forMainFrameOnly: true))
            configuration.userContentController.addUserScript(
                WKUserScript(source: Self.pacer, injectionTime: .atDocumentEnd, forMainFrameOnly: true))
            configuration.userContentController.add(coordinator, name: "saiLive")
            entry = (WKWebView(frame: .zero, configuration: configuration), coordinator)
            Self.views[key] = entry
        }
        entry.coordinator.onNeedsNewLink = onNeedsNewLink
        entry.coordinator.onComputerStopped = onComputerStopped
        let view = entry.view
        let coordinator = entry.coordinator
        container.onMount = { [weak view, weak coordinator] in
            if let view { coordinator?.deliverNotice(from: view) }
        }
        if view.superview !== container {
            view.removeFromSuperview()
            view.frame = container.bounds
            view.autoresizingMask = [.width, .height]
            container.addSubview(view)
        }
        if coordinator.loaded != url {
            coordinator.loaded = url
            coordinator.pendingNotice = nil
            view.load(URLRequest(url: url))
        } else {
            coordinator.deliverNotice(from: view)
        }
    }

    final class LiveContainer: NSView {
        var onMount: () -> Void = {}

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if window != nil { onMount() }
        }
    }

    final class Coordinator: NSObject, WKScriptMessageHandler {
        var loaded: URL?
        var pendingNotice: String?
        var onNeedsNewLink: () -> Void = {}
        var onComputerStopped: () -> Void = {}

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.name == "saiLive", let event = message.body as? String,
                  let view = message.webView, message.frameInfo.isMainFrame else { return }
            pendingNotice = event
            deliverNotice(from: view)
        }

        func deliverNotice(from view: WKWebView) {
            guard pendingNotice != nil else { return }
            DispatchQueue.main.async { [weak view] in
                guard view?.window != nil, let event = self.pendingNotice else { return }
                self.pendingNotice = nil
                switch event {
                case "new-link": self.onNeedsNewLink()
                case "computer-stopped": self.onComputerStopped()
                default: break
                }
            }
        }
    }
}

/// The box's screen, view-only: screenshots taken with the user's box key every couple of seconds
/// while the box is awake (no sign-in needed). A sleeping box is left asleep. "Open" shows the
/// box's live page on box.simular.ai, where the person can type (it asks them to sign in).
struct BoxLiveView: View {
    @EnvironmentObject private var model: AppModel
    @State private var image: NSImage?
    @State private var asleep = false
    @State private var failure: String?
    @State private var viewURL: URL?
    @State private var waking = false
    @State private var attempt = 0
    /// Whether the person's input goes to the box. Off while the agent works.
    @State private var takeOver = false
    /// Inputs run one at a time, in order.
    @State private var inputs: Task<Void, Never>?

    private var controlling: Bool { takeOver && !model.agent.isRunning && !asleep }

    var body: some View {
        Group {
            if let image {
                BoxScreen(image: image, control: controlling, send: send)
                    .aspectRatio(image.size, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(RoundedRectangle(cornerRadius: 12)
                        .stroke(controlling ? Theme.accent : Theme.hairline, lineWidth: controlling ? 2 : 1))
                    .overlay(alignment: .topTrailing) {
                        HStack(spacing: 8) {
                            if asleep {
                                Text("Asleep").font(.caption)
                            } else if model.agent.isRunning {
                                Text("The agent is working").font(.caption)
                            } else {
                                Toggle("Take over", isOn: $takeOver).toggleStyle(.switch).controlSize(.mini).font(.caption)
                                    .accessibilityLabel("Take over the Box")
                                    .help(controlling ? "Release control and return to watching" : "Control the Box with your keyboard and mouse")
                            }
                            if let viewURL {
                                Link("Open", destination: viewURL).font(.caption.weight(.medium))
                                    .help("Open the box on box.simular.ai")
                            }
                        }
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.thinMaterial, in: Capsule())
                        .padding(10)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if asleep {
                ContentUnavailableView {
                    Label("Your Simular Box is asleep", systemImage: "moon.zzz")
                } description: {
                    Text("It wakes when the agent starts working, and its screen shows here.")
                } actions: {
                    Button(waking ? "Waking…" : "Wake it") { wake() }.disabled(waking)
                }
            } else if let failure {
                ContentUnavailableView {
                    Label("Can't show your Simular Box", systemImage: "cloud")
                } description: {
                    Text(failure)
                } actions: {
                    Link("Open box.simular.ai", destination: Box.portal)
                }
            } else if model.hasBoxKey {
                ProgressView("Opening your Simular Box…")
            } else {
                ContentUnavailableView("No Simular Box key", systemImage: "cloud",
                                       description: Text("Add your Simular Box key in the chat to see the box here."))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(16)
        .task(id: "\(model.boxName)-\(model.hasBoxKey)-\(attempt)") { await watch() }
    }

    /// Sends one input to the box, after the ones before it; shows the screenshot that comes back.
    private func send(_ action: [String: Any]) {
        guard controlling else { return }
        let box = model.boxName
        let previous = inputs
        inputs = Task {
            await previous?.value
            do {
                if let data = try await BoxAccount.act(action, box: box), let next = NSImage(data: data) {
                    image = next
                }
            } catch {
                failure = error.localizedDescription
            }
        }
    }

    private func wake() {
        waking = true
        Task {
            defer { waking = false }
            do {
                try await BoxAccount.wake(box: model.boxName)
                attempt += 1
            } catch {
                failure = error.localizedDescription
                asleep = false
            }
        }
    }

    /// While the agent works, refreshes the screen every second. Otherwise shows the last frame
    /// and checks only every 30 s, so watching never keeps an idle box awake or wakes a sleeping one.
    private func watch() async {
        image = nil
        failure = nil
        asleep = false
        guard model.hasBoxKey else { return }
        let box = model.boxName
        viewURL = try? await BoxAccount.viewURL(box: box)
        while !Task.isCancelled {
            do {
                if model.agent.isRunning || image == nil || takeOver {
                    if try await BoxAccount.isRunning(box: box) {
                        asleep = false
                        repeat {
                            if let data = try await BoxAccount.screenshot(box: box), let next = NSImage(data: data) {
                                image = next
                                failure = nil
                            }
                            try await Task.sleep(for: .seconds(1))
                        } while (model.agent.isRunning || takeOver) && !Task.isCancelled
                    } else {
                        asleep = true
                    }
                }
                // Idle: wake up often enough to notice the agent starting.
                for _ in 0..<(image == nil ? 10 : 30) where !model.agent.isRunning && !takeOver { try await Task.sleep(for: .seconds(1)) }
                if !model.agent.isRunning, image != nil {
                    asleep = !(try await BoxAccount.isRunning(box: box))
                }
            } catch is CancellationError {
                return
            } catch {
                if image == nil { failure = error.localizedDescription }
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }
}

/// The box's latest screenshot. While `control` is on, clicks, scrolling and keys over it become
/// box inputs (computer_act) at the matching point of the screenshot.
private struct BoxScreen: NSViewRepresentable {
    let image: NSImage
    let control: Bool
    let send: ([String: Any]) -> Void

    func makeNSView(context: Context) -> ScreenView { ScreenView() }

    func updateNSView(_ view: ScreenView, context: Context) {
        view.image = image
        view.control = control
        view.send = send
        view.needsDisplay = true
        if control, view.window?.firstResponder !== view { view.window?.makeFirstResponder(view) }
    }

    final class ScreenView: NSView {
        var image: NSImage?
        var control = false
        var send: ([String: Any]) -> Void = { _ in }
        /// Characters typed in a row, sent together.
        private var typed = ""
        private var flush: DispatchWorkItem?

        override var acceptsFirstResponder: Bool { control }
        override var isFlipped: Bool { true }

        override func draw(_ dirtyRect: NSRect) {
            // The view is flipped (clicks map from the top left); draw the image upright anyway.
            image?.draw(in: bounds, from: .zero, operation: .copy, fraction: 1, respectFlipped: true, hints: nil)
        }

        /// The point in screenshot pixels (the image is drawn to fill the view).
        private func pixel(_ event: NSEvent) -> (Double, Double)? {
            guard let image, bounds.width > 0, bounds.height > 0 else { return nil }
            let point = convert(event.locationInWindow, from: nil)
            let size = image.representations.first.map { NSSize(width: $0.pixelsWide, height: $0.pixelsHigh) } ?? image.size
            return ((point.x / bounds.width * size.width).rounded(), (point.y / bounds.height * size.height).rounded())
        }

        override func mouseDown(with event: NSEvent) {
            guard control else { return }
            window?.makeFirstResponder(self)
            guard let (x, y) = pixel(event) else { return }
            sendTyped()
            let action = event.clickCount >= 3 ? "triple_click" : event.clickCount == 2 ? "double_click" : "click"
            send(["action": action, "x": x, "y": y])
        }

        override func rightMouseDown(with event: NSEvent) {
            guard control, let (x, y) = pixel(event) else { return }
            sendTyped()
            send(["action": "right_click", "x": x, "y": y])
        }

        private var scrollSum: CGFloat = 0
        override func scrollWheel(with event: NSEvent) {
            guard control, let (x, y) = pixel(event) else { return }
            scrollSum += event.scrollingDeltaY
            // One box scroll step per notch-sized movement.
            let step: CGFloat = event.hasPreciseScrollingDeltas ? 40 : 1
            guard abs(scrollSum) >= step else { return }
            let amount = Int(abs(scrollSum) / step)
            let direction = scrollSum > 0 ? "up" : "down"
            scrollSum = 0
            send(["action": "scroll", "x": x, "y": y, "direction": direction, "amount": min(amount, 10)])
        }

        override func keyDown(with event: NSEvent) {
            guard control else { return super.keyDown(with: event) }
            let flags = event.modifierFlags.intersection([.command, .control, .option])
            if let name = Self.keyNames[event.keyCode] {
                sendTyped()
                if flags.isEmpty { send(["action": "key", "key": name]) }
                else { send(["action": "hotkey", "keys": Self.modifiers(flags) + [name]]) }
                return
            }
            if !flags.isEmpty, let character = event.charactersIgnoringModifiers?.lowercased(), !character.isEmpty {
                sendTyped()
                send(["action": "hotkey", "keys": Self.modifiers(flags) + [character]])
                return
            }
            guard let characters = event.characters, !characters.isEmpty else { return }
            typed += characters
            flush?.cancel()
            let work = DispatchWorkItem { [weak self] in self?.sendTyped() }
            flush = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35, execute: work)
        }

        private func sendTyped() {
            flush?.cancel()
            guard !typed.isEmpty else { return }
            send(["action": "type", "text": typed])
            typed = ""
        }

        /// The Mac's ⌘ is the box's Ctrl (copy, paste, select all…); ⌥ is Alt.
        private static func modifiers(_ flags: NSEvent.ModifierFlags) -> [String] {
            var keys: [String] = []
            if flags.contains(.command) || flags.contains(.control) { keys.append("ctrl") }
            if flags.contains(.option) { keys.append("alt") }
            return keys
        }

        private static let keyNames: [UInt16: String] = [
            36: "return", 76: "return", 48: "tab", 53: "escape", 51: "BackSpace", 117: "Delete",
            123: "Left", 124: "Right", 125: "Down", 126: "Up",
            115: "Home", 119: "End", 116: "Page_Up", 121: "Page_Down",
        ]
    }
}
