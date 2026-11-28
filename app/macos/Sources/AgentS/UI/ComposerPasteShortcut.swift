import AppKit
import SwiftUI

/// Lets the active composer handle image paste while AppKit keeps normal text paste.
struct ComposerPasteShortcut: NSViewRepresentable {
    let isEnabled: Bool
    let paste: () -> Bool

    func makeNSView(context: Context) -> MonitorView { MonitorView() }

    func updateNSView(_ view: MonitorView, context: Context) {
        view.isEnabled = isEnabled
        view.paste = paste
    }

    static func dismantleNSView(_ view: MonitorView, coordinator: Void) {
        view.stopMonitoring()
    }

    final class MonitorView: NSView {
        var isEnabled = false
        var paste: (() -> Bool)?
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            stopMonitoring()
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                let handled = MainActor.assumeIsolated {
                    guard let self, self.isPasteShortcut(event) else { return false }
                    return self.paste?() == true
                }
                return handled ? nil : event
            }
        }

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        func stopMonitoring() {
            guard let monitor else { return }
            NSEvent.removeMonitor(monitor)
            self.monitor = nil
        }

        private func isPasteShortcut(_ event: NSEvent) -> Bool {
            let modifiers = event.modifierFlags.intersection([.command, .shift, .option, .control])
            guard isEnabled, modifiers == .command, event.charactersIgnoringModifiers?.lowercased() == "v",
                  let window, event.window === window,
                  let editor = window.firstResponder as? NSTextView, !editor.hasMarkedText() else { return false }
            return true
        }
    }
}
