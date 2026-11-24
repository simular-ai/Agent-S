import AppKit
import SwiftUI

/// The blank portion of a custom header acts like a window title bar.
struct WindowHeaderSpace: View {
    var minWidth: CGFloat = 0

    var body: some View {
        WindowHeaderHitArea()
            .frame(minWidth: minWidth, maxWidth: .infinity)
            .frame(height: Theme.titleBarHeight)
            .accessibilityHidden(true)
    }
}

private struct WindowHeaderHitArea: NSViewRepresentable {
    func makeNSView(context: Context) -> HitArea { HitArea() }
    func updateNSView(_ view: HitArea, context: Context) {}

    final class HitArea: NSView {
        override var mouseDownCanMoveWindow: Bool { false }
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

        override func mouseDown(with event: NSEvent) {
            guard let window, window.attachedSheet == nil else { return }
            switch event.clickCount {
            case 1 where window.isMovable:
                window.performDrag(with: event)
            case 2 where window.styleMask.contains(.resizable) && !window.styleMask.contains(.fullScreen):
                window.zoom(nil)
            default:
                break
            }
        }
    }
}
