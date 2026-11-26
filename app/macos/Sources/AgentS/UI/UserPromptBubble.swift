import AppKit
import SwiftUI

struct UserPromptBubble: View {
    let prompt: String
    @State private var isClipped = false

    var body: some View {
        // Bound SwiftUI's layout work for large pasted prompts.
        let prefix = prompt.prefix(500)
        let exceedsPreviewLength = prefix.endIndex != prompt.endIndex
        let preview = Text(verbatim: exceedsPreviewLength ? String(prefix) + "…" : prompt)

        VStack(alignment: .trailing, spacing: 4) {
            preview
                .lineLimit(5)
                .fixedSize(horizontal: false, vertical: true)
                .background {
                    if !exceedsPreviewLength {
                        clippingProbe(for: preview)
                    }
                }
                .textSelection(.enabled)
                .foregroundStyle(Theme.onAccent)
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(RoundedRectangle(cornerRadius: 20).fill(Theme.accent))

            if exceedsPreviewLength || isClipped {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(prompt, forType: .string)
                } label: {
                    Label("Copy prompt", systemImage: "document.on.document")
                }
                .font(.caption)
                .buttonStyle(.borderless)
                .help("Copy the full prompt")
            }
        }
    }

    private func clippingProbe(for preview: Text) -> some View {
        GeometryReader { displayed in
            let height = displayed.size.height
            preview
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
                .hidden()
                .accessibilityHidden(true)
                .onGeometryChange(for: Bool.self) {
                    $0.size.height > height + 0.5
                } action: { clipped in
                    isClipped = clipped
                }
        }
    }
}
