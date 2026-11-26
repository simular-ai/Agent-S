import AppKit
import SwiftUI

struct UserPromptBubble: View {
    let prompt: String
    @State private var isClipped = false
    @State private var isExpanded = false
    @State private var isHovered = false

    var body: some View {
        // Bound SwiftUI's layout work for large pasted prompts.
        let prefix = prompt.prefix(500)
        let exceedsPreviewLength = prefix.endIndex != prompt.endIndex
        let displayedText = Text(verbatim: !isExpanded && exceedsPreviewLength ? String(prefix) + "…" : prompt)

        VStack(alignment: .trailing, spacing: 4) {
            if isExpanded || exceedsPreviewLength || isClipped {
                promptActions
            }

            displayedText
                .lineLimit(isExpanded ? nil : 5)
                .fixedSize(horizontal: false, vertical: true)
                .background {
                    if !isExpanded && !exceedsPreviewLength {
                        clippingProbe(for: displayedText)
                    }
                }
                .textSelection(.enabled)
                .foregroundStyle(Theme.onAccent)
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(RoundedRectangle(cornerRadius: 20).fill(Theme.accent))
        }
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
    }

    private var promptActions: some View {
        HStack(spacing: 12) {
            Button(action: copyPrompt) {
                Label("Copy prompt", systemImage: "document.on.document")
            }
            .opacity(isHovered ? 1 : 0)
            .disabled(!isHovered)
            .accessibilityHidden(!isHovered)
            .help("Copy the full prompt")

            Button {
                isExpanded.toggle()
            } label: {
                Label(isExpanded ? "Show less" : "Show more",
                      systemImage: isExpanded ? "chevron.up" : "chevron.down")
            }
            .accessibilityAction(named: Text("Copy prompt"), copyPrompt)
        }
        .font(.caption)
        .buttonStyle(.borderless)
    }

    private func copyPrompt() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(prompt, forType: .string)
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
