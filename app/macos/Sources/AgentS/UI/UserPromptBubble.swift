import AppKit
import SwiftUI

struct UserPromptBubble: View {
    let prompt: String
    var senderName: String? = nil
    @State private var isClipped = false
    @State private var isExpanded = false
    @State private var isHovered = false

    private var incoming: Bool { senderName != nil }

    var body: some View {
        // Bound SwiftUI's layout work for large pasted prompts.
        let prefix = prompt.prefix(500)
        let exceedsPreviewLength = prefix.endIndex != prompt.endIndex
        let displayedText = Text(verbatim: !isExpanded && exceedsPreviewLength ? String(prefix) + "…" : prompt)
        let showsActions = isExpanded || exceedsPreviewLength || isClipped
        let shape = RoundedRectangle(cornerRadius: incoming ? 14 : 20)

        VStack(alignment: incoming ? .leading : .trailing, spacing: 4) {
            header(showsActions: showsActions)

            displayedText
                .lineLimit(isExpanded ? nil : 5)
                .fixedSize(horizontal: false, vertical: true)
                .background {
                    if !isExpanded && !exceedsPreviewLength {
                        clippingProbe(for: displayedText)
                    }
                }
                .textSelection(.enabled)
                .foregroundStyle(incoming ? Color.primary : Theme.onAccent)
                .padding(.horizontal, incoming ? 14 : 16)
                .padding(.vertical, incoming ? 8 : 10)
                .frame(maxWidth: incoming && showsActions ? .infinity : nil, alignment: .leading)
                .background(incoming ? Theme.bubble : Theme.accent, in: shape)
                .overlay {
                    if incoming { shape.strokeBorder(Theme.hairline.opacity(0.6)).allowsHitTesting(false) }
                }
        }
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
    }

    @ViewBuilder private func header(showsActions: Bool) -> some View {
        if let senderName {
            HStack(spacing: 8) {
                Text(verbatim: senderName).font(.callout.weight(.semibold)).lineLimit(1)
                if showsActions {
                    Spacer(minLength: 8)
                    promptActions.fixedSize(horizontal: true, vertical: false)
                }
            }
            .frame(minHeight: 24, alignment: .leading)
        } else if showsActions {
            promptActions
        }
    }

    private var promptActions: some View {
        HStack(spacing: 12) {
            Button(action: copyPrompt) {
                Label("Copy prompt", systemImage: "document.on.document")
                    .labelStyle(.iconOnly)
            }
            .opacity(isHovered ? 1 : 0)
            .disabled(!isHovered)
            .accessibilityLabel("Copy prompt")
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
