import AppKit
import struct Markdown.Document
import struct Markdown.HTMLBlock
import struct Markdown.InlineHTML
import protocol Markdown.Markup
import SwiftMarkdownView
import SwiftUI

struct ChatMarkdown: View {
    let markdown: String

    var body: some View {
        let blocks = MarkdownContent(parsing: markdown).blocks.map(\.forChat)
        if Self.containsRawHTML(markdown) || (blocks.isEmpty && !markdown.isEmpty) {
            Text((try? AttributedString(
                markdown: markdown,
                options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
            )) ?? AttributedString(markdown))
            .textSelection(.enabled)
        } else {
            formattedReply(blocks)
        }
    }

    private func formattedReply(_ blocks: [MarkdownBlock]) -> some View {
        let codeBlocks = blocks.flatMap(\.codeBlocks)

        return VStack(alignment: .leading, spacing: 4) {
            MarkdownSelectableText(MarkdownContent(blocks: blocks), theme: Self.theme)
                .markdownImagePolicy(.bundleOnly)

            if !codeBlocks.isEmpty {
                HStack {
                    Spacer()
                    if codeBlocks.count == 1, let block = codeBlocks.first {
                        Button { copy(block.code) } label: { copyLabel }
                            .buttonStyle(.borderless)
                            .help("Copy code")
                    } else {
                        Menu {
                            ForEach(Array(codeBlocks.enumerated()), id: \.offset) { index, block in
                                Button("Copy \(block.language ?? "code") block \(index + 1)") { copy(block.code) }
                            }
                        } label: { copyLabel }
                            .menuStyle(.borderlessButton)
                            .menuIndicator(.hidden)
                            .help("Copy a code block")
                    }
                }
                .foregroundStyle(.secondary)
            }
        }
    }

    /// SwiftMarkdownView omits raw HTML nodes, which can include reply text in a block.
    private static func containsRawHTML(_ source: String) -> Bool {
        guard source.contains("<") else { return false }
        var pending: [any Markup] = [Document(parsing: source)]
        while let node = pending.popLast() {
            if node is HTMLBlock || node is InlineHTML { return true }
            pending.append(contentsOf: node.children)
        }
        return false
    }

    private var copyLabel: some View {
        Label("Copy code", systemImage: "document.on.document").labelStyle(.iconOnly)
    }

    private func copy(_ code: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
    }

    private static let theme = MarkdownTextTheme(
        baseFont: .systemFont(ofSize: 13),
        codeFont: .monospacedSystemFont(ofSize: 12, weight: .regular),
        textColor: .labelColor,
        secondaryColor: .secondaryLabelColor,
        headingColor: .labelColor,
        linkColor: .linkColor,
        inlineCodeForeground: .labelColor,
        inlineCodeBackground: .labelColor.withAlphaComponent(0.06),
        codeBlockBackground: .labelColor.withAlphaComponent(0.06),
        quoteBarColor: .tertiaryLabelColor,
        ruleColor: .separatorColor,
        paragraphSpacing: 9,
        lineHeightMultiple: 1.15,
        indentStep: 18,
        headingSizes: [20, 17, 15, 13, 13, 13],
        headingWeight: .semibold
    )
}

private extension MarkdownBlock {
    /// Preserve reply line breaks and keep image and diagram content readable without fetching it.
    var forChat: Self {
        switch self {
        case let .paragraph(inlines):
            return .paragraph(inlines.map(\.forChat))
        case let .heading(level, inlines):
            return .heading(level: level, content: inlines.map(\.forChat))
        case let .aside(kind, blocks):
            return .aside(kind: kind, content: blocks.map(\.forChat))
        case let .unorderedList(items):
            return .unorderedList(items.map(\.forChat))
        case let .orderedList(start, items):
            return .orderedList(start: start, items: items.map(\.forChat))
        case let .table(table):
            return .table(TableData(
                headerRow: .init(cells: table.headerRow.cells.map { $0.map(\.forChat) }),
                bodyRows: table.bodyRows.map { row in .init(cells: row.cells.map { $0.map(\.forChat) }) },
                columnAlignments: table.columnAlignments
            ))
        case let .mermaid(source):
            return .codeBlock(language: "mermaid", code: source)
        case .codeBlock, .math, .thematicBreak:
            return self
        }
    }

    var codeBlocks: [(language: String?, code: String)] {
        switch self {
        case let .codeBlock(language, code):
            return [(language, code)]
        case let .aside(_, blocks):
            return blocks.flatMap(\.codeBlocks)
        case let .unorderedList(items), let .orderedList(_, items):
            return items.flatMap { $0.blocks.flatMap(\.codeBlocks) }
        default:
            return []
        }
    }
}

private extension ListItem {
    var forChat: Self { ListItem(blocks: blocks.map(\.forChat), isChecked: isChecked) }
}

private extension MarkdownInline {
    var forChat: Self {
        switch self {
        case .softBreak:
            return .hardBreak
        case let .image(_, alt, _):
            return .text(alt.isEmpty ? "[Image]" : "[Image: \(alt)]")
        case let .emphasis(inlines):
            return .emphasis(inlines.map(\.forChat))
        case let .strong(inlines):
            return .strong(inlines.map(\.forChat))
        case let .strikethrough(inlines):
            return .strikethrough(inlines.map(\.forChat))
        case let .link(destination, title, inlines):
            return .link(destination: destination, title: title, content: inlines.map(\.forChat))
        case .text, .code, .inlineMath, .hardBreak:
            return self
        }
    }
}
