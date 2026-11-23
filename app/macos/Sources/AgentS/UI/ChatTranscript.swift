/// The rendered rows of a conversation, with adjacent tool steps shown as one activity.
struct ChatTranscript {
    struct Row: Identifiable {
        let id: String
        let content: Content
        let isCurrentTurn: Bool
    }

    enum Content {
        case message(ChatItem)
        case activity(ArraySlice<ChatItem>)
    }

    let rows: [Row]
    let lastItem: ChatItem?

    init(_ items: [ChatItem]) {
        // Streaming can begin with line breaks. Keep them in the source in case text follows.
        let items = items.filter { !$0.isBlankAssistantMessage }
        lastItem = items.last
        let turnStart = items.lastIndex { $0.kind == .user && $0.isSteering != true }
        var rows: [Row] = []
        var index = items.startIndex

        while index < items.endIndex {
            let item = items[index]
            let isCurrentTurn = turnStart.map { index >= $0 } ?? false
            let content: Content

            if case .step = item.kind {
                var end = index + 1
                while end < items.endIndex, case .step = items[end].kind {
                    end += 1
                }
                content = .activity(items[index..<end])
                index = end
            } else {
                content = .message(item)
                index += 1
            }

            rows.append(Row(id: item.id, content: content, isCurrentTurn: isCurrentTurn))
        }

        self.rows = rows
    }
}
