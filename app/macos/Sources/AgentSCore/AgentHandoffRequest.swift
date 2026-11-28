import Foundation

public enum AgentHandoffRequest {
    public static func withoutRecipientHeading(_ text: String, recipientName: String) -> String {
        guard !recipientName.isEmpty else { return text }
        let start = text.drop(while: \.isWhitespace)
        let firstLine = start.prefix(while: { !$0.isNewline })
        guard let inline = withoutHeading(firstLine, recipientName: recipientName) else { return text }

        let remaining = start[firstLine.endIndex...]
        if !inline.isEmpty { return String(inline) + remaining }

        let body = skippingBlankLines(remaining)
        let nextLine = body.prefix(while: { !$0.isNewline })
        return isHorizontalRule(nextLine) ? String(skippingBlankLines(body[nextLine.endIndex...])) : String(body)
    }

    /// Returns the rest of the line after a matching heading, or nil if it is ordinary text.
    private static func withoutHeading(_ line: Substring, recipientName: String) -> Substring? {
        var heading = line
        let hashes = heading.prefix(while: { $0 == "#" })
        if (1...6).contains(hashes.count), heading.dropFirst(hashes.count).first?.isWhitespace == true {
            heading = heading.dropFirst(hashes.count).drop(while: \.isWhitespace)
            if let closing = heading.range(of: #"[ \t]+#+[ \t]*$"#, options: .regularExpression) {
                heading = heading[..<closing.lowerBound]
            }
        }

        let emphasis = ["***", "___", "**", "__", "*", "_"].first(where: heading.hasPrefix)
        if let emphasis { heading = heading.dropFirst(emphasis.count) }
        guard let prefix = heading.range(of: "Request for @\(recipientName)", options: [.anchored, .caseInsensitive])
        else { return nil }

        var remainder = heading[prefix.upperBound...].drop(while: \.isWhitespace)
        let colonInside = remainder.first == ":"
        if colonInside { remainder = remainder.dropFirst().drop(while: \.isWhitespace) }
        if let emphasis {
            guard remainder.hasPrefix(emphasis) else { return nil }
            remainder = remainder.dropFirst(emphasis.count).drop(while: \.isWhitespace)
        }
        if !colonInside && remainder.first == ":" {
            return remainder.dropFirst().drop(while: \.isWhitespace)
        }
        return colonInside || remainder.isEmpty ? remainder : nil
    }

    private static func skippingBlankLines(_ text: Substring) -> Substring {
        var remainder = text
        while let end = remainder.firstIndex(where: \.isNewline),
              remainder[..<end].allSatisfy(\.isWhitespace) {
            remainder = remainder[remainder.index(after: end)...]
        }
        return remainder.allSatisfy(\.isWhitespace) ? remainder[remainder.endIndex...] : remainder
    }

    private static func isHorizontalRule(_ line: Substring) -> Bool {
        let indentation = line.prefix(while: { $0 == " " })
        let rule = line.dropFirst(indentation.count)
        guard indentation.count <= 3, let marker = rule.first, "-*_".contains(marker) else { return false }
        let marks = rule.filter { $0 != " " && $0 != "\t" }
        return marks.count >= 3 && marks.allSatisfy { $0 == marker }
    }
}
