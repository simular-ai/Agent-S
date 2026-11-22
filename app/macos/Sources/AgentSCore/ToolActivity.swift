import Foundation

/// Human-readable activity labels. Commands are inspected for display only.
public enum ToolActivity {
    public static func title(tool: String, summary: String, detail: String? = nil) -> String {
        let name = tool.lowercased()
        let kind = name.hasPrefix("mcp__box__") ? String(name.dropFirst("mcp__box__".count)) : name
        switch kind {
        case "bash", "bash-local", "bash_local", "shell", "exec_command", "command_execution":
            return commandTitle(detail ?? summary) ?? "Run a terminal command"
        case "mac", "mac_command", MacAgent.mcpCommandTool:
            return commandTitle(detail ?? summary) ?? "Run a Mac command"
        case "read", "read_file": return "Read a file"
        case "write", "write_file": return "Write a file"
        case "edit", "edit_file", "multiedit", "apply_patch", "file_change": return "Edit a file"
        case "list_dir", "list_files": return "List files"
        case "glob", "find_files": return "Find files"
        case "grep", "search_files": return "Search files"
        case "read_image", "view_image": return "View an image"
        default:
            let fallback = computerTitle(kind)
            if let provided = providerTitle(summary, tool: tool, detail: detail),
               fallback == nil || fallback == "Use the computer" || sameAction(provided, fallback) {
                return provided
            }
            return fallback ?? "Use a tool"
        }
    }

    /// Keep only a recognized input field, exactly as it was received.
    public static func detail(input: [String: Any]) -> String? {
        for key in ["command", "cmd", "file_path", "filePath", "path", "url", "query", "pattern"] {
            if let value = input[key] as? String, value.contains(where: { !$0.isWhitespace }) {
                return value
            }
        }
        return nil
    }

    private static func computerTitle(_ tool: String) -> String? {
        switch tool {
        case "browser_navigate", "browser_open": "Open a webpage"
        case "browser_read", "browser_fetch": "Read a webpage"
        case "browser_snapshot": "Inspect the webpage"
        case "browser_click": "Click a webpage element"
        case "browser_fill", "browser_type": "Type on the webpage"
        case "browser_key", "browser_press_key": "Press a key in the browser"
        case "browser_scroll": "Scroll the webpage"
        case "browser_back": "Go back in the browser"
        case "browser_tabs": "List browser tabs"
        case "screenshot", "computer_screenshot": "Capture the computer screen"
        case "computer_state": "Inspect the computer"
        case "computer_act", "desk", "gui_plan": "Use the computer"
        case "box_status": "Check the Box status"
        case "box_wake": "Wake the Box"
        case "box_hibernate": "Put the Box to sleep"
        case "box_save": "Save the Box"
        default: nil
        }
    }

    /// A provider may supply prose, but older summaries can also be raw commands, paths or input.
    private static func providerTitle(_ summary: String, tool: String, detail: String?) -> String? {
        let title = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, title.count <= 120,
              title.caseInsensitiveCompare(tool) != .orderedSame,
              detail?.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix(title) != true,
              !title.contains(where: { $0.isNewline || "/\\`$;|<>={}[]".contains($0) }) else { return nil }
        let words = title.split(whereSeparator: \.isWhitespace)
        guard words.count > 1, words[0].first?.isUppercase == true,
              providerTitleVerbs.contains(words[0].lowercased()) else { return nil }
        let punctuation = CharacterSet(charactersIn: ".,!?")
        guard !words.dropFirst().contains(where: {
            $0.trimmingCharacters(in: punctuation).contains(".") || $0.hasPrefix("-")
        }) else { return nil }
        return title
    }

    private static func sameAction(_ title: String, _ fallback: String?) -> Bool {
        let verb = title.prefix { !$0.isWhitespace }
        return fallback?.hasPrefix(verb + " ") == true
    }

    private static let providerTitleVerbs: Set<String> = [
        "browse", "capture", "check", "click", "close", "enter", "find", "go", "inspect",
        "list", "look", "navigate", "open", "press", "read", "review", "scroll", "search",
        "select", "switch", "take", "type", "wait",
    ]

    private static func commandTitle(_ source: String, wrapped: Bool = false) -> String? {
        let command = source.trimmingCharacters(in: .whitespacesAndNewlines)
        if let parsed = tokens(in: command[...]) {
            let action = withoutDirectoryChange(parsed)
            if !wrapped, action.count == 3, let executable = action[0].value,
               ["bash", "/bin/bash", "sh", "/bin/sh", "zsh", "/bin/zsh"].contains(executable),
               action[1].value == "-c" || action[1].value == "-lc",
               case let .word(script, quoted: true) = action[2] {
                return commandTitle(script, wrapped: true)
            }
            if isFileRedirect(action) { return "Write a file" }
            let words = action.compactMap(\.value)
            guard words.count == action.count, !words.isEmpty else { return nil }
            return simpleCommandTitle(words)
        }
        return writesLiteralHereDocument(command) ? "Write a file" : nil
    }

    private static func simpleCommandTitle(_ words: [String]) -> String? {
        let executable = words[0]
        let verb = words.dropFirst().first
        if executable == "mac" || executable.hasSuffix("/mac") {
            switch verb {
            case nil, "tree": return "Inspect the front Mac window"
            case "screenshot": return "Capture the Mac screen"
            case "apps": return "List open Mac apps"
            case "open": return "Open a Mac app"
            case "click": return "Click a Mac control"
            case "type": return "Type in a Mac app"
            case "key": return "Press keys on the Mac"
            default: return nil
            }
        }
        if (executable == "simulang" || executable.hasSuffix("/simulang")), verb == "run",
           words.count > 2, !words[2].hasPrefix("-") {
            return "Run a computer-use script"
        }
        if let verb {
            if executable == "ui" || executable == GuestTools.uiPath {
                switch verb {
                case "screenshot": return "Capture the computer screen"
                case "tree": return "Inspect the active window"
                case "open": return "Open a webpage"
                case "click": return "Click an on-screen element"
                case "type": return "Type in an app"
                case "key": return "Press keys in an app"
                default: return nil
                }
            }
            if executable == "web" || executable == GuestTools.webPath {
                switch verb {
                case "open", "snapshot", "read", "click", "type", "key", "back", "scroll", "tabs":
                    return computerTitle("browser_\(verb)")
                default: return nil
                }
            }
        }
        switch executable {
        case "cat", "/bin/cat":
            return words.dropFirst().contains { !$0.hasPrefix("-") && !$0.isEmpty } ? "Read a file" : nil
        case "ls", "/bin/ls": return "List files"
        case "pwd", "/bin/pwd": return "Check the working directory"
        default: return nil
        }
    }

    private static func withoutDirectoryChange(_ tokens: [ShellToken]) -> [ShellToken] {
        guard tokens.count > 3, tokens[0].value == "cd", let directory = tokens[1].value,
              !directory.isEmpty, !directory.hasPrefix("-"), tokens[2] == .then else { return tokens }
        return Array(tokens.dropFirst(3))
    }

    private static func isFileRedirect(_ tokens: [ShellToken]) -> Bool {
        guard tokens.count >= 3, tokens[0].value == "cat" || tokens[0].value == "/bin/cat",
              let target = tokens.last?.value, !target.isEmpty else { return false }
        let operatorIndex = tokens.count - 2
        let operation = tokens[operatorIndex]
        return (operation == .redirect || operation == .append)
            && tokens[..<operatorIndex].allSatisfy { $0.value != nil }
    }

    private static func writesLiteralHereDocument(_ command: String) -> Bool {
        let lines = command.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.count >= 2, let parsed = tokens(in: lines[0]) else { return false }
        let header = withoutDirectoryChange(parsed)
        guard let heredoc = header.firstIndex(of: .hereDocument), heredoc + 1 < header.count,
              case let .word(delimiter, quoted: true) = header[heredoc + 1], !delimiter.isEmpty,
              let last = lines.last, last == delimiter,
              !lines.dropFirst().dropLast().contains(where: { $0 == delimiter }) else { return false }
        var fileRedirect = header
        fileRedirect.removeSubrange(heredoc...heredoc + 1)
        return isFileRedirect(fileRedirect)
    }

    private enum ShellToken: Equatable {
        case word(String, quoted: Bool)
        case then
        case redirect
        case append
        case hereDocument

        var value: String? {
            guard case let .word(value, _) = self else { return nil }
            return value
        }
    }

    /// Accept only the shell syntax that the display labels can describe.
    private static func tokens(in command: Substring) -> [ShellToken]? {
        var input = command
        var tokens: [ShellToken] = []
        while let first = input.first {
            switch first {
            case " ", "\t":
                input = input.dropFirst()
            case "&" where input.hasPrefix("&&"):
                tokens.append(.then)
                input = input.dropFirst(2)
            case ">":
                let appending = input.hasPrefix(">>")
                tokens.append(appending ? .append : .redirect)
                input = input.dropFirst(appending ? 2 : 1)
            case "<" where input.hasPrefix("<<"):
                tokens.append(.hereDocument)
                input = input.dropFirst(2)
            default:
                guard let word = shellWord(&input) else { return nil }
                tokens.append(word)
            }
        }
        return tokens
    }

    private static func shellWord(_ input: inout Substring) -> ShellToken? {
        guard let first = input.first, first != "#",
              !first.isWhitespace, !"&|;<>".contains(first) else { return nil }
        var word = ""
        var quoted = false
        while let character = input.first, !character.isWhitespace, !"&|;<>".contains(character) {
            input = input.dropFirst()
            switch character {
            case "'", "\"":
                guard let end = input.firstIndex(of: character) else { return nil }
                let contents = input[..<end]
                if character == "\"", contents.contains(where: { "$`\\".contains($0) }) { return nil }
                word += contents
                input = input[input.index(after: end)...]
                quoted = true
            case "\\":
                guard let literal = input.first, !literal.isNewline else { return nil }
                word.append(literal)
                input = input.dropFirst()
                quoted = true
            case "$", "`", "(", ")", "{", "}":
                return nil
            default:
                word.append(character)
            }
        }
        return .word(word, quoted: quoted)
    }
}
