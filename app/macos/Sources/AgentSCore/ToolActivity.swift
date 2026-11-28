import Foundation

/// Human-readable activity labels. Commands are inspected for display only.
public enum ToolActivity {
    public static func title(tool: String, summary: String, detail: String? = nil) -> String {
        let kind = toolKind(tool)
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

    /// A more specific transcript label that keeps raw arguments and results in the details.
    public static func activityTitle(tool: String, summary: String, detail: String? = nil) -> String {
        let specific: String?
        switch toolKind(tool) {
        case "bash", "bash-local", "bash_local", "shell", "exec_command", "command_execution",
             "mac", "mac_command", MacAgent.mcpCommandTool:
            return commandActivityTitle(detail ?? summary) ?? title(tool: tool, summary: summary, detail: detail)
        case "read", "read_file":
            specific = detail.flatMap(fileName).map { "Read \($0)" }
        case "write", "write_file":
            specific = detail.flatMap(fileName).map { "Write \($0)" }
        case "edit", "edit_file", "multiedit", "apply_patch", "file_change":
            specific = detail.flatMap(fileName).map { "Edit \($0)" }
        case "read_image", "view_image":
            specific = detail.flatMap(fileName).map { "View \($0)" }
        case "browser_open", "browser_navigate":
            specific = detail.flatMap(webHost).map { "Open \($0)" }
        case "browser_read", "browser_fetch":
            specific = detail.flatMap(webHost).map { "Read \($0)" }
        default:
            specific = nil
        }
        // Free-form provider titles can describe text being typed or include other input values.
        return specific ?? title(tool: tool, summary: tool, detail: detail)
    }

    private static func toolKind(_ tool: String) -> String {
        let name = tool.lowercased()
        return name.hasPrefix("mcp__box__") ? String(name.dropFirst("mcp__box__".count)) : name
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
            if !wrapped, let script = wrappedScript(action) {
                return commandTitle(script, wrapped: true)
            }
            if isFileRedirect(action) { return "Write a file" }
            let words = action.compactMap(\.value)
            guard words.count == action.count, !words.isEmpty else { return browserSequenceTitle(action) }
            return simpleCommandTitle(words)
        }
        return writesLiteralHereDocument(command) ? "Write a file" : nil
    }

    private static func commandActivityTitle(_ source: String, wrapped: Bool = false) -> String? {
        let command = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsed = tokens(in: command[...]) else { return nil }
        let action = withoutDirectoryChange(parsed)
        if !wrapped, let script = wrappedScript(action) { return commandActivityTitle(script, wrapped: true) }
        if isFileRedirect(action), let path = action.last?.value, let name = fileName(path) {
            return "Write \(name)"
        }
        let words = action.compactMap(\.value)
        guard words.count == action.count, let first = words.first, !first.isEmpty,
              let command = first.split(separator: "/").last else { return nil }
        let arguments = words.dropFirst()
        switch command {
        case "git": return gitActivity(arguments)
        case "gh": return githubActivity(arguments)
        case "rg", "grep": return "Search file contents"
        case "find":
            let mutations: Set<String> = ["-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf"]
            return arguments.contains(where: mutations.contains) ? "Run a find command" : "Find files"
        case "ls": return "List files"
        case "cat":
            let paths = arguments.first == "--" ? arguments.dropFirst() : arguments
            guard paths.count == 1, let path = paths.first, !path.hasPrefix("-"), let name = fileName(path) else { return nil }
            return "Read \(name)"
        case "web", "ui", "mac":
            return browserTargetActivity(words)
        case "curl", "wget":
            // Fetch options can also send data, upload files, or write to disk.
            return "Run \(command)"
        default:
            return developmentActivity(command, arguments: arguments)
        }
    }

    private static func wrappedScript(_ tokens: [ShellToken]) -> String? {
        guard tokens.count == 3, let executable = tokens[0].value,
              ["bash", "/bin/bash", "sh", "/bin/sh", "zsh", "/bin/zsh"].contains(executable),
              tokens[1].value == "-c" || tokens[1].value == "-lc",
              case let .word(script, quoted: true) = tokens[2] else { return nil }
        return script
    }

    private static func gitActivity(_ arguments: ArraySlice<String>) -> String {
        var options = arguments
        while let first = options.first {
            switch first {
            case "-C" where options.count > 2: options = options.dropFirst(2)
            case "--no-pager", "--paginate": options = options.dropFirst()
            default: return gitVerb(first)
            }
        }
        return "Run a Git command"
    }

    private static func gitVerb(_ verb: String) -> String {
        switch verb {
        case "status": return "Check Git status"
        case "diff": return "Review Git changes"
        case "log", "reflog": return "Read Git history"
        case "show": return "Inspect a Git revision"
        case "fetch": return "Fetch Git updates"
        case "pull": return "Pull Git changes"
        case "push": return "Push Git changes"
        case "add": return "Stage Git changes"
        case "commit": return "Commit Git changes"
        default: return "Run a Git command"
        }
    }

    private static func githubActivity(_ arguments: ArraySlice<String>) -> String {
        switch Array(arguments.prefix(2)) {
        case ["pr", "view"]: return "Inspect a pull request"
        case ["pr", "list"]: return "List pull requests"
        case ["pr", "checks"]: return "Check pull request status"
        case ["pr", "diff"]: return "Review pull request changes"
        case ["run", "view"], ["run", "watch"]: return "Check a GitHub Actions run"
        case ["run", "list"]: return "List GitHub Actions runs"
        case ["issue", "view"]: return "Inspect a GitHub issue"
        case ["issue", "list"]: return "List GitHub issues"
        case ["release", "view"]: return "Inspect a GitHub release"
        case ["release", "list"]: return "List GitHub releases"
        default: return "Run a GitHub CLI command"
        }
    }

    private static func developmentActivity(_ command: Substring, arguments: ArraySlice<String>) -> String? {
        let first = arguments.first
        switch command {
        case "swift":
            if first == "test" { return "Run Swift tests" }
            if first == "build" { return "Build a Swift package" }
            return "Run a Swift command"
        case "xcodebuild":
            if arguments.contains("-showBuildSettings") { return "Inspect Xcode build settings" }
            if arguments.contains("-list") { return "List Xcode targets" }
            if arguments.contains("test") { return "Run Xcode tests" }
            if arguments.contains("build") { return "Build with Xcode" }
            return "Run an Xcode command"
        case "npm", "pnpm", "yarn", "bun":
            let task = first == "run" ? arguments.dropFirst().first : first
            if task == "test" { return "Run JavaScript tests" }
            if task == "build" { return "Build a JavaScript project" }
            return "Run \(command)"
        case "pytest": return "Run Python tests"
        case "python", "python3":
            return Array(arguments.prefix(2)) == ["-m", "pytest"] ? "Run Python tests" : nil
        case "cargo":
            if first == "test" { return "Run Rust tests" }
            if first == "build" { return "Build a Rust project" }
            return "Run a Cargo command"
        case "go":
            if first == "test" { return "Run Go tests" }
            if first == "build" { return "Build a Go project" }
            return "Run a Go command"
        default: return nil
        }
    }

    private static func browserTargetActivity(_ words: [String]) -> String? {
        guard words.count >= 3 else { return nil }
        if let verb = browserVerb(words), ["open", "read"].contains(verb),
           let host = webHost(words[2]) {
            return verb == "open" ? "Open \(host)" : "Read \(host)"
        }
        if (words[0] == "ui" || words[0] == GuestTools.uiPath), words[1] == "open",
           let host = webHost(words[2]) { return "Open \(host)" }
        if (words[0] == "mac" || words[0].hasSuffix("/mac")), words[1] == "open",
           words.count == 4, let host = webHost(words[3]) { return "Open \(host)" }
        return nil
    }

    private static func fileName(_ path: String) -> String? {
        guard !path.contains("://"), !path.hasSuffix("/"),
              !path.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains),
              let component = path.split(separator: "/").last else { return nil }
        let name = String(component)
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ._-")
        guard name.count <= 48, let first = name.first, first.isLetter || first.isNumber,
              name.unicodeScalars.allSatisfy(allowed.contains),
              name.range(of: #"(?i)(?:credential|password|passwd|secret|token|private[-_. ]?key|^sk-)"#,
                         options: .regularExpression) == nil else { return nil }
        let ext = URL(fileURLWithPath: name).pathExtension.lowercased()
        let commonTypes: Set<String> = [
            "c", "cpp", "css", "csv", "go", "h", "html", "java", "jpeg", "jpg", "js", "json", "jsx",
            "kt", "log", "md", "pdf", "png", "py", "rs", "sh", "sql", "svg", "swift", "toml", "ts",
            "tsx", "txt", "xml", "yaml", "yml",
        ]
        return commonTypes.contains(ext) || ["Makefile", "Dockerfile", "LICENSE", "README"].contains(name) ? name : nil
    }

    private static func webHost(_ target: String) -> String? {
        guard !target.contains(where: \.isWhitespace),
              let url = URLComponents(string: target),
              let scheme = url.scheme?.lowercased(), ["http", "https"].contains(scheme),
              let host = url.host?.lowercased(), host.count <= 64, !host.hasSuffix("."),
              host.range(of: #"\A[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\z"#, options: .regularExpression) != nil,
              !host.contains("..") else { return nil }
        return host
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
                case "launch":
                    guard words.count > 2, !words[2].hasPrefix("-"),
                          words[2].contains(where: { !$0.isWhitespace }) else { return nil }
                    return "Launch an app"
                case "click": return "Click an on-screen element"
                case "type": return "Type in an app"
                case "key": return "Press keys in an app"
                default: return nil
                }
            }
        }
        if let browser = browserVerb(words) { return computerTitle("browser_\(browser)") }
        switch executable {
        case "cat", "/bin/cat":
            return words.dropFirst().contains { !$0.hasPrefix("-") && !$0.isEmpty } ? "Read a file" : nil
        case "ls", "/bin/ls": return "List files"
        case "pwd", "/bin/pwd": return "Check the working directory"
        default: return nil
        }
    }

    private static func browserVerb(_ words: [String]) -> String? {
        guard words.count >= 2, words[0] == "web" || words[0] == GuestTools.webPath else { return nil }
        switch words[1] {
        case "open", "snapshot", "read", "click", "type", "key", "back", "scroll", "tabs": return words[1]
        default: return nil
        }
    }

    /// Browser steps often pause for a page to settle, then inspect and filter its output.
    private static func browserSequenceTitle(_ tokens: [ShellToken]) -> String? {
        let commands = tokens.split(omittingEmptySubsequences: false) { $0 == .then || $0 == .sequence }
        var actions: [String] = []
        for command in commands {
            if !actions.isEmpty, command.count == 2, command.first?.value == "sleep",
               let duration = command.last?.value.flatMap(Double.init), duration.isFinite, duration >= 0 {
                continue
            }

            let pipeline = command.split(separator: .pipe, omittingEmptySubsequences: false)
            guard var browser = pipeline.first, pipeline.dropFirst().allSatisfy(isOutputFilter) else { return nil }
            if browser.last?.value == "/dev/null", browser.dropLast().last == .redirect {
                browser = browser.dropLast(2)
            }
            let words = browser.compactMap(\.value)
            guard words.count == browser.count, let action = browserVerb(words) else { return nil }
            actions.append(action)
        }

        guard let first = actions.first, let title = computerTitle("browser_\(first)") else { return nil }
        if actions.count == 1 { return title }
        switch actions {
        case ["click", "snapshot"]: return "Click and inspect the webpage"
        case ["click", "read"]: return "Click and read the webpage"
        case ["open", "snapshot"]: return "Open and inspect a webpage"
        case ["open", "read"]: return "Open and read a webpage"
        case ["click", "open", "read"]: return "Click, open, and read a webpage"
        default: return title + " and continue in the browser"
        }
    }

    /// Accept filters that only read their input stream. Other shell commands keep the generic label.
    private static func isOutputFilter(_ tokens: ArraySlice<ShellToken>) -> Bool {
        let words = tokens.compactMap(\.value)
        guard words.count == tokens.count, let executable = words.first else { return false }
        let arguments = words.dropFirst()
        switch executable {
        case "head", "tail":
            if arguments.isEmpty { return true }
            if arguments.count == 1, let option = arguments.first, option.hasPrefix("-"),
               let count = Int(option.dropFirst()) { return count >= 0 }
            return arguments.count == 2 && ["-n", "-c"].contains(words[1]) && Int(words[2]).map { $0 >= 0 } == true
        case "grep":
            guard let pattern = arguments.last, !pattern.hasPrefix("-") else { return false }
            var options = arguments.dropLast()
            while let option = options.popFirst() {
                if option.hasPrefix("-"), option.count > 1, option.dropFirst().allSatisfy({ "EFcinovwx".contains($0) }) {
                    continue
                }
                guard ["-A", "-B", "-C"].contains(String(option.prefix(2))) else { return false }
                let count = option.count == 2 ? options.popFirst() : String(option.dropFirst(2))
                guard let count, let lines = Int(count), lines >= 0 else { return false }
            }
            return true
        case "sed":
            guard arguments.count == 2, words[1] == "-n" else { return false }
            return words[2].range(of: #"\A(?:[0-9]+|/[^/\r\n]+/)(?:,(?:[0-9]+|\$|\+[0-9]+))?p\z"#,
                                  options: .regularExpression) != nil
        default: return false
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
        case sequence
        case pipe
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
            case ";":
                tokens.append(.sequence)
                input = input.dropFirst()
            case "|" where !input.hasPrefix("||"):
                tokens.append(.pipe)
                input = input.dropFirst()
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
