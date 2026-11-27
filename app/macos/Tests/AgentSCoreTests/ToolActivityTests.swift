import Testing
@testable import AgentSCore

@Suite("Tool activity")
struct ToolActivityTests {
    @Test func describesKnownMacCommandsUsingExactInput() {
        #expect(ToolActivity.title(tool: "Bash", summary: "Do something",
                                   detail: "/Users/example/.agent-s/bin/mac screenshot") == "Capture the Mac screen")
        #expect(ToolActivity.title(tool: "Bash", summary: #"cd /tmp && "/a folder/mac" open Safari"#) == "Open a Mac app")
        #expect(ToolActivity.title(tool: "Shell", summary: "/bin/bash -lc 'cd /tmp && /a/mac tree'") == "Inspect the front Mac window")
    }

    @Test func describesDefaultMacAndSimulangActions() {
        #expect(ToolActivity.title(tool: "Bash", summary: "/Users/example/.agent-s/bin/mac") == "Inspect the front Mac window")
        #expect(ToolActivity.title(tool: "Shell", summary: "simulang run '/tmp/private script.ts'") == "Run a computer-use script")
    }

    @Test func keepsKnownGuestActionsAndFileTitlesGeneric() {
        #expect(ToolActivity.title(tool: "Bash", summary: "web open 'https://example.com/private?a=1&b=2'") == "Open a webpage")
        #expect(ToolActivity.title(tool: "bash", summary: "/usr/local/bin/ui type 7 'private text'") == "Type in an app")
        #expect(ToolActivity.title(tool: "Read", summary: "Read /private/test.txt") == "Read a file")
        #expect(ToolActivity.title(tool: "mcp__box__edit_file", summary: "private.txt") == "Edit a file")
        #expect(ToolActivity.title(tool: "Bash", summary: "cat > '/tmp/test file.txt'") == "Write a file")
        #expect(ToolActivity.title(tool: "Bash", summary: """
        cat > /tmp/test.sh <<'SCRIPT'
        curl https://example.com
        SCRIPT
        """) == "Write a file")
    }

    @Test func labelsGuestAppLaunchesWithoutTheirArguments() {
        let launch = "/usr/local/bin/ui launch '/home/example/.local/share/applications/Sample Notes.desktop'"
        #expect(ToolActivity.title(tool: "Bash", summary: "ui launch xterm") == "Launch an app")
        #expect(ToolActivity.title(tool: "Shell", summary: "Open the sample app", detail: launch) == "Launch an app")
        #expect(ToolActivity.detail(input: ["command": launch]) == launch)
        #expect(ToolActivity.title(tool: "Shell", summary: "ui launch") == "Run a terminal command")
        #expect(ToolActivity.title(tool: "Shell", summary: "ui launch --help") == "Run a terminal command")
    }

    @Test func describesBrowserActionsThroughWaitsAndOutputFilters() {
        #expect(ToolActivity.title(tool: "bash", summary: "Inspect",
                                   detail: "web snapshot | grep -n -E 'heading|button' | head -80") == "Inspect the webpage")
        #expect(ToolActivity.title(tool: "bash", summary: "Inspect",
                                   detail: "web snapshot | grep -n -i -B4 -A8 -C 2 'button'") == "Inspect the webpage")
        #expect(ToolActivity.title(tool: "bash", summary: "Click",
                                   detail: "web click e12 && sleep 0.5 && web snapshot | grep -n 'button' | head -n 80") ==
                "Click and inspect the webpage")
        #expect(ToolActivity.title(tool: "bash", summary: "Continue",
                                   detail: "web click e12 > /dev/null; web open example.com && sleep 1 && web read | tail -60") ==
                "Click, open, and read a webpage")
        #expect(ToolActivity.title(tool: "bash", summary: "Inspect",
                                   detail: "web snapshot | sed -n '/heading/,$p' | head") == "Inspect the webpage")
        #expect(ToolActivity.title(tool: "Shell", summary: #"/bin/bash -lc 'cd /tmp && web click e5 && web read | head'"#) ==
                "Click and read the webpage")
        #expect(ToolActivity.title(tool: "bash", summary: "Open",
                                   detail: "web open 'https://example.com/?a=1&b=2' && sleep 1 && web read | tail") ==
                "Open and read a webpage")
    }

    @Test(arguments: [
        "echo '/a/mac screenshot'",
        "mac screenshot && web open https://example.com",
        "mac screenshot; rm /tmp/file",
        "mac screenshot\nweb read",
        "mkdir -p /tmp/scratch && mac screenshot",
        "web navigate example.com",
        "mac\" screenshot\"",
        #"mac type 7 "$(other-command)""#,
        "cat > /tmp/example <<EOF\n$(other-command)\nEOF",
        "cat > /tmp/example <<'EOF'\nbody\nEOF\nother-command",
        "web snapshot; rm /tmp/file",
        "web read || web open example.com",
        "web read | head -n 5 /tmp/file",
        "web read | grep 'heading' /tmp/file",
        "web read | sed -n '1e echo other-command'",
        "web read > /tmp/file",
        "echo 'web click e12' | head",
    ])
    func keepsUnclearShellCommandsGeneric(_ command: String) {
        #expect(ToolActivity.title(tool: "Bash", summary: "Capture the screen", detail: command) == "Run a terminal command")
        #expect(ToolActivity.title(tool: MacAgent.mcpCommandTool, summary: command) == "Run a Mac command")
    }

    @Test func keepsPathlessProviderTitles() {
        #expect(ToolActivity.title(tool: "browser_navigate", summary: "Open the report",
                                   detail: "https://example.com/report") == "Open the report")
        #expect(ToolActivity.title(tool: "browser_navigate", summary: "Open /private/report") == "Open a webpage")
        #expect(ToolActivity.title(tool: "computer_state", summary: "/private/report") == "Inspect the computer")
        #expect(ToolActivity.title(tool: "screenshot", summary: "Open the report") == "Capture the computer screen")
    }

    @Test func preservesOnlyKnownExactInputFields() {
        let command = "  mac screenshot\nother-command  "
        #expect(ToolActivity.detail(input: ["command": command, "description": "Capture the Mac screen"]) == command)
        #expect(ToolActivity.detail(input: ["file_path": "/tmp/example\nfile", "content": "private"]) == "/tmp/example\nfile")
        #expect(ToolActivity.detail(input: ["command": "\t", "path": "/tmp/file"]) == "/tmp/file")
        #expect(ToolActivity.detail(input: ["title": "A title", "content": "private", "prompt": "private"]) == nil)
    }

    @Test func namesCommonTerminalActionsWithoutArguments() {
        let cases: [(String, String)] = [
            ("/bin/bash -lc 'cd /Users/example/work && git status --short'", "Check Git status"),
            ("git -C '/private/repository path' --no-pager diff --cached", "Review Git changes"),
            ("gh pr checks private-branch --repo example/private-name", "Check pull request status"),
            ("rg -n 'private input' /private/work", "Search file contents"),
            ("swift test --filter PrivateTests", "Run Swift tests"),
            ("xcodebuild -scheme PrivateScheme build", "Build with Xcode"),
            ("python3 -m pytest /private/work", "Run Python tests"),
            ("npm run test -- private-input", "Run JavaScript tests"),
            ("find . -delete", "Run a find command"),
            ("curl -H 'Authorization: Bearer example-token' https://example.test", "Run curl"),
        ]
        for (command, expected) in cases {
            #expect(ToolActivity.activityTitle(tool: "Shell", summary: "Terminal", detail: command) == expected)
        }
        #expect(ToolActivity.title(tool: "Shell", summary: "git status") == "Run a terminal command")
    }

    @Test func usesOnlyConservativeFileNamesAndWebHosts() {
        #expect(ToolActivity.activityTitle(tool: "Read", summary: "Report", detail: "/private/Report 2026.md") == "Read Report 2026.md")
        #expect(ToolActivity.activityTitle(tool: "mcp__box__write_file", summary: "Ignored", detail: "/private/Package.swift") == "Write Package.swift")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "cat '/private/Report 2026.md'") == "Read Report 2026.md")
        #expect(ToolActivity.activityTitle(tool: "Read", summary: "Ignored", detail: "/private/.env") == "Read a file")
        #expect(ToolActivity.activityTitle(tool: "Read", summary: "Ignored", detail: "/private/API credentials.json") == "Read a file")
        #expect(ToolActivity.activityTitle(tool: "Read", summary: "Ignored", detail: "/private/unsafe\nname.txt") == "Read a file")

        let target = "https://example-user:example-pass@example.test:8443/private?token=example-secret#account"
        #expect(ToolActivity.activityTitle(tool: "browser_navigate", summary: "Open the report", detail: target) == "Open example.test")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "web open '\(target)'") == "Open example.test")
        #expect(ToolActivity.activityTitle(tool: "browser_open", summary: "Ignored", detail: "file:///private/report") == "Open a webpage")
    }

    @Test func doesNotExposeTypedTextOrMislabelBroadShellCommands() {
        #expect(ToolActivity.activityTitle(tool: "browser_type", summary: "Type example password into the field") == "Type on the webpage")
        #expect(ToolActivity.activityTitle(tool: "unknown", summary: "Type example secret into a field") == "Use a tool")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "mac type 9 'example private text'") == "Type in a Mac app")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "PRIVATE_TOKEN=sample git status") == "Run a terminal command")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "git status && rm /tmp/example") == "Run a terminal command")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "python script.py") == "Run a terminal command")
        #expect(ToolActivity.activityTitle(tool: "Shell", summary: "python3 -c 'print(42)'") == "Run a terminal command")
    }
}
