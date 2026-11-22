import Testing
@testable import AgentSCore

@Suite("Mac UI activity")
struct MacActivityTests {
    private let macToolPath = "/Users/example/.agent-s/bin/mac"

    @Test(arguments: ["apps", "open Safari", "tree --all", "click 12", "type 7 hello", "key cmd+l", "screenshot"])
    func recognizesMacUIVerbs(_ arguments: String) {
        #expect(MacAgent.isUICommand(tool: "Bash", input: ["command": "\(macToolPath) \(arguments)"],
                                     macToolPath: macToolPath))
    }

    @Test func recognizesDirectShellSpellings() {
        #expect(MacAgent.isUICommand(tool: "Bash", input: ["command": "  mac\ttree|head"], macToolPath: macToolPath))
        #expect(MacAgent.isUICommand(tool: "Bash", input: ["command": "'\(macToolPath)' open 'System Settings'"],
                                     macToolPath: macToolPath))
        let pathWithSpaces = "/Users/Example User/.agent-s/bin/mac"
        #expect(MacAgent.isUICommand(tool: "Bash", input: ["command": "\"\(pathWithSpaces)\" screenshot"],
                                     macToolPath: pathWithSpaces))
    }

    @Test func recognizesDefaultTreeCommand() {
        #expect(MacAgent.isUICommand(tool: "Bash", input: ["command": macToolPath], macToolPath: macToolPath))
        #expect(MacAgent.isUICommand(tool: "Bash", input: ["command": "  mac\t "], macToolPath: macToolPath))
    }

    @Test func ignoresShellAndFileOperations() {
        let commands = ["echo mac open Safari", "cat \(macToolPath)", "\(macToolPath)-copy tree", "macbook tree",
                        "mac unknown", "mac\nscreenshot"]
        for command in commands {
            #expect(!MacAgent.isUICommand(tool: "Bash", input: ["command": command], macToolPath: macToolPath))
        }
        #expect(!MacAgent.isUICommand(tool: "Read", input: ["file_path": macToolPath], macToolPath: macToolPath))
    }
}
