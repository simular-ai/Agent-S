import Testing
@testable import AgentSCore

@Suite("Claude Code account agents")
struct ClaudeAccountTests {
    @Test func macUsesNativeCredentialsAndOnlyTheApprovalBridge() {
        let command = MacClaudeCommand(prompt: "--help", resume: "session", model: "gemma4", provider: .claudeAccount,
                                       apiKey: "local-proxy-token", baseURL: "http://127.0.0.1:11434", macToolPath: "/mac")
        let env = command.environment(home: "/Users/example", configDir: "/account", path: "/bin",
                                      approvalPort: 9999, approvalToken: "approval-token")
        #expect(env["CLAUDE_CONFIG_DIR"] == "/account")
        #expect(env["AGENT_S_APPROVAL_TOKEN"] == "approval-token")
        #expect(env["AGENT_S_APPROVAL_PORT"] == "9999")
        #expect(env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false")
        #expect(env["ENABLE_TOOL_SEARCH"] == "false")
        #expect(!env.keys.contains { $0.hasPrefix("ANTHROPIC_") || $0.hasPrefix("CLAUDE_CODE_MAX_") })

        let args = command.arguments(settingsPath: "/claude-settings.json", mcpConfigPath: "/mac-mcp.json")
        #expect(args.contains("--strict-mcp-config"))
        #expect(args.contains("--no-chrome"))
        #expect(args.drop(while: { $0 != "--tools" }).dropFirst().first == "")
        #expect(args.drop(while: { $0 != "--allowedTools" }).dropFirst().first == MacAgent.mcpCommandTool)
        #expect(args.drop(while: { $0 != "--mcp-config" }).dropFirst().first == "/mac-mcp.json")
        #expect(args.contains("dontAsk"))
        #expect(!args.contains("bypassPermissions"))
        #expect(!args.contains("/claude-settings.json"))
        #expect(args.suffix(4) == ["--resume", "session", "--", "--help"])
    }

    @Test func macProxyStillUsesItsEndpointAndApprovalHook() {
        let command = MacClaudeCommand(prompt: "Hi", resume: nil, model: "gemma4", provider: .ollama,
                                       apiKey: "ollama", baseURL: "http://127.0.0.1:11434", macToolPath: "/mac")
        let env = command.environment(home: "/Users/example", configDir: "/agent", path: "/bin",
                                      approvalPort: 9999, approvalToken: "approval-token")
        #expect(env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434")
        #expect(env["ANTHROPIC_MODEL"] == "gemma4")
        let args = command.arguments(settingsPath: "/claude-settings.json")
        #expect(args.drop(while: { $0 != "--tools" }).dropFirst().first == MacAgent.tools.joined(separator: ","))
        #expect(args.contains("/claude-settings.json"))
        #expect(!args.contains("--mcp-config"))
    }

    @Test func boxUsesNativeCredentialsAndOnlyBoxTools() {
        let command = BoxClaudeCommand(prompt: "Hi", resume: nil, model: "gemma4", provider: .claudeAccount,
                                       apiKey: "local-proxy-token", boxName: nil, baseURL: "http://127.0.0.1:11434")
        let env = command.environment(home: "/Users/example", configDir: "/account", boxToken: "box-key")
        #expect(env["CLAUDE_CONFIG_DIR"] == "/account")
        #expect(env["BOX_TOKEN"] == "box-key")
        #expect(env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false")
        #expect(env["ENABLE_TOOL_SEARCH"] == "false")
        #expect(!env.keys.contains { $0.hasPrefix("ANTHROPIC_") || $0.hasPrefix("CLAUDE_CODE_MAX_") })

        let args = command.arguments(mcpConfigPath: "/box-mcp.json")
        #expect(args.contains("--strict-mcp-config"))
        #expect(args.contains("--no-chrome"))
        #expect(args.contains("/box-mcp.json"))
        #expect(args.contains(Box.blockedTools.joined(separator: ",")))
        #expect(args.drop(while: { $0 != "--permission-mode" }).dropFirst().first == "dontAsk")
        #expect(args.drop(while: { $0 != "--allowedTools" }).dropFirst().first == "mcp__box__*")
        #expect(args.drop(while: { $0 != "--tools" }).dropFirst().first == "")
        #expect(args.drop(while: { $0 != "--setting-sources" }).dropFirst().first == "")
        #expect(Box.onlyBoxAccountTools(["mcp__box__shell", "EndConversation"]))
        #expect(!Box.onlyBoxAccountTools(["mcp__box__box_delete"]))
        #expect(!Box.onlyBoxAccountTools(["mcp__foreign__shell"]))
    }
}
