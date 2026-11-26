import Testing
import AgentSCore

@Suite("DeepSeek Harness on Box")
struct DeepSeekBoxTests {
    @Test func onlyTheBoxBridgeProvidesTools() {
        for tool in ["tool-fs", "tool-working-directory", "persistent-bash", "persistent-pwsh", "mcp-resources"] {
            #expect(DeepSeekBoxProfile.patch.contains("- id: \(tool)\n  disabled: true"))
        }
        #expect(DeepSeekBoxProfile.patch.contains("serverName: box"))
        #expect(DeepSeekBoxProfile.patch.contains("transport: streamable-http"))
        #expect(DeepSeekBoxProfile.patch.contains("Authorization: !!js process.env.AGENT_S_BOX_AUTHORIZATION"))
    }

    @Test func servesTheSDKProtocolInsteadOfAHeadlessTurn() {
        for patch in [DeepSeekBoxProfile.patch, DeepSeekMacProfile.patch] {
            #expect(!patch.contains("sdk-jsonrpc-server"))
            #expect(!patch.contains("sdk-app-startup"))
            #expect(!patch.contains("headless"))
            #expect(patch.contains("providers:\n          \(DeepSeekSDKProfile.provider):"))
        }
    }
}

@Suite("DeepSeek SDK runtime")
struct DeepSeekSDKProfileTests {
    @Test func launchesTheMinimalProfileWithoutPromptArguments() {
        #expect(DeepSeekSDKProfile.arguments(entry: "/app/dsh/bin.js", patchPath: "/tmp/mac.yml") ==
                ["/app/dsh/bin.js", "--profile", "sdk-minimal", "--patch", "/tmp/mac.yml"])
        #expect(DeepSeekSDKProfile.arguments(entry: nil, patchPath: "/tmp/mac.yml") ==
                ["--profile", "sdk-minimal", "--patch", "/tmp/mac.yml"])
    }

    @Test func acceptsOnlySessionIdentities() {
        let id = DeepSeekSDKProfile.newSessionID()
        #expect(DeepSeekSDKProfile.isSessionID(id))
        #expect(DeepSeekSDKProfile.isSessionID("session-18dc9fe4-1044-4115-8d56-9eef81b45712"))
        #expect(!DeepSeekSDKProfile.isSessionID("../../other-agent"))
        #expect(!DeepSeekSDKProfile.isSessionID("session-../../x"))
    }
}

@Suite("DeepSeek Harness on This Mac")
struct DeepSeekMacTests {
    @Test func onlyTheApprovalBridgeProvidesMacTools() {
        let patch = DeepSeekMacProfile.patch
        for plugin in ["tool-fs", "tool-working-directory", "persistent-bash", "persistent-pwsh",
                       "terminal-bash", "terminal-pwsh", "pty", "subprocess", "mcp-resources",
                       "llm-deepseek", "session-log-deepseek"] {
            #expect(patch.contains("- id: \(plugin)\n  disabled: true"))
        }
        #expect(patch.contains("serverName: agent_s"))
        #expect(patch.contains("transport: streamable-http"))
        #expect(patch.contains("url: !!js process.env.AGENT_S_MAC_URL"))
        #expect(patch.contains("Authorization: !!js process.env.AGENT_S_MAC_AUTHORIZATION"))
        #expect(patch.contains("\n        toolCallTimeoutMs: 1800000\n"))
        #expect(patch.contains("failOnStartupError: true"))
        #expect(!patch.contains("AGENT_S_BOX_"))
    }
}
