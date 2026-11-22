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

    @Test func usesASeparateProfileAndTreatsThePromptAsInput() {
        let resume = "session-18dc9fe4-1044-4115-8d56-9eef81b45712"
        let args = DeepSeekBoxProfile.arguments(prompt: "--help", resume: resume, patchPath: "/tmp/box.yml")
        #expect(args == ["--profile", "sdk-minimal", "--patch", "/tmp/box.yml", "--json", "--session-id", resume, "--", "--help"])
        let fresh = DeepSeekBoxProfile.arguments(prompt: "Hello", resume: "../../other-agent", patchPath: "/tmp/box.yml")
        #expect(!fresh.contains("--session-id"))
    }
}
