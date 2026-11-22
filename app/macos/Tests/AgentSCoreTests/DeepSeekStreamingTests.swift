import Testing
@testable import AgentSCore

@Suite("DeepSeek live text")
struct DeepSeekStreamingTests {
    @Test func reconcilesPartialTextWithTheCommittedReply() {
        var stream = DeepSeekStream()
        #expect(stream.consume(line: #"{"type":"agent_s_text_stream","id":"session:1"}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"agent_s_text_delta","id":"session:1","text":"Hello"}"#) ==
                [.textDelta(id: "session:1", text: "Hello")])
        #expect(stream.consume(line: #"{"type":"agent_s_text_delta","id":"session:1","text":"\n\n"}"#) ==
                [.textDelta(id: "session:1", text: "\n\n")])
        #expect(stream.consume(line: #"{"type":"thinking","text":"Hidden reasoning."}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"text","text":"Hello\n\n**world**"}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"agent_s_text_snapshot","id":"session:1","text":"Hello\n\n**world**"}"#) ==
                [.textSnapshot(id: "session:1", text: "Hello\n\n**world**")])
        #expect(stream.consume(line: #"{"type":"text","text":"A message outside any attempt."}"#) ==
                [.text("A message outside any attempt.")])
    }

    @Test func removesFailedAttemptsAndKeepsCanonicalOnlyText() {
        var stream = DeepSeekStream()
        #expect(stream.consume(line: #"{"type":"agent_s_text_stream","id":"session:1"}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"agent_s_text_delta","id":"session:1","text":"Unfinished"}"#) ==
                [.textDelta(id: "session:1", text: "Unfinished")])
        #expect(stream.consume(line: #"{"type":"agent_s_text_snapshot","id":"session:old","text":""}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"agent_s_text_snapshot","id":"session:1","text":""}"#) ==
                [.textSnapshot(id: "session:1", text: "")])

        #expect(stream.consume(line: #"{"type":"agent_s_text_stream","id":"session:2"}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"text","text":"Complete"}"#).isEmpty)
        #expect(stream.consume(line: #"{"type":"agent_s_text_snapshot","id":"session:2","text":"Complete"}"#) ==
                [.textSnapshot(id: "session:2", text: "Complete")])
    }

    @Test func bundlesTheSamePluginForExistingGuestsAndBox() {
        let path = "./" + DeepSeekStreamingPlugin.filename
        let guest = DeepSeekCommand(prompt: "Hello", resume: nil, model: "gemma4")
        #expect(guest.patch.contains("name: " + path))
        #expect(guest.remoteShell.contains("> ~/.dsh/" + DeepSeekStreamingPlugin.filename))
        #expect(DeepSeekBoxProfile.patch.contains("name: " + path))
    }
}
