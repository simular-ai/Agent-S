import Foundation
import Testing
@testable import AgentSCore

@Suite("Assistant memory trailers")
struct MemoryWireTests {
    @Test func hidesTheTrailerAcrossEveryMarkerSplit() {
        let visible = "Cafe\u{301} 🍵\n"
        let payload = #"{"operations":[{"op":"add","kind":"preference","text":"Drinks tea."}]}"#
        for split in 0...markers(in: MemoryWire()).open.count {
            var wire = MemoryWire()
            let markers = markers(in: wire)
            #expect(wire.consume(.textDelta(id: "reply", text: visible + markers.open.prefix(split))) ==
                    [.textDelta(id: "reply", text: visible)])
            #expect(wire.consume(.textDelta(id: "reply", text: String(markers.open.dropFirst(split)) + payload)).isEmpty)
            #expect(wire.consume(.textDelta(id: "reply", text: markers.close)).isEmpty)
            let completion = wire.finish()
            #expect(completion.events.isEmpty)
            #expect(completion.operations == [.add(kind: .preference, text: "Drinks tea.")])
            #expect(wire.finish().operations == nil)
        }
    }

    @Test func reconcilesMultipleBlocksAndAuthoritativeSnapshots() {
        var wire = MemoryWire()
        let markers = markers(in: wire)
        let start = String(markers.open.prefix(8))
        let end = String(markers.open.dropFirst(8))
        #expect(wire.consume(.textDelta(id: "first", text: "Draft" + start)) == [.textDelta(id: "first", text: "Draft")])
        #expect(wire.consume(.textDelta(id: "second", text: end + #"{"operations":[]}"# + markers.close)).isEmpty)
        #expect(wire.consume(.textSnapshot(id: "first", text: "Draft" + start)).isEmpty)
        #expect(wire.consume(.textSnapshot(id: "second", text: end + #"{"operations":[]}"# + markers.close)).isEmpty)
        #expect(wire.consume(.textSnapshot(id: "first", text: "")) == [.textSnapshot(id: "first", text: "")])
        #expect(wire.consume(.textSnapshot(id: "second", text: "")).isEmpty)
        #expect(wire.consume(.textSnapshot(id: "canonical", text: "Final 🍵")) == [.textDelta(id: "canonical", text: "Final 🍵")])
        #expect(wire.finish().operations == nil)
    }

    @Test func preservesNaturalTextAndIgnoresTrailersBeforeTools() {
        var wire = MemoryWire()
        let markers = markers(in: wire)
        let tool = ClaudeEvent.toolCall(id: "call", name: "Read", summary: "Read a file")
        #expect(wire.consume(.textDelta(id: "prior", text: "Checking…" + markers.open + #"{"operations":[]}"# + markers.close)) ==
                [.textDelta(id: "prior", text: "Checking…")])
        #expect(wire.consume(tool) == [tool])
        #expect(wire.consume(.textDelta(id: "final", text: "0 < x <")) == [.textDelta(id: "final", text: "0 < x ")])
        let completion = wire.finish()
        #expect(completion.events == [.textDelta(id: "final", text: "<")])
        #expect(completion.operations == nil)
    }

    @Test func decodesFencedPayloadWithStrictOperationFields() {
        var wire = MemoryWire()
        let markers = markers(in: wire)
        let id = UUID()
        let payload = """
        {"operations":[
          {"op":"update","id":"\(id)","text":"New result"},
          {"op":"resolve","id":"\(id)"},
          {"op":"supersede","id":"\(id)","kind":"workflow","text":"Work from a clean branch."}
        ]}
        """
        let trailer = markers.open + "\r\n```json\r\n" + payload + "\r\n```\r\n" + markers.close
        #expect(wire.consume(.text(trailer)).isEmpty)
        #expect(wire.finish().operations == [
            .update(id: id, kind: nil, text: "New result"), .resolve(id: id),
            .supersede(id: id, kind: .workflow, text: "Work from a clean branch.")
        ])
    }

    @Test func rejectsBadNonceTruncationAndInvalidPayloadWithoutDisplayingThem() {
        let cases: [(String, String, String) -> String] = [
            { _, _, wrong in wrong + #"{"operations":[]}"# },
            { open, _, _ in String(open.prefix(18)) },
            { open, _, _ in open + #"{"operations":["# },
            { open, close, _ in open + "bad JSON" + close },
            { open, close, _ in open + #"{"operations":[{"op":"add","kind":"decision","text":"x","pinned":true}]}"# + close },
            { open, close, _ in open + #"{"operations":[]}"# + close + " extra text" },
            { open, close, _ in open + String(repeating: " ", count: 16_385) + #"{"operations":[]}"# + close },
            { open, close, _ in open + #"{"operations":[{"op":"add","kind":"decision","text":""# +
                   String(repeating: "年", count: 350) + #""}]}"# + close },
        ]
        for makeTrailer in cases {
            var wire = MemoryWire()
            let markers = markers(in: wire)
            let wrong = self.markers(in: MemoryWire()).open
            #expect(markers.open != wrong)
            let trailer = makeTrailer(markers.open, markers.close, wrong)
            #expect(wire.consume(.textDelta(id: "reply", text: "Reply.\n" + trailer)) ==
                    [.textDelta(id: "reply", text: "Reply.\n")])
            let completion = wire.finish()
            #expect(completion.events.isEmpty)
            #expect(completion.operations == nil)
        }
    }

    @Test func sanitizesAnErrorResultWithoutUsingItForMemory() {
        var wire = MemoryWire()
        let markers = markers(in: wire)
        let error = ClaudeEvent.finished(result: "Failed.\n" + markers.open + #"{"operations":[]}"# + markers.close,
                                         isError: true, costUSD: nil)
        #expect(wire.consume(error) == [.finished(result: "Failed.\n", isError: true, costUSD: nil)])
        #expect(wire.finish().operations == nil)
    }

    private func markers(in wire: MemoryWire) -> (open: String, close: String) {
        let parts = wire.instructions.split(whereSeparator: \.isNewline)
        return (String(parts[1]), String(parts[3]))
    }
}
