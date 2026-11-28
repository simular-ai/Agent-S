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

    @Test func hidesShortenedTrailersAcrossEveryMarkerSplitWithoutAcceptingTheirProposal() {
        let visible = "Finished. 🍵\n"
        let payload = #"{"operations":[{"op":"add","kind":"decision","text":"Don't store this alias."}]}"#
        for split in 0...markers(in: MemoryWire(), shortened: true).open.count {
            var wire = MemoryWire()
            let markers = markers(in: wire, shortened: true)
            #expect(wire.consume(.textDelta(id: "first", text: visible + markers.open.prefix(split))) ==
                    [.textDelta(id: "first", text: visible)])
            #expect(wire.consume(.textDelta(id: "second", text: String(markers.open.dropFirst(split)) + payload)).isEmpty)
            #expect(wire.consume(.textDelta(id: "second", text: markers.close)).isEmpty)
            let completion = wire.finish()
            #expect(completion.events.isEmpty)
            #expect(completion.operations == nil)
        }
    }

    @Test func hidesShortenedTrailersDuringCharacterAtATimeStreamingAndSnapshots() {
        var wire = MemoryWire()
        let markers = markers(in: wire, shortened: true)
        let payload = #"{"operations":[]}"#
        for character in markers.open + payload + markers.close {
            #expect(wire.consume(.textDelta(id: "reply", text: String(character))).isEmpty)
        }
        #expect(wire.consume(.textSnapshot(id: "reply", text: "Done.\n" + markers.open + payload + markers.close)) ==
                [.textDelta(id: "reply", text: "Done.\n")])
        #expect(wire.finish().operations == nil)
    }

    @Test func scrubsPreviouslySavedAndIdentifiablyTruncatedShortenedTrailers() {
        let markers = markers(in: MemoryWire(), shortened: true)
        let visible = "Answer.\n"
        #expect(MemoryWire.visibleText(visible + markers.open + #"{"operations":[]}"# + markers.close) == visible)
        for truncated in ["<agent-memory", "<agent-memory:", String(markers.open.prefix(21))] {
            #expect(MemoryWire.visibleText(visible + truncated) == visible)
            var wire = MemoryWire()
            #expect(wire.consume(.textSnapshot(id: "reply", text: visible + truncated)) ==
                    [.textDelta(id: "reply", text: visible)])
            #expect(wire.finish().events.isEmpty)
        }

        let ordinaryText = "Use <agent-memory:reference> or <agent-mem when discussing syntax."
        #expect(MemoryWire.visibleText(ordinaryText) == ordinaryText)
        var wire = MemoryWire()
        #expect(wire.consume(.textSnapshot(id: "reply", text: ordinaryText)) == [.textDelta(id: "reply", text: ordinaryText)])
        #expect(wire.finish().events.isEmpty)
    }

    @Test func releasesUnrelatedProseWhenMultipleBlocksDisambiguateAShortenedMarkerCandidate() {
        var wire = MemoryWire()
        let prefix = "Details: "
        let first = "<agent-memory:"
        let second = "a brief annotation> can be ordinary text."
        #expect(wire.consume(.textDelta(id: "first", text: prefix + first)) == [.textDelta(id: "first", text: prefix)])
        #expect(wire.consume(.textDelta(id: "second", text: "a")).isEmpty)
        #expect(wire.consume(.textDelta(id: "second", text: String(second.dropFirst()))) == [
            .textDelta(id: "first", text: first), .textDelta(id: "second", text: second),
        ])
        #expect(wire.finish().events.isEmpty)
    }

    @Test func ignoresACanonicalProposalAfterAShortenedTrailer() {
        var wire = MemoryWire()
        let shortened = markers(in: wire, shortened: true)
        let canonical = markers(in: wire)
        let payload = #"{"operations":[{"op":"add","kind":"preference","text":"Drinks tea."}]}"#
        #expect(wire.consume(.text(shortened.open + shortened.close + canonical.open + payload + canonical.close)).isEmpty)
        #expect(wire.finish().operations == nil)
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

    @Test func showsTheReplyAfterSteeringPastAnEarlierMemoryTrailer() {
        var wire = MemoryWire()
        let markers = markers(in: wire)
        let initialReply = "I’m checking the available options."
        let earlierProposal = #"{"operations":[{"op":"add","kind":"decision","text":"Use the first option."}]}"#
        let priorText = initialReply + markers.open + earlierProposal + markers.close
        #expect(wire.consume(.textDelta(id: "initial", text: priorText)) ==
                [.textDelta(id: "initial", text: initialReply)])

        #expect(wire.consume(.userInputReceived).isEmpty)
        let followUpReply = "The second option meets your requirements."
        let finalProposal = #"{"operations":[{"op":"add","kind":"decision","text":"Use the second option."}]}"#
        #expect(wire.consume(.textDelta(id: "follow-up", text: followUpReply)) ==
                [.textDelta(id: "follow-up", text: followUpReply)])
        #expect(wire.consume(.textSnapshot(id: "follow-up",
                                           text: followUpReply + markers.open + finalProposal + markers.close)).isEmpty)
        #expect(wire.finish().operations == [.add(kind: .decision, text: "Use the second option.")])
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

    private func markers(in wire: MemoryWire, shortened: Bool = false) -> (open: String, close: String) {
        let parts = wire.instructions.split(whereSeparator: \.isNewline)
        if shortened {
            return (parts[1].replacingOccurrences(of: "agent-s-memory", with: "agent-memory"),
                    parts[3].replacingOccurrences(of: "agent-s-memory", with: "agent-memory"))
        }
        return (String(parts[1]), String(parts[3]))
    }
}
