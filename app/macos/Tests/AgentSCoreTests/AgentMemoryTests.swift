import Foundation
import Testing
@testable import AgentSCore

@Suite("Agent memory")
struct AgentMemoryTests {
    private let local = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                         ollamaInference: .local, model: "synthetic")
    private let hosted = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                          ollamaInference: .hosted, model: "synthetic")
    private let unknown = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                           model: "synthetic")
    private let router = AgentMemoryRoute(privacy: .localVM, harness: .deepSeek, provider: .openRouter,
                                          model: "synthetic")

    @Test func manualNotesDefaultToPrivateAndSharingIsExplicit() throws {
        var memory = AgentMemory()
        #expect(try memory.saveUserNote(kind: .preference, text: "Preferred display setting"))
        let id = try #require(memory.notes.first?.id)
        #expect(memory.notes.first?.scope == .private)
        #expect(memory.context(for: local) == nil)

        #expect(try memory.setScope(.route(local), id: id))
        #expect(try memory.saveUserNote(id: id, kind: .preference, text: "Updated display setting"))
        #expect(memory.notes(for: local).first?.scope == .route(local))
        #expect(memory.notes(for: local).first?.id == id)
        #expect(memory.context(for: local)?.contains("Updated display setting") == true)
        #expect(memory.context(for: router) == nil)

        #expect(try memory.setScope(.private, id: id))
        #expect(memory.context(for: local) == nil)
        #expect(memory.notes.first?.text == "Updated display setting")
    }

    @Test func isolatesLocalHostedAndUnconfirmedOllama() throws {
        var memory = AgentMemory()
        try memory.apply([.add(kind: .decision, text: "Local note")], expecting: memory.revision, route: local)
        try memory.apply([.add(kind: .decision, text: "Hosted note")], expecting: memory.revision, route: hosted)
        try memory.apply([.add(kind: .decision, text: "Unconfirmed note")], expecting: memory.revision, route: unknown)
        #expect(local.allowsAutomaticRecall)
        #expect(hosted.allowsAutomaticRecall)
        #expect(!unknown.allowsAutomaticRecall)
        #expect(memory.context(for: local)?.contains("Local note") == true)
        #expect(memory.context(for: local)?.contains("Hosted note") == false)
        #expect(memory.context(for: hosted)?.contains("Hosted note") == true)
        #expect(memory.context(for: hosted)?.contains("Local note") == false)
        #expect(memory.context(for: unknown) == nil)
        #expect(memory.notes(for: unknown).map(\.text) == ["Unconfirmed note"])
        #expect(memory.context(for: router) == nil)

        let otherModel = AgentMemoryRoute(privacy: .fullyPrivate, harness: .deepSeek, provider: .ollama,
                                           ollamaInference: .local, model: "other")
        #expect(memory.context(for: otherModel) == nil)
    }

    @Test func separatesCloudComputerTargetsAndProviders() throws {
        var memory = AgentMemory()
        let firstBox = AgentMemoryRoute(privacy: .remote, harness: .codex, provider: .codexAccount, computer: .box, boxName: "first")
        let secondBox = AgentMemoryRoute(privacy: .remote, harness: .codex, provider: .codexAccount, computer: .box, boxName: "second")
        let boxRouter = AgentMemoryRoute(privacy: .remote, harness: .codex, provider: .openRouter, computer: .box, boxName: "first")
        let firstSai = AgentMemoryRoute(privacy: .remote, harness: .sai, provider: nil, computer: .sai, saiMachine: "first")
        let secondSai = AgentMemoryRoute(privacy: .remote, harness: .sai, provider: nil, computer: .sai, saiMachine: "second")

        try memory.apply([.add(kind: .workflow, text: "First Box workflow")], expecting: memory.revision, route: firstBox)
        try memory.apply([.add(kind: .workflow, text: "First Sai workflow")], expecting: memory.revision, route: firstSai)
        #expect(memory.notes(for: firstBox).map(\.text) == ["First Box workflow"])
        #expect(memory.notes(for: firstSai).map(\.text) == ["First Sai workflow"])
        #expect(memory.notes(for: secondBox).isEmpty)
        #expect(memory.notes(for: boxRouter).isEmpty)
        #expect(memory.notes(for: secondSai).isEmpty)
    }

    @Test func updatesInPlaceAndRetainsResolvedHistory() throws {
        var memory = AgentMemory()
        let started = Date(timeIntervalSince1970: 10)
        let updated = Date(timeIntervalSince1970: 30)
        try memory.apply([.add(kind: .activeWork, text: "  Draft the plan  ")], expecting: memory.revision, route: local,
                         sourceChatItemID: "msg-started", now: started)
        let id = try #require(memory.notes.first?.id)
        let didPin = memory.setPinned(true, id: id)
        #expect(didPin)
        try memory.apply([.update(id: id, kind: nil, text: "Review the draft plan")], expecting: memory.revision, route: local,
                         sourceChatItemID: "msg-updated", now: updated)
        #expect(memory.notes.first?.id == id)
        #expect(memory.notes.first?.isPinned == true)
        #expect(memory.notes.first?.sourceChatItemID == "msg-updated")
        #expect(memory.notes.first?.updatedAt == updated)

        try memory.apply([.supersede(id: id, kind: .decision, text: "Use the approved plan")],
                         expecting: memory.revision, route: local, sourceChatItemID: "msg-decision")
        #expect(memory.notes.first?.status == .resolved)
        #expect(memory.notes.last?.status == .open)
        #expect(memory.notes.last?.id != id)
        #expect(memory.context(for: local)?.contains("Review the draft plan") == false)

        let replacement = try #require(memory.notes.last?.id)
        try memory.apply([.resolve(id: replacement)], expecting: memory.revision, route: local)
        #expect(memory.context(for: local) == nil)
        let didReopen = memory.setStatus(.open, id: replacement)
        #expect(didReopen)
        #expect(memory.context(for: local)?.contains("Use the approved plan") == true)
    }

    @Test func rejectsStaleAndCrossRouteProposalsWithoutPartialMutation() throws {
        var memory = AgentMemory()
        try memory.apply([.add(kind: .activeWork, text: "In progress")], expecting: memory.revision, route: local)
        let id = try #require(memory.notes.first?.id)
        let original = memory
        #expect(throws: AgentMemoryError.unavailableNote) {
            try memory.apply([.add(kind: .decision, text: "Uncommitted note"), .resolve(id: id)],
                             expecting: original.revision, route: hosted)
        }
        #expect(memory == original)

        #expect(try memory.saveUserNote(id: id, kind: .activeWork, text: "My edit"))
        #expect(throws: AgentMemoryError.staleRevision) {
            try memory.apply([.update(id: id, kind: nil, text: "Old proposal")], expecting: original.revision, route: local)
        }
        let editedRevision = memory.revision
        let didForget = memory.forget(id: id)
        #expect(didForget)
        #expect(throws: AgentMemoryError.staleRevision) {
            try memory.apply([.add(kind: .activeWork, text: "My edit")], expecting: editedRevision, route: local)
        }
        #expect(memory.notes.isEmpty)
    }

    @Test func forgetsSelectedNotesAndRejectsProposalsFromTheEarlierRevision() throws {
        var memory = AgentMemory()
        try memory.apply([.add(kind: .activeWork, text: "Local work")], expecting: memory.revision, route: local)
        try memory.apply([.add(kind: .activeWork, text: "Hosted work")], expecting: memory.revision, route: hosted)
        try memory.saveUserNote(kind: .activeWork, text: "Keep this private")
        let selection = Set(memory.notes.prefix(2).map(\.id))
        let sentRevision = memory.revision

        let didForget = memory.forget(ids: selection)
        #expect(didForget)
        #expect(memory.revision != sentRevision)
        #expect(memory.notes.map(\.text) == ["Keep this private"])
        #expect(throws: AgentMemoryError.staleRevision) {
            try memory.apply([.add(kind: .activeWork, text: "Local work")], expecting: sentRevision, route: local)
        }

        let afterDeletion = memory.revision
        let removedAgain = memory.forget(ids: selection)
        #expect(!removedAgain)
        #expect(memory.revision == afterDeletion)
    }

    @Test func boundsEncodedContextAndKeepsPinnedReferences() throws {
        var memory = AgentMemory()
        let timestamp = Date(timeIntervalSince1970: 30)
        for index in 0..<8 {
            try memory.saveUserNote(kind: .workflow, text: "\(index)" + String(repeating: "\"", count: 950),
                                    route: local, now: timestamp.addingTimeInterval(Double(index)))
        }
        let pinned = try #require(memory.notes.first?.id)
        let didPin = memory.setPinned(true, id: pinned, now: timestamp)
        #expect(didPin)
        let context = try #require(memory.context(for: local))
        let decoded = try JSONDecoder().decode([[String: String]].self, from: Data(context.utf8))
        #expect(context.utf8.count <= AgentMemory.maxContextBytes)
        #expect(decoded.count < memory.notes.count)
        #expect(decoded.first?["id"] == pinned.uuidString)
        #expect(decoded.first?["kind"] == AgentMemoryKind.workflow.rawValue)
        #expect(decoded.first?["text"] == memory.notes.first?.text)
    }

    @Test func ranksRelevantNotesAndGivesResumesDistinctRouteKeys() throws {
        var memory = AgentMemory()
        let time = Date(timeIntervalSince1970: 30)
        try memory.saveUserNote(kind: .workflow, text: "Deploy the calendar update", route: local, now: time)
        try memory.saveUserNote(kind: .decision, text: "Use a revised icon", route: local, now: time.addingTimeInterval(10))
        let context = try #require(memory.context(for: local, matching: "How do I deploy the calendar?"))
        let decoded = try JSONDecoder().decode([[String: String]].self, from: Data(context.utf8))
        #expect(decoded.first?["text"] == "Deploy the calendar update")
        #expect(Set([local, hosted, unknown, router].map(\.continuationKey)).count == 4)
        let encoded = try JSONEncoder().encode(local)
        #expect(try JSONDecoder().decode(AgentMemoryRoute.self, from: encoded).continuationKey == local.continuationKey)
    }

    @Test func validatesUpdatesAndRoundTripsVersionedNotes() throws {
        var memory = AgentMemory()
        let revision = memory.revision
        #expect(throws: AgentMemoryError.invalidNote) {
            try memory.apply([.add(kind: .decision, text: " ")], expecting: revision, route: local)
        }
        #expect(throws: AgentMemoryError.invalidNote) {
            try memory.saveUserNote(kind: .preference, text: String(repeating: "🍎", count: AgentMemory.maxNoteBytes))
        }
        #expect(throws: AgentMemoryError.tooManyOperations) {
            try memory.apply(Array(repeating: .add(kind: .decision, text: "duplicate"), count: AgentMemory.maxOperationCount + 1),
                             expecting: revision, route: local)
        }
        #expect(memory.revision == revision)

        try memory.apply([.add(kind: .decision, text: "Persistent note")], expecting: revision, route: local,
                         sourceChatItemID: "source-item", now: Date(timeIntervalSince1970: 10))
        let data = try JSONEncoder().encode(memory)
        #expect(try JSONDecoder().decode(AgentMemory.self, from: data) == memory)
        #expect(memory.version == AgentMemory.currentVersion)
        var raw = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        raw["version"] = AgentMemory.currentVersion + 1
        let newerVersion = try JSONSerialization.data(withJSONObject: raw)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(AgentMemory.self, from: newerVersion)
        }
    }
}
