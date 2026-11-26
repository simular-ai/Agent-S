import Testing
@testable import AgentSCore

@Suite("Sidebar reordering")
struct ReorderTests {
    private let letters = ["a", "b", "c", "d"]

    @Test func draggingDownLandsAfterTheTarget() {
        #expect(letters.moving({ $0 == "a" }, to: { $0 == "c" }) == ["b", "c", "a", "d"])
    }

    @Test func draggingUpLandsBeforeTheTarget() {
        #expect(letters.moving({ $0 == "d" }, to: { $0 == "b" }) == ["a", "d", "b", "c"])
    }

    @Test func reachesBothEnds() {
        #expect(letters.moving({ $0 == "b" }, to: { $0 == "d" }) == ["a", "c", "d", "b"])
        #expect(letters.moving({ $0 == "c" }, to: { $0 == "a" }) == ["c", "a", "b", "d"])
    }

    @Test func leavesOtherElementsInOrder() {
        // Agents of other sections sit between; they keep their relative order.
        let mixed = ["mac1", "vm1", "mac2", "vm2", "mac3"]
        #expect(mixed.moving({ $0 == "mac3" }, to: { $0 == "mac1" }) == ["mac3", "mac1", "vm1", "mac2", "vm2"])
    }

    @Test func doesNothingForTheSameOrAMissingElement() {
        #expect(letters.moving({ $0 == "b" }, to: { $0 == "b" }) == nil)
        #expect(letters.moving({ $0 == "x" }, to: { $0 == "b" }) == nil)
        #expect(letters.moving({ $0 == "b" }, to: { $0 == "x" }) == nil)
    }
}
