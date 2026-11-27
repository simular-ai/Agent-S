import Darwin
import Foundation

/// The transport can prove this follow-up never reached the model.
enum SteeringDeliveryError: Error {
    case notDelivered
}

/// The input side of one headless DeepSeek turn. Delivery is confirmed by the harness.
@MainActor
final class DeepSeekSteering {
    let input = Pipe()
    private let writes = DispatchQueue(label: "agent-s.deepseek-input")
    private var pending: (id: String, reply: CheckedContinuation<Void, Error>)?
    private(set) var isClosed = false

    init() {
        // The turn can exit between the readiness event and the write.
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)
    }

    func send(_ prompt: String) async throws {
        guard !isClosed, pending == nil, prompt.utf8.count <= 65_536 else {
            throw SteeringDeliveryError.notDelivered
        }
        let id = UUID().uuidString
        var line = try JSONSerialization.data(withJSONObject: ["type": "agent_s_steer", "id": id, "text": prompt])
        line.append(0x0A)
        let packet = line
        let writer = input.fileHandleForWriting
        try await withCheckedThrowingContinuation { reply in
            pending = (id, reply)
            // SSH can stop reading before it notices a disconnection. Keep Stop responsive.
            writes.async { [weak self] in
                do { try writer.write(contentsOf: packet) }
                catch { Task { @MainActor in self?.writeFailed(id: id, error: error) } }
            }
        }
    }

    func acknowledge(id: String, accepted: Bool) {
        guard let request = pending, request.id == id else { return }
        pending = nil
        if accepted { request.reply.resume() }
        else { request.reply.resume(throwing: SteeringDeliveryError.notDelivered) }
    }

    func close() {
        guard !isClosed else { return }
        isClosed = true
        let writer = input.fileHandleForWriting
        writes.async { try? writer.close() }
        let request = pending
        pending = nil
        request?.reply.resume(throwing: CancellationError())
    }

    private func writeFailed(id: String, error: Error) {
        guard let request = pending, request.id == id else { return }
        pending = nil
        close()
        request.reply.resume(throwing: error)
    }
}
