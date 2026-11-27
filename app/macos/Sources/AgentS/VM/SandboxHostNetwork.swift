import Darwin
import Foundation
import Virtualization

/// The host networking helper handles Ethernet frames from the sandbox and forwards SSH on loopback.
@MainActor
final class SandboxHostNetwork {
    let attachment: VZFileHandleNetworkDeviceAttachment
    let sshPort: UInt16
    var onUnexpectedExit: (() -> Void)? { didSet { reportExitIfNeeded() } }

    private let socket: FileHandle
    private let process: Process
    private var stopped = false
    private var reportedExit = false

    private init(socket: FileHandle, process: Process, sshPort: UInt16) {
        self.socket = socket
        self.process = process
        self.sshPort = sshPort
        attachment = VZFileHandleNetworkDeviceAttachment(fileHandle: socket)
        process.terminationHandler = { [weak self] _ in
            Task { @MainActor in self?.reportExitIfNeeded() }
        }
    }

    static func start(macAddress: String) async throws -> SandboxHostNetwork {
        let executable = Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/AgentSSandboxNetwork")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else { throw Failure.helperMissing }

        let (vmSocket, helperSocket) = try socketPair()
        let readiness = Pipe()
        defer { try? readiness.fileHandleForReading.close() }

        let process = Process()
        process.executableURL = executable
        process.arguments = [macAddress]
        process.standardInput = helperSocket
        process.standardOutput = readiness
        process.standardError = FileHandle.standardError

        do {
            try process.run()
            // The child has inherited its endpoint on descriptor 0.
            try helperSocket.close()
            let port = try await readPort(from: readiness.fileHandleForReading)
            let network = SandboxHostNetwork(socket: vmSocket, process: process, sshPort: port)
            guard process.isRunning else { throw Failure.exited }
            return network
        } catch {
            if process.isRunning { process.terminate() }
            try? helperSocket.close()
            try? vmSocket.close()
            throw error
        }
    }

    func stop() {
        guard !stopped else { return }
        stopped = true
        onUnexpectedExit = nil
        if process.isRunning { process.terminate() }
        try? socket.close()
    }

    deinit {
        if process.isRunning { process.terminate() }
    }

    private func reportExitIfNeeded() {
        guard !stopped, !reportedExit, !process.isRunning, let onUnexpectedExit else { return }
        reportedExit = true
        onUnexpectedExit()
    }

    private static func socketPair() throws -> (FileHandle, FileHandle) {
        var descriptors: [Int32] = [-1, -1]
        guard Darwin.socketpair(AF_UNIX, SOCK_DGRAM, 0, &descriptors) == 0 else { throw socketError() }
        do {
            for descriptor in descriptors {
                // Virtualization expects the receive buffer to be at least twice the send buffer.
                var sendBuffer: Int32 = 64 * 1024
                var receiveBuffer: Int32 = 256 * 1024
                let size = socklen_t(MemoryLayout<Int32>.size)
                guard fcntl(descriptor, F_SETFD, FD_CLOEXEC) == 0,
                      setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF, &sendBuffer, size) == 0,
                      setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, &receiveBuffer, size) == 0 else {
                    throw socketError()
                }
            }
        } catch {
            for descriptor in descriptors { _ = Darwin.close(descriptor) }
            throw error
        }
        return (FileHandle(fileDescriptor: descriptors[0], closeOnDealloc: true),
                FileHandle(fileDescriptor: descriptors[1], closeOnDealloc: true))
    }

    private static func readPort(from readiness: FileHandle) async throws -> UInt16 {
        let descriptor = readiness.fileDescriptor
        guard fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0 else { throw socketError() }

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(10))
        var buffer = [UInt8](repeating: 0, count: 8)
        var response: [UInt8] = []
        while clock.now < deadline {
            try Task.checkCancellation()
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            if count > 0 {
                response.append(contentsOf: buffer.prefix(count))
                if response.last == 10 {
                    let digits = response.dropLast()
                    guard digits.allSatisfy({ (48...57).contains($0) }),
                          let port = UInt16(String(decoding: digits, as: UTF8.self)), port > 0 else {
                        throw Failure.invalidPort
                    }
                    return port
                }
                guard response.count <= 5 else { throw Failure.invalidPort }
            } else if count == 0 {
                throw Failure.exited
            } else if errno != EAGAIN && errno != EINTR {
                throw socketError()
            }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw Failure.timedOut
    }

    private static func socketError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }

    private enum Failure: LocalizedError {
        case helperMissing, invalidPort, exited, timedOut

        var errorDescription: String? {
            switch self {
            case .helperMissing: "The sandbox networking helper is missing. Rebuild or reinstall Agent S."
            case .invalidPort: "The sandbox networking helper reported an invalid SSH port."
            case .exited: "The sandbox networking helper stopped before the sandbox connected."
            case .timedOut: "The sandbox networking helper took too long to start."
            }
        }
    }
}
