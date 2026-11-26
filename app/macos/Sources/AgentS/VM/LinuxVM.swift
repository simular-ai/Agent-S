import Foundation
import Virtualization

/// The guest: Debian ARM64 on Apple's Virtualization framework.
///
/// EFI boot from the cloud image's disk, the cloud-init seed as a second read-only disk,
/// networking with a fixed MAC, a virtio GPU whose one display is
/// the screen the app shows, a keyboard and pointer for taking over, and the shared folder over
/// virtio-fs.
@MainActor
final class LinuxVM: NSObject, VZVirtualMachineDelegate {
    enum State: Equatable { case stopped, starting, running, stopping, failed(String) }

    enum MemoryAllocation: Int, CaseIterable {
        case twoGB = 2
        case fourGB = 4
        case sixGB = 6

        var bytes: UInt64 { UInt64(rawValue) * 1024 * 1024 * 1024 }
    }

    static let cpus = 4
    static let screen = (width: 1440, height: 900)

    private(set) var machine: VZVirtualMachine?
    private(set) var state: State = .stopped { didSet { onStateChange?(state) } }
    var onStateChange: ((State) -> Void)?
    let files: GuestFiles
    private var diskOwner: GuestDiskLock?
    private var hostNetwork: SandboxHostNetwork?
    private(set) var networkMode: SandboxNetworkMode?
    var forwardedSSHPort: UInt16? { hostNetwork?.sshPort }
    private var runningConfiguration: VZVirtualMachineConfiguration?
    private var startup: Task<Bool, Never>?
    private var shutdown: Task<Bool, Never>?
    private var savingCheckpoint = false

    private var checkpoint: GuestCheckpoint { GuestCheckpoint(files: files) }
    var hasSavedState: Bool { checkpoint.exists }

    var supportsCheckpoint: Bool {
        #if arch(arm64)
        guard let runningConfiguration else { return false }
        return (try? runningConfiguration.validateSaveRestoreSupport()) != nil
        #else
        return false
        #endif
    }

    init(files: GuestFiles) { self.files = files }

    /// The MAC address its network device keeps across launches.
    var macAddress: String {
        if let saved = try? String(contentsOf: files.macAddress, encoding: .utf8) {
            return saved.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let mac = VZMACAddress.randomLocallyAdministered().string
        try? mac.write(to: files.macAddress, atomically: true, encoding: .utf8)
        return mac
    }

    /// Returns true when the sandbox resumed saved state. The VM state reports the outcome of a cold boot.
    @discardableResult
    func start(memory: MemoryAllocation, networkMode: SandboxNetworkMode, restoreSavedState: Bool = true) async -> Bool {
        guard machine == nil, startup == nil, shutdown == nil, !savingCheckpoint else { return false }
        state = .starting
        let task = Task {
            defer { startup = nil }
            return await boot(memory: memory, networkMode: networkMode, restoreSavedState: restoreSavedState)
        }
        startup = task
        return await task.value
    }

    private func boot(memory: MemoryAllocation, networkMode: SandboxNetworkMode, restoreSavedState: Bool) async -> Bool {
        var restoring = false
        do {
            try Task.checkCancellation()
            let owner = try GuestDiskLock(url: files.diskLock)
            diskOwner = owner
            defer { withExtendedLifetime(owner) {} }

            restoring = restoreSavedState && checkpoint.exists
            let saved = restoring ? try checkpoint.metadata() : nil
            let allocation: MemoryAllocation
            if let saved {
                // Checkpoints from before this setting used Apple’s standard NAT.
                let savedMode = saved.networkMode ?? .standard
                guard savedMode == networkMode else { throw GuestCheckpoint.Failure.differentNetwork(savedMode) }
                guard let restored = MemoryAllocation(rawValue: saved.memoryGB) else {
                    throw GuestCheckpoint.Failure.incompatible
                }
                allocation = restored
            } else {
                allocation = memory
            }
            let address = macAddress
            let attachment: VZNetworkDeviceAttachment
            if networkMode == .followMac {
                let network = try await SandboxHostNetwork.start(macAddress: address)
                self.hostNetwork = network
                attachment = network.attachment
                network.onUnexpectedExit = { [weak self, weak network] in
                    guard let self, let network, self.hostNetwork === network else { return }
                    Task { await self.hostNetworkEnded(network) }
                }
            } else {
                attachment = VZNATNetworkDeviceAttachment()
            }
            try Task.checkCancellation()
            self.networkMode = networkMode
            let config = try configuration(memory: allocation, macAddress: address, attachment: attachment)
            try config.validate()
            let vm = VZVirtualMachine(configuration: config)
            vm.delegate = self
            machine = vm
            runningConfiguration = config
            if let saved {
                try await restore(vm, configuration: config, saved: saved)
            } else {
                try checkpoint.consume()
                checkpoint.cleanStaging()
                try await vm.start()
            }
            guard state == .starting, machine === vm, vm.state == .running else { return false }
            state = .running
            return restoring
        } catch {
            // Shutdown owns cleanup while it waits for the startup operation to finish.
            guard state != .stopping else { return false }
            if machine.map(Self.hasStopped) ?? true { detachMachine() }
            state = .failed(restoring ? "Couldn't resume the sandbox. \(error.localizedDescription)" : error.localizedDescription)
            return false
        }
    }

    /// Save runtime state for app termination. On success the guest stays paused until the app exits.
    func saveForQuit() async throws {
        guard let vm = machine else { return }
        guard !savingCheckpoint, state == .running, vm.state == .running,
              let configuration = runningConfiguration, let owner = diskOwner, let networkMode,
              let memory = MemoryAllocation.allCases.first(where: { $0.bytes == configuration.memorySize }) else {
            throw GuestCheckpoint.Failure.notRunning
        }

        #if arch(arm64)
        try configuration.validateSaveRestoreSupport()
        savingCheckpoint = true
        defer {
            savingCheckpoint = false
            withExtendedLifetime(owner) {}
        }
        do {
            try await vm.pause()
            try await checkpoint.save(memoryGB: memory.rawValue, networkMode: networkMode) { url in
                try await vm.saveMachineStateTo(url: url)
            }
            hostNetwork?.stop()
        } catch {
            if vm.state == .paused {
                do { try await vm.resume() }
                catch { state = .failed("Couldn't resume the sandbox after saving failed. \(error.localizedDescription)") }
            }
            throw error
        }
        #else
        throw GuestCheckpoint.Failure.unsupported
        #endif
    }

    private func restore(_ vm: VZVirtualMachine, configuration: VZVirtualMachineConfiguration,
                         saved: GuestCheckpoint.Metadata) async throws {
        #if arch(arm64)
        try configuration.validateSaveRestoreSupport()
        try await vm.restoreMachineStateFrom(url: checkpoint.stateURL)
        try Task.checkCancellation()
        try checkpoint.consume()
        defer { checkpoint.cleanStaging() }
        do {
            try await vm.resume()
        } catch {
            if vm.state == .paused { try? checkpoint.retainAfterFailedResume(saved) }
            throw error
        }
        #else
        throw GuestCheckpoint.Failure.unsupported
        #endif
    }

    /// Waits for startup and shuts down the guest before releasing its resources.
    @discardableResult
    func stop(timeout: TimeInterval = 20) async -> Bool {
        guard !savingCheckpoint else { return false }
        if let shutdown { return await shutdown.value }
        guard startup != nil || machine != nil else { return true }
        state = .stopping
        let task = Task {
            defer { shutdown = nil }
            if let startup {
                startup.cancel()
                _ = await startup.value
            }
            return await stopMachine(timeout: timeout)
        }
        shutdown = task
        return await task.value
    }

    private func stopMachine(timeout: TimeInterval) async -> Bool {
        if let vm = machine {
            if vm.canRequestStop, (try? vm.requestStop()) != nil {
                let deadline = ContinuousClock.now.advanced(by: .seconds(timeout))
                while !Self.hasStopped(vm), ContinuousClock.now < deadline {
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
            if vm.canStop {
                do {
                    try await vm.stop()
                } catch where !Self.hasStopped(vm) {
                    state = .failed("Couldn’t shut down the sandbox. \(error.localizedDescription)")
                    return false
                } catch {}
            }
            guard Self.hasStopped(vm) else {
                state = .failed("Couldn’t finish shutting down the sandbox. Try turning it off again.")
                return false
            }
        }
        detachMachine()
        state = .stopped
        return true
    }

    private static func hasStopped(_ vm: VZVirtualMachine) -> Bool {
        vm.state == .stopped || vm.state == .error
    }

    private func hostNetworkEnded(_ network: SandboxHostNetwork) async {
        guard hostNetwork === network, state != .stopping else { return }
        await stop()
        guard machine == nil, state == .stopped else { return }
        state = .failed("This Mac’s network connection to the sandbox stopped. Turn on the sandbox to reconnect.")
    }

    private func detachMachine() {
        machine?.delegate = nil
        machine = nil
        runningConfiguration = nil
        hostNetwork?.stop()
        hostNetwork = nil
        networkMode = nil
        diskOwner = nil
    }

    // MARK: VZVirtualMachineDelegate

    nonisolated func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        Task { @MainActor in
            guard self.machine === virtualMachine, self.state != .stopping else { return }
            self.detachMachine()
            self.state = .stopped
        }
    }

    nonisolated func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) {
        Task { @MainActor in
            guard self.machine === virtualMachine, self.state != .stopping else { return }
            self.detachMachine()
            self.state = .failed(error.localizedDescription)
        }
    }

    // MARK: Configuration

    private func configuration(memory: MemoryAllocation, macAddress: String,
                               attachment: VZNetworkDeviceAttachment) throws -> VZVirtualMachineConfiguration {
        let config = VZVirtualMachineConfiguration()
        config.platform = try platform()
        config.cpuCount = Self.cpus
        config.memorySize = memory.bytes

        let boot = VZEFIBootLoader()
        boot.variableStore = try efiVariableStore()
        config.bootLoader = boot

        // Cached + full sync: with the default (automatic) caching, Linux guests on
        // Virtualization.framework can see I/O errors and ext4 corruption (Lima hit the same).
        let disk = try VZDiskImageStorageDeviceAttachment(url: files.disk, readOnly: false,
                                                          cachingMode: .cached, synchronizationMode: .full)
        let seed = try VZDiskImageStorageDeviceAttachment(url: files.seed, readOnly: true)
        config.storageDevices = [
            VZVirtioBlockDeviceConfiguration(attachment: disk),
            VZVirtioBlockDeviceConfiguration(attachment: seed),
        ]

        let network = VZVirtioNetworkDeviceConfiguration()
        network.attachment = attachment
        if let mac = VZMACAddress(string: macAddress) { network.macAddress = mac }
        config.networkDevices = [network]

        let graphics = VZVirtioGraphicsDeviceConfiguration()
        graphics.scanouts = [
            VZVirtioGraphicsScanoutConfiguration(widthInPixels: Self.screen.width,
                                                 heightInPixels: Self.screen.height),
        ]
        config.graphicsDevices = [graphics]
        config.keyboards = [VZUSBKeyboardConfiguration()]
        config.pointingDevices = [VZUSBScreenCoordinatePointingDeviceConfiguration()]

        let share = VZVirtioFileSystemDeviceConfiguration(tag: "sai")
        share.share = VZSingleDirectoryShare(directory: VZSharedDirectory(url: files.share, readOnly: false))
        config.directorySharingDevices = [share]

        config.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]
        config.memoryBalloonDevices = [VZVirtioTraditionalMemoryBalloonDeviceConfiguration()]
        config.serialPorts = []

        // Clipboard sharing with the Mac: the SPICE agent port, served in the guest by spice-vdagent.
        let console = VZVirtioConsoleDeviceConfiguration()
        let agent = VZVirtioConsolePortConfiguration()
        agent.name = VZSpiceAgentPortAttachment.spiceAgentPortName
        let spice = VZSpiceAgentPortAttachment()
        spice.sharesClipboard = true
        agent.attachment = spice
        console.ports[0] = agent
        config.consoleDevices = [console]
        return config
    }

    /// The same machine identity every launch, so the guest's EFI and disk see one computer.
    private func platform() throws -> VZGenericPlatformConfiguration {
        let platform = VZGenericPlatformConfiguration()
        if let data = try? Data(contentsOf: files.machineIdentifier),
           let id = VZGenericMachineIdentifier(dataRepresentation: data) {
            platform.machineIdentifier = id
        } else {
            let id = VZGenericMachineIdentifier()
            try id.dataRepresentation.write(to: files.machineIdentifier)
            platform.machineIdentifier = id
        }
        return platform
    }

    private func efiVariableStore() throws -> VZEFIVariableStore {
        if FileManager.default.fileExists(atPath: files.efiVariables.path) {
            return VZEFIVariableStore(url: files.efiVariables)
        }
        return try VZEFIVariableStore(creatingVariableStoreAt: files.efiVariables)
    }
}
