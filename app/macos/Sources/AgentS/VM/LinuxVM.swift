import Foundation
import Virtualization

/// The guest: Debian ARM64 on Apple's Virtualization framework.
///
/// EFI boot from the cloud image's disk, the cloud-init seed as a second read-only disk, NAT
/// networking with a fixed MAC (so its DHCP lease can be found), a virtio GPU whose one display is
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
    private var runningConfiguration: VZVirtualMachineConfiguration?
    private var savingCheckpoint = false

    private var checkpoint: GuestCheckpoint { GuestCheckpoint(files: files) }
    var hasSavedState: Bool { checkpoint.exists }

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
    func start(memory: MemoryAllocation, restoreSavedState: Bool = true) async -> Bool {
        guard machine == nil, state != .starting, state != .stopping, !savingCheckpoint else { return false }
        state = .starting
        var restoring = false
        do {
            let owner = try GuestDiskLock(url: files.diskLock)
            diskOwner = owner
            defer { withExtendedLifetime(owner) {} }

            restoring = restoreSavedState && checkpoint.exists
            let saved = restoring ? try checkpoint.metadata() : nil
            let allocation: MemoryAllocation
            if let saved {
                guard let restored = MemoryAllocation(rawValue: saved.memoryGB) else {
                    throw GuestCheckpoint.Failure.incompatible
                }
                allocation = restored
            } else {
                allocation = memory
            }
            let config = try configuration(memory: allocation)
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
            guard machine === vm, vm.state == .running else { return false }
            state = .running
            return restoring
        } catch {
            detachMachine()
            state = .failed(restoring ? "Couldn't resume the sandbox. \(error.localizedDescription)" : error.localizedDescription)
            return false
        }
    }

    /// Save runtime state for app termination. On success the guest stays paused until the app exits.
    func saveForQuit() async throws {
        guard let vm = machine else { return }
        guard !savingCheckpoint, state == .running, vm.state == .running,
              let configuration = runningConfiguration, let owner = diskOwner,
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
            try await checkpoint.save(memoryGB: memory.rawValue) { url in
                try await vm.saveMachineStateTo(url: url)
            }
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

    /// Asks the guest to shut down; forces it off if it is still up after `timeout` seconds.
    func stop(timeout: TimeInterval = 20) async {
        guard let vm = machine, !savingCheckpoint, state != .stopping else { return }
        state = .stopping
        if vm.canRequestStop { try? vm.requestStop() }
        let deadline = Date().addingTimeInterval(timeout)
        while vm.state != .stopped, Date() < deadline {
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        if vm.state != .stopped, vm.canStop { try? await vm.stop() }
        if machine === vm {
            detachMachine()
            state = .stopped
        }
    }

    private func detachMachine() {
        machine = nil
        runningConfiguration = nil
        diskOwner = nil
    }

    // MARK: VZVirtualMachineDelegate

    nonisolated func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        Task { @MainActor in
            guard self.machine === virtualMachine else { return }
            self.detachMachine()
            self.state = .stopped
        }
    }

    nonisolated func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) {
        Task { @MainActor in
            guard self.machine === virtualMachine else { return }
            self.detachMachine()
            self.state = .failed(error.localizedDescription)
        }
    }

    // MARK: Configuration

    private func configuration(memory: MemoryAllocation) throws -> VZVirtualMachineConfiguration {
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
        network.attachment = VZNATNetworkDeviceAttachment()
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
