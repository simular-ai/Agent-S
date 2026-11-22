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

    func start(memory: MemoryAllocation) async {
        guard machine == nil else { return }
        state = .starting
        do {
            let config = try configuration(memory: memory)
            try config.validate()
            let vm = VZVirtualMachine(configuration: config)
            vm.delegate = self
            machine = vm
            try await vm.start()
            state = .running
        } catch {
            machine = nil
            state = .failed(error.localizedDescription)
        }
    }

    /// Asks the guest to shut down; forces it off if it is still up after `timeout` seconds.
    func stop(timeout: TimeInterval = 20) async {
        guard let vm = machine else { return }
        state = .stopping
        if vm.canRequestStop { try? vm.requestStop() }
        let deadline = Date().addingTimeInterval(timeout)
        while vm.state != .stopped, Date() < deadline {
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        if vm.state != .stopped, vm.canStop { try? await vm.stop() }
        machine = nil
        state = .stopped
    }

    // MARK: VZVirtualMachineDelegate

    nonisolated func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        Task { @MainActor in
            self.machine = nil
            self.state = .stopped
        }
    }

    nonisolated func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) {
        Task { @MainActor in
            self.machine = nil
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
