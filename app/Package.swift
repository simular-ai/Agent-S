// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AgentS",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "AgentS", targets: ["AgentS"]),
    ],
    targets: [
        // Pure logic: parsing Claude Code's stream, DHCP leases, the guest's cloud-init, the
        // commands the app runs. No AppKit, no Virtualization, so it is testable anywhere.
        .target(name: "AgentSCore"),
        // The app: the VM (Virtualization.framework), its screen, the chat, the local models.
        .executableTarget(
            name: "AgentS",
            dependencies: ["AgentSCore"],
            // Virtualization's delegates and Process callbacks predate strict concurrency; the app
            // keeps its state on the main actor by hand.
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [.linkedFramework("Virtualization")]
        ),
        .testTarget(name: "AgentSCoreTests", dependencies: ["AgentSCore"]),
    ]
)
