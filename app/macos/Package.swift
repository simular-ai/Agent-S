// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AgentS",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "AgentS", targets: ["AgentS"]),
    ],
    dependencies: [
        .package(url: "https://github.com/no-problem-dev/swift-markdown-view.git", from: "8.0.0"),
        .package(url: "https://github.com/swiftlang/swift-markdown.git", from: "0.9.0"),
    ],
    targets: [
        // Pure logic: parsing Claude Code's stream, DHCP leases, the guest's cloud-init, the
        // commands the app runs. No AppKit, no Virtualization, so it is testable anywhere.
        .target(name: "AgentSCore"),
        // The app: the VM (Virtualization.framework), its screen, the chat, the local models.
        .executableTarget(
            name: "AgentS",
            dependencies: [
                "AgentSCore",
                .product(name: "SwiftMarkdownView", package: "swift-markdown-view"),
                .product(name: "Markdown", package: "swift-markdown"),
            ],
            // Virtualization's delegates and Process callbacks predate strict concurrency; the app
            // keeps its state on the main actor by hand.
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [.linkedFramework("Virtualization")]
        ),
        .testTarget(name: "AgentSCoreTests", dependencies: ["AgentSCore"]),
    ]
)
