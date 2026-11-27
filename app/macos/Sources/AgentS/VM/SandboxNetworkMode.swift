enum SandboxNetworkMode: String, Codable {
    case followMac
    case standard

    var title: String {
        switch self {
        case .followMac: "Follow this Mac’s network"
        case .standard: "Standard sandbox network"
        }
    }
}
