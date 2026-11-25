import Foundation

public enum ModelInputCapability: Hashable, Sendable {
    case textOnly
    case multimodal

    public var label: String {
        switch self {
        case .textOnly: "Text only"
        case .multimodal: "Multimodal"
        }
    }

    public static func fromInputModalities(_ modalities: [String]?) -> Self? {
        let inputs = Set((modalities ?? []).map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }.filter { !$0.isEmpty })
        if inputs == ["text"] { return .textOnly }
        return inputs.count > 1 ? .multimodal : nil
    }
}
