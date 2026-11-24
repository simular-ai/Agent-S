import SwiftUI

struct DownloadProgressView: View {
    let label: String
    let progress: Double?

    private var fraction: Double? { progress.map { min(max($0, 0), 1) } }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).lineLimit(2)
                Spacer(minLength: 8)
                if let fraction {
                    Text(fraction.formatted(.percent.precision(.fractionLength(0)).rounded(rule: .down)))
                        .monospacedDigit()
                }
            }
            .font(.callout)
            .foregroundStyle(.secondary)

            if let fraction {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Theme.hairline)
                        Capsule().fill(Theme.accent)
                            .frame(width: geometry.size.width * fraction)
                    }
                }
                .frame(height: 6)
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
                    .controlSize(.small)
                    .tint(Theme.accent)
            }
        }
        .accessibilityRepresentation {
            ProgressView(value: fraction) { Text(label) }
        }
    }
}
