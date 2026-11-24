import SwiftUI

struct DownloadProgressView: View {
    let label: String
    let progress: Double?

    var body: some View {
        ProgressView(value: progress) {
            HStack {
                Text(label).lineLimit(2)
                Spacer(minLength: 8)
                if let progress {
                    Text(progress.formatted(.percent.precision(.fractionLength(0)).rounded(rule: .down)))
                        .monospacedDigit()
                        .accessibilityHidden(true)
                }
            }
            .font(.callout)
            .foregroundStyle(.secondary)
        }
        .progressViewStyle(.linear)
        .controlSize(.small)
        .tint(Theme.accent)
    }
}
