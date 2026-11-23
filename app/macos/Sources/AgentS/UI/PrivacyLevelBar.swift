import SwiftUI
import AgentSCore

struct PrivacyLevelBar: View {
    let level: AgentPrivacyLevel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Privacy level").font(.callout.weight(.semibold))
                Spacer()
                Text("\(level.score)/\(AgentPrivacyLevel.maximum)")
                    .font(.callout.weight(.medium)).monospacedDigit()
            }
            ProgressView(value: Double(level.score), total: Double(AgentPrivacyLevel.maximum))
                .progressViewStyle(.linear)
                .tint(Theme.accent)
            Text(level.summary).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Privacy level")
        .accessibilityValue("\(level.score) out of \(AgentPrivacyLevel.maximum). \(level.summary)")
        .help("Based on where tools and inference run and the retention setting Simular requests.")
    }
}
