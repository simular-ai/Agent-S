import SwiftUI
import AgentSCore

/// OpenRouter, bring-your-own-key: the key (kept in the Keychain, never shown again or sent to the
/// computer) and the model, from a search of OpenRouter's public list or typed by id.
struct OpenRouterSettings: View {
    @EnvironmentObject private var model: AppModel
    /// Keys are set in the app's Settings; an agent's settings only show one that is missing.
    var showKey = false
    @State private var key = ""
    @State private var keyStatus: String?
    @State private var search = ""

    var body: some View {
        Section("OpenRouter") {
            if showKey || !hasKey { KeySetup.openRouter(model) }
            if let keyStatus { Text(keyStatus).font(.caption).foregroundStyle(.red) }

            TextField("Model", text: $model.openRouterModel, prompt: Text("anthropic/claude-sonnet-5.5"))
            TextField("Search models", text: $search)
            if !search.isEmpty {
                ForEach(matches) { found in
                    Button { model.openRouterModel = found.id; search = "" } label: {
                        HStack {
                            Text(found.id)
                            Spacer()
                            if let context = found.contextLength {
                                Text("\(context / 1000)K").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .task { await model.refreshOpenRouterModels() }
    }

    private var matches: [OpenRouter.Model] {
        Array(model.openRouterModels.filter {
            $0.id.localizedCaseInsensitiveContains(search) || $0.name.localizedCaseInsensitiveContains(search)
        }.prefix(8))
    }

    private var hasKey: Bool { model.hasOpenRouterKey }
}
