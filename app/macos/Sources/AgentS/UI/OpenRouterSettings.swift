import SwiftUI
import AgentSCore

/// OpenRouter, bring-your-own-key: the key (kept in the Keychain, never shown again or sent to the
/// computer) and the model, from a search of OpenRouter's public list or typed by id.
struct OpenRouterSettings: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Section("OpenRouter") {
            if !model.hasOpenRouterKey { KeySetup.openRouter(model) }
            OpenRouterZDRSetting(isOn: $model.openRouterRequiresZDR)
            OpenRouterModelSearch(selection: $model.openRouterModel)
        }
    }
}

struct OpenRouterModelSearch: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selection: String
    @State private var search = ""

    var body: some View {
        TextField("Search models", text: $search, prompt: Text("Name or provider"))
            .task { await model.refreshOpenRouterModels() }
        if !terms.isEmpty {
            let results = matches
            ForEach(results) { found in
                Button { selection = found.id; search = "" } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(found.name)
                            if found.name != found.id {
                                Text(found.id).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .lineLimit(1)
                        Spacer()
                        if let context = found.contextLength {
                            Text("\(context / 1000)K context").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            if results.isEmpty, model.isRefreshingOpenRouterModels {
                ProgressView("Loading models…").controlSize(.small)
            } else if results.isEmpty, model.openRouterModels.isEmpty {
                Button("Reload model list") { Task { await model.refreshOpenRouterModels() } }
            } else if results.isEmpty {
                Text("No matching models").font(.caption).foregroundStyle(.secondary)
            }
        }
        TextField("Model ID", text: $selection, prompt: Text(OpenRouter.suggested[0]))
    }

    private var terms: [Substring] { search.split(whereSeparator: \.isWhitespace) }

    private var matches: [OpenRouter.Model] {
        let terms = terms
        let catalogue = model.openRouterModels.isEmpty
            ? OpenRouter.suggested.map { OpenRouter.Model(id: $0, name: $0, contextLength: nil) }
            : model.openRouterModels
        return Array(catalogue.lazy.filter { found in
            terms.allSatisfy { found.name.localizedCaseInsensitiveContains($0) || found.id.localizedCaseInsensitiveContains($0) }
        }.prefix(8))
    }
}

struct OpenRouterZDRSetting: View {
    @Binding var isOn: Bool
    var appliesOnCreate = false

    var body: some View {
        Toggle(isOn: $isOn) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Request Zero Data Retention (ZDR)")
                Text(appliesOnCreate ? "Applies to all OpenRouter agents after you create this agent."
                     : "Applies to all OpenRouter agents.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Safety Retention follows your OpenRouter privacy settings.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .toggleStyle(.switch)
    }
}
