import SwiftUI
import AgentSCore

/// OpenRouter, bring-your-own-key: the key (kept in the Keychain, never shown again or sent to the
/// computer) and the model, from a search of OpenRouter's public list or typed by id.
struct OpenRouterSettings: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Section(ModelProvider.openRouter.displayName) {
            if !model.hasOpenRouterKey { KeySetup.openRouter(model) }
            OpenRouterZDRSetting(isOn: $model.openRouterRequiresZDR)
            OpenRouterModelSearch(selection: $model.openRouterModel)
        }
    }
}

struct OpenRouterModelSearch: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selection: String
    var showsSelectedInput = true
    @State private var showsCatalog = false

    var body: some View {
        Button("Browse OpenRouter models…") { showsCatalog = true }
            .sheet(isPresented: $showsCatalog) {
                OpenRouterCatalogView(selection: $selection)
            }
        TextField("Model ID", text: $selection, prompt: Text(OpenRouter.suggested[0]))
            .task { await model.refreshOpenRouterModels() }
        let selectedID = selection.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selectedID.isEmpty {
            if showsSelectedInput {
                let selected = model.openRouterModels.first { $0.id == selectedID }
                LabeledContent("Model input", value: selected?.inputCapability?.label ?? "Unknown")
            }
            let availability = model.openRouterZDRAvailability(for: selectedID)
            LabeledContent("Zero Data Retention") {
                if case .unknown = availability, model.isRefreshingOpenRouterModels {
                    Text("Checking…")
                } else {
                    OpenRouterZDRIndicator(availability: availability)
                }
            }
            if case .safetyRetentionOnly = availability {
                Text("Using this model with ZDR requested requires allowing Safety Retention in OpenRouter.")
                    .font(.caption).foregroundStyle(.secondary)
            } else if case .unavailable = availability {
                Text("OpenRouter may reject this model when ZDR is requested.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// The public OpenRouter catalog can be browsed before the user connects an account.
struct OpenRouterCatalogView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @Binding var selection: String
    @State private var search = ""
    @State private var finishedLoad = false
    @State private var reload = 0

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                Text("OpenRouter models").font(.title3.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                TextField("Search models", text: $search, prompt: Text("Name, provider, or model ID"))
                    .textFieldStyle(.roundedBorder)
                Text("For strict ZDR, choose a model marked \(OpenRouterZDRAvailability.available.label). With ZDR requested, allow Safety Retention in OpenRouter to use a Safety Retention-only model.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(16)
            modelList
            Divider()
            HStack {
                if !model.openRouterModels.isEmpty {
                    if model.isRefreshingOpenRouterModels {
                        ProgressView("Checking ZDR availability…").controlSize(.small)
                    } else if model.openRouterZDRNeedsRefresh {
                        Button("Refresh ZDR availability") { reload += 1 }
                    }
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            .padding(12)
        }
        .frame(width: 560, height: 480)
        .task(id: reload) {
            await model.refreshOpenRouterModels(force: reload > 0)
            if !Task.isCancelled { finishedLoad = true }
        }
    }

    private var modelList: some View {
        List {
            if model.openRouterModels.isEmpty {
                if !finishedLoad || model.isRefreshingOpenRouterModels {
                    ProgressView("Loading models…").controlSize(.small)
                } else {
                    Text("Couldn’t load OpenRouter models.").foregroundStyle(.secondary)
                    Button("Try again") {
                        finishedLoad = false
                        reload += 1
                    }
                }
            } else {
                let results = matches
                let selectedID = selection.trimmingCharacters(in: .whitespacesAndNewlines)
                if results.isEmpty { Text("No matching models").foregroundStyle(.secondary) }
                ForEach(results) { found in
                    let selected = found.id == selectedID
                    Button {
                        if selection != found.id { selection = found.id }
                        dismiss()
                    } label: {
                        modelRow(found, selected: selected)
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(selected ? .isSelected : [])
                }
            }
        }
    }

    private func modelRow(_ found: OpenRouter.Model, selected: Bool) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(found.name).lineLimit(1)
                    Text(found.inputCapability?.label ?? "Input unknown")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: true, vertical: false)
                }
                if found.name != found.id {
                    Text(found.id).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                OpenRouterZDRIndicator(availability: model.openRouterZDRAvailability(for: found.id))
                    .font(.caption.weight(.medium))
                if let context = found.contextLength {
                    Text("\(context.formatted(.number.notation(.compactName))) context")
                }
            }
            .font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: true, vertical: false)
            if selected { Image(systemName: "checkmark").accessibilityHidden(true) }
        }
        .contentShape(Rectangle())
    }

    private var matches: [OpenRouter.Model] {
        let terms = search.split(whereSeparator: \.isWhitespace)
        guard !terms.isEmpty else { return model.openRouterModels }
        return model.openRouterModels.filter { found in
            terms.allSatisfy { found.name.localizedCaseInsensitiveContains($0) || found.id.localizedCaseInsensitiveContains($0) }
        }
    }
}

private struct OpenRouterZDRIndicator: View {
    let availability: OpenRouterZDRAvailability

    var body: some View {
        HStack(spacing: 4) {
            switch availability {
            case .safetyRetentionOnly, .unavailable:
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .accessibilityHidden(true)
            case .available, .unknown:
                EmptyView()
            }
            Text(availability.label)
        }
        .accessibilityElement(children: .combine)
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
