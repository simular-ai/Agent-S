import SwiftUI
import AgentSCore

/// Lists installed models and the public Ollama library. Variant details load when a family is opened.
struct OllamaCatalogView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @Binding var selection: String
    @State private var search = ""
    @State private var family: String?
    @State private var library: CatalogLoad<[String]> = .loading
    @State private var reload = 0

    var body: some View {
        VStack(spacing: 0) {
            if let family {
                HStack(spacing: 12) {
                    Button {
                        self.family = nil
                    } label: {
                        Label("All models", systemImage: "chevron.left")
                    }
                    Text(family).font(.headline).lineLimit(1)
                    Spacer()
                }
                .padding(16)
                OllamaVariantList(family: family, selection: selection, select: select)
                    .id(family)
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Ollama models").font(.title3.weight(.semibold))
                        .accessibilityAddTraits(.isHeader)
                    TextField("Search models", text: $search, prompt: Text("Search by model name"))
                        .textFieldStyle(.roundedBorder)
                }
                .padding(16)
                libraryList
            }

            Divider()
            HStack {
                Link("Ollama model library", destination: OllamaCatalog.libraryURL)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            .padding(12)
        }
        .frame(width: 560, height: 480)
        .task { await model.refreshOllama() }
        .task(id: reload) {
            library = .loading
            let result = await CatalogLoad.fetch { try await OllamaCatalog.shared.families() }
            if !Task.isCancelled { library = result }
        }
    }

    private var libraryList: some View {
        List {
            if !installed.isEmpty {
                Section("Installed on This Mac") {
                    ForEach(installed) { entry in
                        let name = Ollama.normalizedName(entry.name)
                        OllamaCatalogRow(name: Ollama.displayName(for: name),
                                         input: model.ollamaInputCapability(for: name), detail: entry.sizeLabel,
                                         selected: name == Ollama.normalizedName(selection)) {
                            select(name)
                        }
                    }
                }
            }
            Section("Browse model families") {
                switch library {
                case .loading:
                    ProgressView("Loading the Ollama library…").controlSize(.small)
                case .failed:
                    Text("The Ollama library is unavailable.").foregroundStyle(.secondary)
                    Button("Try again") { reload += 1 }
                case let .loaded(names):
                    let familySearch = String(search.split(separator: ":", maxSplits: 1).first ?? "")
                    let matching = names.filter { matches($0, query: familySearch) }
                    if matching.isEmpty { Text("No matching model families").foregroundStyle(.secondary) }
                    ForEach(matching, id: \.self) { name in
                        Button {
                            family = name
                        } label: {
                            HStack {
                                Text(name)
                                Spacer()
                                Image(systemName: "chevron.right").foregroundStyle(.secondary)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var installed: [Ollama.Model] {
        model.ollamaModels.filter {
            !$0.usesRemoteInference && !Ollama.isCloudReference($0.name) && matches($0.name, query: search)
        }
    }

    private func select(_ name: String, input: ModelInputCapability? = nil) {
        model.rememberOllamaInput(input, for: name)
        let normalized = Ollama.normalizedName(name)
        if normalized != Ollama.normalizedName(selection) { selection = normalized }
        dismiss()
    }
}

private struct OllamaVariantList: View {
    let family: String
    let selection: String
    let select: (String, ModelInputCapability?) -> Void
    @EnvironmentObject private var model: AppModel
    @State private var search = ""
    @State private var variants: CatalogLoad<[OllamaCatalog.Variant]> = .loading
    @State private var reload = 0

    var body: some View {
        VStack(spacing: 0) {
            TextField("Filter variants", text: $search, prompt: Text("Filter by tag"))
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal, 16).padding(.bottom, 12)
            List {
                Section("Local variants") {
                    switch variants {
                    case .loading:
                        ProgressView("Loading variants…").controlSize(.small)
                    case .failed:
                        Text("Couldn’t load this model’s variants.").foregroundStyle(.secondary)
                        Button("Try again") { reload += 1 }
                    case let .loaded(entries):
                        let matching = entries.filter { matches($0.name, query: search) }
                        if entries.isEmpty {
                            Text("This model has no local variants.").foregroundStyle(.secondary)
                        } else if matching.isEmpty {
                            Text("No matching variants").foregroundStyle(.secondary)
                        }
                        ForEach(matching) { entry in
                            let installed = model.ollamaModels.first { $0.matches(entry.name) && !$0.usesRemoteInference }
                            let detail = installed?.sizeLabel
                                ?? entry.size.map { "About \($0)" }
                                ?? "Download size unavailable"
                            let input = installed == nil ? entry.inputCapability : model.ollamaInputCapability(for: entry.name)
                            OllamaCatalogRow(name: entry.name, input: input, detail: detail,
                                             selected: Ollama.normalizedName(entry.name) == Ollama.normalizedName(selection)) {
                                select(entry.name, entry.inputCapability)
                            }
                        }
                    }
                }
            }
        }
        .task(id: reload) {
            variants = .loading
            let result = await CatalogLoad.fetch { try await OllamaCatalog.shared.variants(for: family) }
            if !Task.isCancelled { variants = result }
        }
    }
}

private struct OllamaCatalogRow: View {
    let name: String
    let input: ModelInputCapability?
    let detail: String
    let selected: Bool
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(name).lineLimit(1)
                        if let input {
                            Text(input.label)
                                .font(.caption).foregroundStyle(.secondary)
                                .fixedSize()
                        }
                    }
                    Text(Ollama.Billing.local.label).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Text(detail).font(.caption).foregroundStyle(.secondary)
                if selected { Image(systemName: "checkmark").accessibilityHidden(true) }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// The selected model’s size and a warning when available RAM may fall short.
struct OllamaModelDetails: View {
    private struct Lookup: Hashable {
        let name: String
        let installed: Ollama.Model?
    }

    @EnvironmentObject private var model: AppModel
    let name: String
    let sandboxMemory: UInt64
    @State private var result: (lookup: Lookup, variant: OllamaCatalog.Variant?)?

    var body: some View {
        let lookup = Lookup(name: name, installed: model.ollamaModels.first { $0.matches(name) })
        VStack(alignment: .leading, spacing: 6) {
            Text(description(for: lookup))
                .foregroundStyle(.secondary)
            let billing = billing(for: lookup)
            if billing == .cloud {
                Link(billing.label, destination: Ollama.cloudPricingURL)
            } else {
                Text(billing.label).foregroundStyle(.secondary)
            }
            if let estimate = memoryEstimate(for: lookup) {
                TimelineView(.periodic(from: .now, by: 5)) { _ in
                    memoryWarning(for: estimate)
                }
            }
        }
        .font(.caption)
        .task(id: lookup) {
            guard lookup.installed == nil, presetSize == nil, !Ollama.isCloudReference(name) else { return }
            let variant = try? await publishedVariant()
            guard !Task.isCancelled else { return }
            result = (lookup, variant)
            model.rememberOllamaInput(variant?.inputCapability, for: name)
        }
    }

    @ViewBuilder private func memoryWarning(for estimate: OllamaMemoryEstimate) -> some View {
        let pendingSandbox = model.vm.machine == nil ? sandboxMemory : 0
        if let available = SystemMemory.available(),
           estimate.exceedsBudget(availableMemory: available, sandboxMemory: pendingSandbox) {
            let modelBudget = SystemMemory.format(available - min(available, pendingSandbox))
            Label {
                Text("This model may need more than the \(modelBudget) of RAM available. Try a smaller local model or a cloud model.")
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
        }
    }

    private var presetSize: String? {
        switch Ollama.normalizedName(name) {
        case Ollama.nemotronNano4B: "2.8 GB – 2.9 GB"
        case Ollama.nemotronBoltE4B: "7.2 GB – 7.9 GB"
        default: nil
        }
    }

    private func description(for lookup: Lookup) -> String {
        let loaded = result?.lookup == lookup ? result : nil
        if let installed = lookup.installed { return installed.sizeLabel }
        if Ollama.isCloudReference(name) { return "Hosted by Ollama" }
        if let size = presetSize ?? loaded?.variant?.size { return "Estimated download · \(size)" }
        return loaded == nil ? "Checking download size…" : "Download size unavailable"
    }

    private func billing(for lookup: Lookup) -> Ollama.Billing {
        if lookup.installed == nil, result?.lookup == lookup, result?.variant != nil { return .local }
        return Ollama.billing(for: name, installed: lookup.installed)
    }

    private func memoryEstimate(for lookup: Lookup) -> OllamaMemoryEstimate? {
        guard !Ollama.isCloudReference(name), lookup.installed?.usesRemoteInference != true else { return nil }
        if let installed = lookup.installed, installed.size > 0 {
            return OllamaMemoryEstimate(modelBytes: UInt64(installed.size))
        }
        if let size = presetSize ?? (result?.lookup == lookup ? result?.variant?.size : nil) {
            return OllamaMemoryEstimate(catalogSize: size)
        }
        return nil
    }

    private func publishedVariant() async throws -> OllamaCatalog.Variant? {
        guard let family = name.split(separator: ":", maxSplits: 1).first.map(String.init),
              try await OllamaCatalog.shared.families().contains(family) else { return nil }
        let variants = try await OllamaCatalog.shared.variants(for: family)
        return variants.first { Ollama.normalizedName($0.name) == Ollama.normalizedName(name) }
    }
}

private extension Ollama.Model {
    var sizeLabel: String {
        if Ollama.billing(for: name, installed: self) == .cloud { return "Hosted by Ollama" }
        if usesRemoteInference { return "Hosted inference" }
        guard size > 0 else { return "Installed" }
        return "Installed · \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))"
    }
}

private enum CatalogLoad<Value> {
    case loading
    case loaded(Value)
    case failed

    static func fetch(_ load: () async throws -> Value) async -> Self {
        do { return .loaded(try await load()) }
        catch { return .failed }
    }
}

private func matches(_ name: String, query: String) -> Bool {
    let normalizedName = name.replacingOccurrences(of: "-", with: " ")
    let terms = query.replacingOccurrences(of: "-", with: " ").split(whereSeparator: \.isWhitespace)
    return terms.allSatisfy { normalizedName.localizedCaseInsensitiveContains($0) }
}
