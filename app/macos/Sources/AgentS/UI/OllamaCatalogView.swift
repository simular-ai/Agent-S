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
                        OllamaCatalogRow(name: name, detail: entry.sizeLabel, selected: name == Ollama.normalizedName(selection)) {
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

    private func select(_ name: String) {
        let normalized = Ollama.normalizedName(name)
        if normalized != Ollama.normalizedName(selection) { selection = normalized }
        dismiss()
    }
}

private struct OllamaVariantList: View {
    let family: String
    let selection: String
    let select: (String) -> Void
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
                            OllamaCatalogRow(name: entry.name, detail: detail,
                                             selected: Ollama.normalizedName(entry.name) == Ollama.normalizedName(selection)) {
                                select(entry.name)
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
    let detail: String
    let selected: Bool
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            HStack {
                Text(name)
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

/// The selected model’s reported footprint on this Mac or its estimated download from Ollama.
struct OllamaModelSize: View {
    @EnvironmentObject private var model: AppModel
    let name: String
    @State private var lookup: (name: String, size: String?)?

    var body: some View {
        Text(description).font(.caption).foregroundStyle(.secondary)
            .task(id: name) {
                guard installed == nil else { return }
                let size = try? await publishedSize()
                if !Task.isCancelled { lookup = (name, size) }
            }
    }

    private var installed: Ollama.Model? {
        model.ollamaModels.first { $0.matches(name) }
    }

    private var description: String {
        if let installed { return installed.sizeLabel }
        if Ollama.isCloudReference(name) { return "Hosted by Ollama" }
        if lookup?.name == name, let size = lookup?.size { return "Estimated download · \(size)" }
        if name == OllamaModelSetup.nemotron { return "Estimated download · 2.8–2.9 GB" }
        return lookup?.name == name ? "Download size unavailable" : "Checking download size…"
    }

    private func publishedSize() async throws -> String? {
        guard !Ollama.isCloudReference(name),
              let family = name.split(separator: ":", maxSplits: 1).first.map(String.init),
              try await OllamaCatalog.shared.families().contains(family) else { return nil }
        let variants = try await OllamaCatalog.shared.variants(for: family)
        return variants.first { Ollama.normalizedName($0.name) == Ollama.normalizedName(name) }?.size
    }
}

private extension Ollama.Model {
    var sizeLabel: String {
        if usesRemoteInference || Ollama.isCloudReference(name) { return "Hosted by Ollama" }
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
