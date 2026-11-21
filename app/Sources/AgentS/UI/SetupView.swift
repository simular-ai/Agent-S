import SwiftUI
import AgentSCore

/// Global Settings for Mac permissions, the private computer, account keys, and Ollama.
struct SetupView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Form {
            Section("Mac Permissions") {
                MacAccessRows()
            }

            Section("Computer") {
                LabeledContent("Linux") {
                    Text(model.files.isPrepared ? "Debian 13, ready" : "Not downloaded yet")
                }
                Picker("Memory", selection: $model.computerMemory) {
                    ForEach(LinuxVM.MemoryAllocation.allCases, id: \.self) { allocation in
                        Text(SystemMemory.format(allocation.bytes)).tag(allocation)
                    }
                }
                Text("Applies the next time you turn on the private computer.")
                    .font(.caption).foregroundStyle(.secondary)
                if case let .preparing(step) = model.phase {
                    VStack(alignment: .leading) {
                        Text(step)
                        if let progress = model.downloadProgress { ProgressView(value: progress) }
                    }
                } else if !model.files.isPrepared {
                    Button("Set up the computer") { Task { await model.prepare() } }
                    Text("Downloads Debian (about 300 MB) and makes a 32 GB disk that only uses the space the computer fills.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                LabeledContent("Shared folder") {
                    Button("Show in Finder") { NSWorkspace.shared.open(model.files.share) }
                }
            }

            Section("Accounts") {
                KeySetup.sai(model)
                KeySetup.box(model)
                KeySetup.openRouter(model)
                Text("Keys stay in this Mac's Keychain and are shared by all your agents. They never enter the computer.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Ollama") {
                LabeledContent("Status") {
                    if let version = model.ollamaVersion {
                        Text(model.ollamaReady ? "Running \(version)" : "\(version): needs \(AppModel.minimumOllama) or newer")
                            .foregroundStyle(model.ollamaReady ? Color.primary : Color.red)
                    } else if model.ollamaChecked {
                        Text("Not running").foregroundStyle(.red)
                    } else {
                        Text("Unchecked").foregroundStyle(.secondary)
                    }
                }
                if model.ollamaChecked, model.ollamaVersion == nil {
                    Text("Install Ollama from ollama.com and open it. It stays on this Mac; the computer reaches it through this app only.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button(model.ollamaChecked ? "Check again" : "Check status") { Task { await model.refreshOllama() } }
            }

            Section {
                Text("Ollama runs models on this Mac. Simular Box agents run their harness in Agent S and use the Box’s tools. For OpenRouter, Agent S adds the key from this Mac’s Keychain when it sends each request.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .onAppear { model.refreshMacAccess() }
    }
}
