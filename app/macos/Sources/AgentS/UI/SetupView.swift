import SwiftUI
import AgentSCore

/// Global Settings for Mac permissions, the local sandbox, account keys, and Ollama.
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
                Text("Applies when you turn on a powered-off sandbox.")
                    .font(.caption).foregroundStyle(.secondary)
                if case let .preparing(step) = model.phase {
                    if case let .downloadingDebian(progress) = step {
                        DownloadProgressView(label: step.label, progress: progress)
                    } else {
                        Text(step.label)
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

            Section("Sandbox Network") {
                Picker("Connection", selection: $model.sandboxNetworkMode) {
                    Text(SandboxNetworkMode.followMac.title).tag(SandboxNetworkMode.followMac)
                    Text(SandboxNetworkMode.standard.title).tag(SandboxNetworkMode.standard)
                }
                Text(model.sandboxNetworkMode == .followMac
                     ? "Uses this Mac’s IPv4 routes and DNS, including VPN rules."
                     : "Uses Apple’s shared virtual machine network.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.sandboxNetworkModeRequiresRestart {
                    Text("Restart the sandbox to apply this change.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Accounts") {
                KeySetup.sai(model)
                KeySetup.box(model)
                KeySetup.openRouter(model)
                OpenRouterZDRSetting(isOn: $model.openRouterRequiresZDR)
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
                Text("Ollama runs models on this Mac. Simular Box agents run their harness on this Mac and use the Box’s tools. Agent S adds your OpenRouter key from the Mac’s Keychain to each OpenRouter request.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .onAppear { model.refreshMacAccess() }
    }
}
