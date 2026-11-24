import SwiftUI

/// Installation and download actions for the selected Ollama model.
struct OllamaModelSetup: View {
    static let nemotron = "nemotron-3-nano:4b"

    @EnvironmentObject private var model: AppModel
    let name: String

    var body: some View {
        if case let .downloading(current, progress) = model.ollamaDownload, current == name {
            DownloadProgressView(label: "Downloading \(Self.title(for: name))…", progress: progress)
        } else if case let .verifying(current) = model.ollamaDownload, current == name {
            DownloadProgressView(label: "Checking the model in Ollama…", progress: 1)
        } else if !model.ollamaChecked {
            ProgressView("Checking Ollama…").controlSize(.small)
        } else if !model.ollamaReady {
            let needsUpdate = model.ollamaVersion != nil
            Text(needsUpdate
                 ? "Update Ollama to \(AppModel.minimumOllama) or later, then open it."
                 : "Install and open Ollama to run models on this Mac.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Link(needsUpdate ? "Update Ollama" : "Download Ollama",
                     destination: URL(string: "https://ollama.com/download/mac")!)
                Button("Check connection") { Task { await model.refreshOllama() } }
            }
        } else if !model.hasOllamaModel(name) {
            if case let .failed(current, message) = model.ollamaDownload, current == name {
                Text(message).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Button("Download \(Self.title(for: name))") { Task { await model.downloadOllamaModel(name) } }
                    .disabled(model.isDownloading)
                if name == Self.nemotron {
                    Text("About 2.9 GB").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    static func title(for name: String) -> String {
        switch name {
        case AppModel.defaultAgentModel: "Gemma 4"
        case nemotron: "Nemotron 3 Nano 4B"
        default: name
        }
    }
}
