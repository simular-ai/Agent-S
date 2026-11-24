import AppKit
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
            OllamaInstallation()
            OllamaModelSize(name: name)
        } else if !model.hasOllamaModel(name) {
            if case let .failed(current, message) = model.ollamaDownload, current == name {
                Text(message).font(.caption).foregroundStyle(.red)
            }
            Button("Download \(Self.title(for: name))") { Task { await model.downloadOllamaModel(name) } }
                .disabled(model.isDownloading)
            OllamaModelSize(name: name)
        } else {
            OllamaModelSize(name: name)
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

private struct OllamaInstallation: View {
    private static let downloadURL = URL(string: "https://ollama.com/download/Ollama.dmg")!

    private enum Check: Equatable {
        case once(UUID)
        case untilReady(UUID)
    }

    @EnvironmentObject private var model: AppModel
    @State private var check: Check?
    @State private var isOpening = false
    @State private var actionError: String?

    var body: some View {
        let app = installedApp
        let needsUpdate = model.ollamaVersion != nil
        VStack(alignment: .leading, spacing: 8) {
            Text(instructions(needsUpdate: needsUpdate, installed: app != nil))
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                if let app, !needsUpdate {
                    Button(isOpening ? "Opening Ollama…" : "Open Ollama") {
                        Task { await openOllama(at: app) }
                    }
                    .disabled(isOpening)
                    .buttonStyle(.borderedProminent)
                } else {
                    Button(needsUpdate ? "Update Ollama" : "Download Ollama") { download() }
                        .buttonStyle(.borderedProminent)
                }
                Button("Check again") { check = .once(UUID()) }
            }
            if let actionError {
                Text(actionError).font(.caption).foregroundStyle(.red)
            }
        }
        .task(id: check) {
            guard let check else { return }
            switch check {
            case .once:
                await model.refreshOllama(timeout: 2)
            case .untilReady:
                let deadline = ContinuousClock.now + .seconds(60)
                while !Task.isCancelled, !model.ollamaReady, ContinuousClock.now < deadline {
                    await model.refreshOllama(timeout: 2)
                    if model.ollamaReady { break }
                    do { try await Task.sleep(for: .seconds(2)) } catch { return }
                }
            }
            if self.check == check { self.check = nil }
        }
        .onAppear(perform: checkIfIdle)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            checkIfIdle()
        }
    }

    private func checkIfIdle() {
        if check == nil { check = .once(UUID()) }
    }

    private var installedApp: URL? {
        let files = FileManager.default
        let locations = [
            URL(fileURLWithPath: "/Applications/Ollama.app", isDirectory: true),
            files.homeDirectoryForCurrentUser.appendingPathComponent("Applications/Ollama.app", isDirectory: true),
        ]
        return locations.first { location in
            guard let executable = Bundle(url: location)?.executableURL else { return false }
            return files.isExecutableFile(atPath: executable.path)
        }
    }

    private func instructions(needsUpdate: Bool, installed: Bool) -> String {
        if needsUpdate {
            return "Ollama \(AppModel.minimumOllama) or later is required. Quit Ollama, open the downloaded disk image, replace Ollama in Applications, then open it again."
        }
        if installed { return "Open Ollama to run this model on your Mac." }
        return "Open the downloaded disk image, drag Ollama to Applications, then open it."
    }

    private func download() {
        actionError = nil
        if NSWorkspace.shared.open(Self.downloadURL) {
            check = .untilReady(UUID())
        } else {
            actionError = "Couldn’t open the Ollama download. Try again."
        }
    }

    private func openOllama(at app: URL) async {
        isOpening = true
        actionError = nil
        defer { isOpening = false }
        do {
            _ = try await NSWorkspace.shared.openApplication(at: app, configuration: NSWorkspace.OpenConfiguration())
            check = .untilReady(UUID())
        } catch {
            actionError = "Couldn’t open Ollama. Try opening it from Applications."
        }
    }
}
