import SwiftUI
import AppKit

/// Getting an API key in without leaving the screen: open the page that makes one, paste it,
/// done. Used wherever a key is needed (Sai, box, OpenRouter).
struct KeySetup: View {
    let service: String
    /// What a key starts with, to catch a wrong paste ("sapi_", "cb_", "sk-or-").
    let prefix: String
    let page: URL
    /// What to click on that page, in a few words.
    let howTo: String
    let hasKey: Bool
    let save: (String) async -> Bool
    let remove: () -> Void

    @State private var entered = ""
    @State private var status: String?
    @State private var saving = false

    var body: some View {
        if hasKey {
            LabeledContent("\(service) key") {
                HStack(spacing: 8) {
                    Label("Saved in Keychain", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).labelStyle(.titleAndIcon)
                    Button("Remove", action: remove)
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 10) {
                Text("Connect \(service)").font(.callout.weight(.semibold))
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    step(1)
                    VStack(alignment: .leading, spacing: 4) {
                        Button("Get a \(service) key") { NSWorkspace.shared.open(page) }
                            .buttonStyle(.borderedProminent).tint(Theme.accent)
                        Text(howTo).font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    step(2)
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Button("Paste key") { paste() }
                            Text("or").font(.caption).foregroundStyle(.secondary)
                            SecureField("\(prefix)…", text: $entered).onSubmit { commit(entered) }
                            Button("Save") { commit(entered) }
                                .disabled(entered.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || saving)
                        }
                        if let status { Text(status).font(.caption).foregroundStyle(.red) }
                        Text("It's kept in this Mac's Keychain.").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func step(_ number: Int) -> some View {
        Text("\(number)")
            .font(.caption.weight(.bold)).foregroundStyle(Theme.onAccent)
            .frame(width: 20, height: 20)
            .background(Circle().fill(Theme.accent))
    }

    private func paste() {
        let text = NSPasteboard.general.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !text.isEmpty else { status = "The clipboard is empty: copy the key first."; return }
        commit(text)
    }

    private func commit(_ key: String) {
        let key = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard key.hasPrefix(prefix) else {
            status = "That doesn't look like a \(service) key (they start with \(prefix))."
            return
        }
        saving = true
        Task {
            let ok = await save(key)
            saving = false
            entered = ""
            status = ok ? nil : "Couldn't save the key to the Keychain."
        }
    }
}

extension KeySetup {
    /// The three keys the app can use, configured once.
    static func sai(_ model: AppModel) -> KeySetup {
        KeySetup(service: "Sai", prefix: "sapi_", page: URL(string: "https://platform.simular.ai/api-keys")!,
                 howTo: "Sign in, click Create key, then copy it (it's shown once).",
                 hasKey: model.hasSaiKey, save: { await model.saveSaiKey($0) }, remove: { model.removeSaiKey() })
    }

    static func box(_ model: AppModel) -> KeySetup {
        KeySetup(service: "Simular Box", prefix: "cb_", page: URL(string: "https://box.simular.ai/access")!,
                 howTo: "Sign in, click Create key under Keys and devices, then copy it (it's shown once).",
                 hasKey: model.hasBoxKey, save: { model.saveBoxKey($0) }, remove: { model.removeBoxKey() })
    }

    static func openRouter(_ model: AppModel) -> KeySetup {
        KeySetup(service: "OpenRouter", prefix: "sk-or-", page: URL(string: "https://openrouter.ai/keys")!,
                 howTo: "Sign in, click Create Key, then copy it. Setting a credit limit on it is a good idea.",
                 hasKey: model.hasOpenRouterKey, save: { model.saveOpenRouterKey($0) }, remove: { model.removeOpenRouterKey() })
    }
}
