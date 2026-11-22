import SwiftUI
import AgentSCore

/// The Sai harness: the user's Sai API key (Keychain), which Sai computer runs the tasks, and the
/// model. Agent S’s own computer is listed, but Sai can't drive it through the API yet.
struct SaiSettings: View {
    @EnvironmentObject private var model: AppModel
    /// Keys are set in the app's Settings; an agent's settings only show one that is missing.
    var showKey = false
    @State private var key = ""
    @State private var keyStatus: String?

    var body: some View {
        Section("Sai") {
            if showKey || !hasKey { KeySetup.sai(model) }
            if let keyStatus { Text(keyStatus).font(.caption).foregroundStyle(.red) }
            if let status = model.saiStatus { Text(status).font(.caption).foregroundStyle(.red) }

            Picker("Computer", selection: $model.saiMachine) {
                ForEach(model.saiMachines) { machine in
                    Text("\(machine.name) · \(machine.kind == "own" ? "your \(machine.os ?? "PC")" : "Sai cloud")\(machine.online ? "" : " (offline)")")
                        .tag(machine.id)
                }
                Text("This Mac's computer (coming: needs the API path)").tag(SaiAPI.localMachine)
                if model.saiMachine.isEmpty { Text("None").tag("") }
            }
            Picker("Model", selection: $model.saiModel) {
                ForEach(model.saiModels) { choice in
                    Text(choice.allowed ? choice.name : "\(choice.name) (not on your plan)").tag(choice.id)
                }
                if !model.saiModels.contains(where: { $0.id == model.saiModel }) { Text(model.saiModel).tag(model.saiModel) }
            }
            Button("Refresh") { Task { await model.refreshSai() } }
            Text("With the Sai harness, Sai (Simular's agent) runs your tasks on the Sai computer you pick, using your Sai plan. Your key stays in this Mac's Keychain.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .task { await model.refreshSai() }
    }

    private var hasKey: Bool { model.hasSaiKey }
}
