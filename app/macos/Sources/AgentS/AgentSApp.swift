import SwiftUI

@main
struct AgentSApp: App {
    @StateObject private var model = AppModel()
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate

    var body: some Scene {
        WindowGroup("Agent S") {
            ContentView()
                .environmentObject(model)
                .frame(minHeight: 560)
                .onAppear {
                    delegate.model = model
                    model.refreshMacAccess()
                }
                .onChange(of: model.selectedAgentID) { _, _ in
                    if model.privacy == .thisMac { model.refreshMacAccess() }
                }
        }
        .windowStyle(.hiddenTitleBar)
        // The window is never narrower than its content, so opening the computer panel widens it.
        .windowResizability(.contentMinSize)
    }
}

/// Shuts the guest down cleanly when the app quits, so its disk is never left mid-write.
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?

    /// Two copies of the app must not share one computer: if the earlier "Saige" build is
    /// running, ask the user to quit it first.
    func applicationDidFinishLaunching(_ notification: Notification) {
        guard GuestFiles.legacyAppRunning else { return }
        let alert = NSAlert()
        alert.messageText = "Quit Saige first"
        alert.informativeText = "Agent S is the new name for Saige. Quit the running Saige app, then open Agent S again: it will bring over your agents and the computer."
        alert.addButton(withTitle: "Quit Agent S")
        alert.runModal()
        NSApp.terminate(nil)
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        model?.refreshMacAccess()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model else { return .terminateNow }
        model.stopAppWork()
        guard model.vm.machine != nil else { return .terminateNow }
        Task { @MainActor in
            await model.vm.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
