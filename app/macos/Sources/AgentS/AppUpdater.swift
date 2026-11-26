import Combine
import Foundation
import Sparkle

@MainActor
final class AppUpdater: NSObject, ObservableObject, SPUUpdaterDelegate {
    @Published private(set) var canCheckForUpdates = false

    private var controller: SPUStandardUpdaterController?
    private var isInstallingUpdate = false

    // Developer builds can run from source without a release key or feed.
    let isConfigured = Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") is String
        && Bundle.main.object(forInfoDictionaryKey: "SUFeedURL") is String

    func start() {
        guard isConfigured, controller == nil, !GuestFiles.legacyAppRunning else { return }

        let controller = SPUStandardUpdaterController(startingUpdater: true, updaterDelegate: self, userDriverDelegate: nil)
        self.controller = controller
        controller.updater.publisher(for: \.canCheckForUpdates)
            .assign(to: &$canCheckForUpdates)
    }

    func checkForUpdates() {
        controller?.checkForUpdates(nil)
    }

    // Sparkle sends a normal quit request. Wait in that request so a turn started
    // during the download can finish before the app saves the sandbox.
    func waitBeforeQuitting(_ model: AppModel) async -> Bool {
        guard isInstallingUpdate else { return true }

        while isInstallingUpdate, model.hasWorkToFinishBeforeUpdate {
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return false
            }
        }
        return isInstallingUpdate
    }

    func updater(_ updater: SPUUpdater, willInstallUpdate item: SUAppcastItem) {
        isInstallingUpdate = true
    }

    func updater(_ updater: SPUUpdater, didFinishUpdateCycleFor updateCheck: SPUUpdateCheck, error: Error?) {
        isInstallingUpdate = false
    }
}
