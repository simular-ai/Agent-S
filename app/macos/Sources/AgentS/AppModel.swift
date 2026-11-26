import Foundation
import AppKit
import AgentSCore

/// The app's state and the order things happen in: prepare the guest, boot it, find it, relay
/// Ollama to it, wait for its first-boot setup, then chat.
@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable {
        case needsSetup
        case preparing(GuestImage.PreparationStep)
        case stopped
        case booting(String)
        case stopping
        case ready
        case failed(String)
    }

    enum AccountStatus: Equatable {
        case unchecked
        case checking
        case signedOut
        case connecting
        case awaitingSignIn
        case awaitingDeviceCode(CodexAccount.DeviceCode)
        case awaitingBrowserApproval(URL)
        case connected(HarnessAccountPlan)
        case failed(String)

        var isConnected: Bool {
            if case .connected = self { return true }
            return false
        }
    }

    enum OllamaDownload: Equatable {
        case downloading(name: String, progress: Double?)
        case verifying(name: String)
        case failed(name: String, message: String)
    }

    private struct OpenRouterZDRCheck {
        let catalog: OpenRouterZDRCatalog
        let publicModelIDs: Set<String>?
        let checkedAt = ContinuousClock.now

        var isFresh: Bool { checkedAt.duration(to: .now) < .seconds(300) }
    }

    private struct AccountRoute: Hashable {
        let provider: ModelProvider
        let inSandbox: Bool

        init(_ config: AgentConfig) {
            provider = config.provider
            inSandbox = config.usesLocalComputer
        }
    }

    /// The preselected Ollama model for new agents.
    static let defaultAgentModel = Ollama.nemotronNano4B
    /// Ollama >= 0.14 serves the Anthropic Messages API Claude Code speaks.
    static let minimumOllama = "0.14.0"
    private static let computerMemoryKey = "computerMemoryGB"
    private static let sandboxNetworkModeKey = "sandboxNetworkMode"
    private nonisolated static let openRouterZDRKey = "openRouterRequiresZDR"

    private nonisolated static var storedOpenRouterRequiresZDR: Bool {
        UserDefaults.standard.object(forKey: openRouterZDRKey) as? Bool ?? true
    }

    @Published private(set) var phase: Phase
    @Published private(set) var guestIP: String?
    @Published private(set) var isSavingSandbox = false
    private var sandboxStartup: Task<Void, Never>?
    private var attemptedSandboxRestore = false
    private var sandboxCanCheckpoint = false
    private var sandboxNeedsFullToolInstall = false
    @Published var computerMemory = LinuxVM.MemoryAllocation(rawValue: UserDefaults.standard.integer(forKey: AppModel.computerMemoryKey)) ?? .twoGB {
        didSet { UserDefaults.standard.set(computerMemory.rawValue, forKey: Self.computerMemoryKey) }
    }
    @Published var sandboxNetworkMode = UserDefaults.standard.string(forKey: AppModel.sandboxNetworkModeKey)
        .flatMap(SandboxNetworkMode.init(rawValue:)) ?? .followMac {
        didSet { UserDefaults.standard.set(sandboxNetworkMode.rawValue, forKey: Self.sandboxNetworkModeKey) }
    }
    var sandboxNetworkModeRequiresRestart: Bool {
        vm.networkMode.map { $0 != sandboxNetworkMode } ?? false
    }

    @Published private(set) var ollamaChecked = false
    @Published private(set) var ollamaVersion: String?
    @Published private(set) var ollamaModels: [Ollama.Model] = []
    @Published private var installedOllamaInputs: [Ollama.Model: ModelInputCapability] = [:]
    @Published private var libraryOllamaInputs: [String: ModelInputCapability] = [:]
    private var ollamaInputsTask: Task<Void, Never>?
    @Published private(set) var ollamaDownload: OllamaDownload?
    private var ollamaRefreshGeneration = 0
    private var appliedOllamaRefreshGeneration = 0
    @Published private(set) var openRouterModels: [OpenRouter.Model] = []
    @Published private(set) var isRefreshingOpenRouterModels = false
    @Published private var openRouterZDRCheck: OpenRouterZDRCheck?
    @Published private(set) var hasOpenRouterKey = OpenRouter.hasKey
    @Published var openRouterRequiresZDR = AppModel.storedOpenRouterRequiresZDR {
        didSet { UserDefaults.standard.set(openRouterRequiresZDR, forKey: Self.openRouterZDRKey) }
    }
    // Sai harness (cloud agents): the user's Sai key, computers and models.
    @Published private(set) var hasSaiKey = SaiAPI.hasKey
    @Published private(set) var saiKeyRevision = 0
    @Published private(set) var saiMachines: [SaiAPI.Machine] = []
    @Published private(set) var saiModels: [SaiAPI.Model] = []
    @Published private(set) var saiStatus: String?
    @Published private var saiScreenStates: [String: SaiAPI.ScreenState] = [:]
    private let openRouterProxy = OpenRouterProxy(requiresZDR: { AppModel.storedOpenRouterRequiresZDR })
    @Published private var accountStatuses: [AccountRoute: AccountStatus] = [:]
    @Published private(set) var sandboxSignInRequest = UUID()
    private var accountChecks: [AccountRoute: UUID] = [:]
    private var accountSignIns: [AccountRoute: Task<Void, Never>] = [:]
    private var sandboxAccountGeneration = UUID()

    let files = GuestFiles.standard
    let vm: LinuxVM
    var hasSavedSandbox: Bool { vm.hasSavedState }

    // MARK: Agents

    /// The user's agents, private ones (on this Mac) and cloud ones; each with its own context.
    @Published private(set) var agents: [AgentSession] = []
    @Published private(set) var needsFirstAgentSetup = false
    @Published private(set) var needsFirstAgentPreparation = false
    @Published var selectedAgentID: UUID {
        didSet {
            if !needsFirstAgentSetup { UserDefaults.standard.set(selectedAgentID.uuidString, forKey: "selectedAgent") }
        }
    }
    /// The selected agent, whose chat and settings the window shows.
    var agent: AgentSession { agents.first { $0.id == selectedAgentID } ?? agents[0] }

    func agents(at location: AgentLocation) -> [AgentSession] {
        agents.filter { $0.config.privacy.location == location }
    }

    /// Puts an agent in another's place in the sidebar. Agents stay in their own section; call
    /// `saveAgents()` once the move is done to keep the order.
    func moveAgent(_ id: UUID, to target: UUID) {
        guard let from = agents.first(where: { $0.id == id }), let to = agents.first(where: { $0.id == target }),
              from.config.privacy.location == to.config.privacy.location,
              let reordered = agents.moving({ $0.id == id }, to: { $0.id == target }) else { return }
        agents = reordered
    }

    /// Moves an agent one place up or down within its section, for VoiceOver and keyboard users.
    func moveAgent(_ session: AgentSession, by offset: Int) {
        let list = agents(at: session.config.privacy.location)
        guard let index = list.firstIndex(where: { $0.id == session.id }), list.indices.contains(index + offset) else { return }
        moveAgent(session.id, to: list[index + offset].id)
        saveAgents()
    }

    // The selected agent's settings, so views keep binding to `model.harness` and friends.
    private func update(_ change: (inout AgentConfig) -> Void) {
        objectWillChange.send()
        change(&agent.config)
    }
    var privacy: PrivacyMode { agent.config.privacy }
    var harness: Harness {
        get { agent.config.harness }
        set {
            guard !agent.isRunning, Harness.allowed(in: privacy, computer: computer).contains(newValue) else { return }
            update {
                $0.harness = newValue
                if $0.usesLocalComputer,
                   let provider = ModelProvider.fitting($0.provider, at: .linuxVM, harness: newValue, computer: nil) {
                    $0.provider = provider
                    $0.privacy = AgentLocation.linuxVM.mode(for: provider)
                }
                $0 = $0.enforcingPrivacy
            }
        }
    }
    /// The selected cloud computer, if any.
    var computer: CloudComputer? { agent.config.computer }
    var boxName: String {
        get { agent.config.boxName ?? "" }
        set { update { $0.boxName = newValue.trimmingCharacters(in: .whitespaces).isEmpty ? nil : newValue } }
    }
    @Published private(set) var hasBoxKey = BoxAccount.hasKey

    func saveBoxKey(_ key: String) -> Bool {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, BoxAccount.save(key: trimmed) else { return false }
        hasBoxKey = true
        return true
    }

    func removeBoxKey() {
        BoxAccount.delete()
        hasBoxKey = false
    }
    var provider: ModelProvider {
        get { agent.config.provider }
        set {
            guard !agent.isRunning, availableProviders.contains(newValue) else { return }
            update {
                $0.provider = newValue
                if $0.usesLocalComputer { $0.privacy = AgentLocation.linuxVM.mode(for: newValue) }
            }
        }
    }
    var availableProviders: [ModelProvider] {
        ModelProvider.allowed(at: privacy.location, harness: harness, computer: computer)
    }
    var agentModel: String {
        get { agent.config.ollamaModel }
        set { update { $0.ollamaModel = newValue } }
    }
    var openRouterModel: String {
        get { agent.config.openRouterModel }
        set { update { $0.openRouterModel = newValue } }
    }
    var saiMachine: String {
        get { agent.config.saiMachine }
        set { update { $0.saiMachine = newValue } }
    }
    var saiModel: String {
        get { agent.config.saiModel }
        set { update { $0.saiModel = newValue } }
    }

    /// Saves a new agent with a fresh context and the next headshot color.
    func createAgent(_ config: AgentConfig, selectAfterCreation: Bool = true) throws {
        guard !needsFirstAgentSetup, !GuestFiles.legacyAppRunning else {
            throw FirstAgentSetupError.unavailable
        }
        var config = config.enforcingPrivacy
        if config.avatarColor == nil { config.avatarColor = agents.count % AgentAvatar.palette.count }
        let session = AgentSession(config: config)
        try files.ensureFolders()
        try writeAgents(agents.map(\.saved) + [session.saved], to: agentsFile)
        adopt(session)
        if selectAfterCreation { selectedAgentID = config.id }
    }

    func completeFirstAgentSetup(_ config: AgentConfig) throws {
        guard needsFirstAgentSetup, agents.count == 1, let starter = agents.first,
              starter.id == selectedAgentID, !GuestFiles.legacyAppRunning,
              !FileManager.default.fileExists(atPath: files.root.path) else {
            throw FirstAgentSetupError.unavailable
        }
        guard config.usesLocalComputer else { throw FirstAgentSetupError.invalidLocation }

        var config = config
        config.privacy = AgentLocation.linuxVM.mode(for: config.provider)
        config = config.enforcingPrivacy
        config.id = starter.id
        if config.avatarColor == nil { config.avatarColor = 0 }
        var snapshot = starter.saved
        snapshot.config = config
        try writeFirstAgent(snapshot)
        starter.config = config
        needsFirstAgentPreparation = true
        needsFirstAgentSetup = false
        selectedAgentID = starter.id
    }

    func finishFirstAgentPreparation() throws {
        try FileManager.default.removeItem(at: firstAgentPreparationFile)
        needsFirstAgentPreparation = false
    }

    private enum FirstAgentSetupError: LocalizedError {
        case unavailable
        case invalidLocation

        var errorDescription: String? {
            switch self {
            case .unavailable: "Agent setup is unavailable."
            case .invalidLocation: "Choose a model for the sandbox on this Mac."
            }
        }
    }

    func rename(_ session: AgentSession, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        session.config.name = trimmed
    }

    /// Deletes an agent and its context; the last agent stays.
    func delete(_ session: AgentSession) {
        guard agents.count > 1, !GuestFiles.legacyAppRunning else { return }
        do {
            try session.deleteSavedMemory()
        } catch {
            session.noticeFromApp("Couldn't delete this agent's memory. \(error.localizedDescription)")
            return
        }
        session.stop()
        macTurnPanel.remove(session)
        agents.removeAll { $0.id == session.id }
        if selectedAgentID == session.id { selectedAgentID = agents[0].id }
        saveAgents()
    }

    func select(_ session: AgentSession) { selectedAgentID = session.id }

    /// Reorder within the agent's computer group and save the new sidebar order.
    func moveAgent(_ sourceID: UUID, relativeTo target: AgentSession, after: Bool) {
        guard sourceID != target.id, !needsFirstAgentSetup, !GuestFiles.legacyAppRunning else { return }

        let slots = agents.indices.filter { agents[$0].config.privacy.location == target.config.privacy.location }
        var group = slots.map { agents[$0] }
        guard let sourceIndex = group.firstIndex(where: { $0.id == sourceID }),
              let targetIndex = group.firstIndex(where: { $0 === target }) else { return }

        var destination = targetIndex + (after ? 1 : 0)
        if sourceIndex < targetIndex { destination -= 1 }
        guard destination != sourceIndex else { return }

        let source = group.remove(at: sourceIndex)
        group.insert(source, at: destination)
        var reordered = agents
        for (slot, session) in zip(slots, group) { reordered[slot] = session }

        do {
            try files.ensureFolders()
            try writeAgents(reordered.map(\.saved), to: agentsFile)
            agents = reordered
        } catch {
            NSLog("Agent S: couldn't save the sidebar order: %@", error.localizedDescription)
        }
    }

    func revealChatItem(_ id: String, in session: AgentSession) {
        guard agents.contains(where: { $0 === session }) else { return }
        select(session)
        session.revealChatItem(id)
    }

    // MARK: Headshots

    /// Picks an image and keeps a copy in the data folder, so the agent's picture survives the
    /// original being moved or deleted.
    func chooseAvatar(for session: AgentSession) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a picture for \(session.config.name)"
        guard panel.runModal() == .OK, let source = panel.url else { return }
        let name = "\(session.id.uuidString).\(source.pathExtension.isEmpty ? "png" : source.pathExtension.lowercased())"
        let target = files.avatars.appendingPathComponent(name)
        do {
            try FileManager.default.createDirectory(at: files.avatars, withIntermediateDirectories: true)
            if let old = session.config.avatarImage { try? FileManager.default.removeItem(at: files.avatars.appendingPathComponent(old)) }
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.copyItem(at: source, to: target)
            session.config.avatarImage = name
        } catch {
            NSLog("Agent S: couldn't use that picture: %@", "\(error)")
        }
    }

    func setAvatarColor(_ index: Int, for session: AgentSession) {
        session.config.avatarColor = index
    }

    func clearAvatar(for session: AgentSession) {
        if let old = session.config.avatarImage { try? FileManager.default.removeItem(at: files.avatars.appendingPathComponent(old)) }
        session.config.avatarImage = nil
    }

    private func adopt(_ session: AgentSession) {
        session.attachMemory(dataFolder: files.root)
        session.onChange = { [weak self] in
            self?.objectWillChange.send()
            self?.saveAgents()
        }
        session.onMacUIActivity = { [weak self] session, needsApproval in
            self?.showMacTurnPanel(for: session, presentation: .macUI(needsApproval: needsApproval))
        }
        session.onTurnEnded = { [weak self] session, succeeded in
            guard succeeded, !session.followUpsPaused else { return }
            self?.sendQueuedFollowUp(for: session)
        }
        session.onForcedFollowUpReady = { [weak self] session, id in
            self?.sendQueuedFollowUp(for: session, forcedID: id)
        }
        agents.append(session)
    }

    private var agentsFile: URL { files.root.appendingPathComponent("agents.json") }
    private var firstAgentPreparationFile: URL { files.root.appendingPathComponent("first-agent-preparation") }

    /// Chats can hold page text and tool output: the file is readable by this user only.
    func saveAgents() {
        // While an earlier build runs, this copy only shows "quit it first" and writes nothing.
        guard !needsFirstAgentSetup, !GuestFiles.legacyAppRunning else { return }
        try? files.ensureFolders()
        try? writeAgents(agents.map(\.saved), to: agentsFile)
    }

    private func writeAgents(_ snapshot: [SavedAgent], to url: URL) throws {
        let data = try JSONEncoder().encode(snapshot)
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    /// Publish the first support folder only after its private agent file has been written.
    private func writeFirstAgent(_ snapshot: SavedAgent) throws {
        let manager = FileManager.default
        let parent = files.root.deletingLastPathComponent()
        let staging = parent.appendingPathComponent(".agent-s-setup-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? manager.removeItem(at: staging) }
        try GuestFiles(root: staging).ensureFolders()
        try writeAgents([snapshot], to: staging.appendingPathComponent(agentsFile.lastPathComponent))
        try Data(snapshot.config.id.uuidString.utf8)
            .write(to: staging.appendingPathComponent(firstAgentPreparationFile.lastPathComponent), options: .atomic)
        try manager.moveItem(at: staging, to: files.root)
    }

    private var isFreshInstallation: Bool {
        guard !GuestFiles.legacyAppRunning else { return false }

        let folders = [files.root] + GuestFiles.legacy.map { _, name in
            files.root.deletingLastPathComponent().appendingPathComponent(name, isDirectory: true)
        }
        guard !folders.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return false }

        let priorSettings = ["selectedAgent", "agentModel", "openRouterModel", "harness", "harness.private", "harness.cloud",
                             "provider", "provider.cloud", "saiMachine", "saiModel", Self.computerMemoryKey, Self.openRouterZDRKey]
        let defaults = UserDefaults.standard
        guard !priorSettings.contains(where: { defaults.object(forKey: $0) != nil }) else { return false }
        return !GuestFiles.legacy.contains { bundleID, _ in
            let prior = defaults.persistentDomain(forName: bundleID) ?? [:]
            return priorSettings.contains { prior[$0] != nil }
        }
    }

    /// Saved agents, or the first ones from the settings before agents existed.
    private func loadAgents() {
        if let data = try? Data(contentsOf: agentsFile),
           let saved = try? JSONDecoder().decode([SavedAgent].self, from: data), !saved.isEmpty {
            saved.forEach { adopt(AgentSession(saved: $0)) }
            // Agents saved before headshots get a color each.
            for (index, session) in agents.enumerated() where session.config.avatarColor == nil {
                session.config.avatarColor = index % AgentAvatar.palette.count
            }
            return
        }
        let defaults = UserDefaults.standard
        let ollamaModel = defaults.string(forKey: "agentModel") ?? Self.defaultAgentModel
        let openRouterModel = defaults.string(forKey: "openRouterModel") ?? OpenRouter.suggested[0]
        func harness(_ key: String) -> Harness? { defaults.string(forKey: key).flatMap(Harness.init(rawValue:)) }
        let privateHarness = [harness("harness.private"), harness("harness")].compactMap { $0 }
            .first { $0.runsInComputer } ?? .claudeCode
        adopt(AgentSession(config: AgentConfig(name: "Sai", privacy: .fullyPrivate, harness: privateHarness,
                                               provider: .ollama, ollamaModel: ollamaModel, openRouterModel: openRouterModel)))
        // A cloud agent too, if cloud settings were in use.
        let cloudProvider = (defaults.string(forKey: "provider.cloud") ?? defaults.string(forKey: "provider"))
            .flatMap(ModelProvider.init(rawValue:))
        let cloudHarness = harness("harness.cloud") ?? (harness("harness") == .sai ? .sai : nil)
        if cloudProvider == .openRouter || cloudHarness != nil {
            adopt(AgentSession(config: AgentConfig(name: "Sai", privacy: .remote, harness: cloudHarness ?? .claudeCode,
                                                   provider: cloudProvider ?? .openRouter, ollamaModel: ollamaModel,
                                                   openRouterModel: openRouterModel,
                                                   saiMachine: defaults.string(forKey: "saiMachine") ?? "",
                                                   saiModel: defaults.string(forKey: "saiModel") ?? "auto")))
        }
        saveAgents()
    }

    init() {
        vm = LinuxVM(files: files)
        phase = files.isPrepared ? .stopped : .needsSetup
        selectedAgentID = UUID()
        needsFirstAgentSetup = isFreshInstallation
        loadAgents()
        let saved = (UserDefaults.standard.string(forKey: "selectedAgent")
            ?? UserDefaults(suiteName: GuestFiles.legacy[0].0)?.string(forKey: "selectedAgent"))
            .flatMap(UUID.init(uuidString:))
        if let pending = try? String(contentsOf: firstAgentPreparationFile, encoding: .utf8),
           let id = UUID(uuidString: pending),
           agents.contains(where: { $0.id == id && $0.config.usesLocalComputer }) {
            selectedAgentID = id
            needsFirstAgentPreparation = true
        } else {
            selectedAgentID = agents.first { $0.id == saved }?.id ?? agents[0].id
        }
        vm.onStateChange = { [weak self] state in self?.vmChanged(state) }
    }

    var shell: GuestShell { GuestShell(files: files, mac: vm.macAddress, forwardedSSHPort: vm.forwardedSSHPort) }

    var ollamaReady: Bool {
        guard let ollamaVersion else { return false }
        return Ollama.isAtLeast(ollamaVersion, Self.minimumOllama)
    }

    static func minimumOllamaVersion(for name: String) -> String {
        Ollama.normalizedName(name) == Ollama.nemotronBoltE4B ? "0.40.10" : minimumOllama
    }

    func ollamaUpgradeMessage(for name: String) -> String? {
        guard let ollamaVersion else { return nil }
        let minimum = Self.minimumOllamaVersion(for: name)
        guard !Ollama.isAtLeast(ollamaVersion, minimum) else { return nil }
        return "Update Ollama to \(minimum) or later to use \(Ollama.displayName(for: name))."
    }

    func ollamaReady(for name: String) -> Bool {
        ollamaVersion != nil && ollamaUpgradeMessage(for: name) == nil
    }

    var hasAgentModel: Bool { hasOllamaModel(agentModel) }

    func hasOllamaModel(_ name: String) -> Bool { ollamaModels.contains { $0.matches(name) } }

    var isDownloading: Bool {
        switch ollamaDownload {
        case .downloading, .verifying: true
        default: false
        }
    }

    var hasOllamaAgents: Bool {
        agents.contains { $0.config.harness != .sai && $0.config.provider == .ollama }
    }

    func memoryRoute(for session: AgentSession) -> AgentMemoryRoute {
        let config = session.config.enforcingPrivacy
        let inference = ollamaInference(for: config.ollamaModel)
        let model = config.harness == .sai ? config.saiModel : (config.provider == .ollama ? config.ollamaModel : config.openRouterModel)
        return AgentMemoryRoute(privacy: config.privacy, harness: config.harness, provider: config.provider,
                                computer: config.computer, boxName: config.boxName, saiMachine: config.saiMachine,
                                ollamaInference: inference, model: model)
    }

    func openWork(for session: AgentSession) -> [AgentMemoryNote] {
        session.memory.notes(for: memoryRoute(for: session)).filter { $0.kind == .activeWork && $0.status == .open }
    }

    private func ollamaInference(for name: String) -> AgentMemoryRoute.OllamaInference {
        let model = ollamaModels.first { $0.matches(name) }
        if model?.usesRemoteInference == true || Ollama.isCloudReference(name) { return .hosted }
        return model != nil && ollamaReady ? .local : .unconfirmed
    }

    private func memoryRouteVerifier(for provider: ModelProvider) -> ((AgentMemoryRoute) async -> AgentMemoryRoute)? {
        guard provider == .ollama else { return nil }
        return { route in await Self.verifyOllamaMemoryRoute(route) }
    }

    private nonisolated static func verifyOllamaMemoryRoute(_ route: AgentMemoryRoute) async -> AgentMemoryRoute {
        guard route.provider == .ollama, let model = route.model else { return route }
        let matches = await Ollama.models(timeout: 2).filter { $0.matches(model) }
        let inference: AgentMemoryRoute.OllamaInference
        if Ollama.isCloudReference(model) || matches.contains(where: \.usesRemoteInference) {
            inference = .hosted
        } else {
            inference = matches.isEmpty ? .unconfirmed : .local
        }
        return AgentMemoryRoute(privacy: route.privacy, harness: route.harness, provider: route.provider,
                                computer: route.computer, boxName: route.target, ollamaInference: inference, model: model)
    }

    /// Use the selected model's metadata so Ollama cloud models are identified as hosted.
    func sidebarInference(for config: AgentConfig) -> (label: String, symbol: String, help: String) {
        if config.harness == .sai { return ("Hosted inference", "cloud", "Sai") }

        switch config.provider {
        case .openRouter: return ("Hosted inference", "cloud", "OpenRouter · \(config.openRouterModel)")
        case .claudeAccount, .codexAccount:
            let confirmed = accountStatuses[AccountRoute(config)] == .connected(.subscription)
            let provider = config.provider == .claudeAccount ? "Claude Code account" : "Codex account"
            return confirmed ? ("Subscription", "creditcard", provider) : ("Account", "person.crop.circle", provider)
        case .ollama:
            let provider = "Ollama · \(config.ollamaModel)"
            switch ollamaInference(for: config.ollamaModel) {
            case .hosted: return ("Hosted inference", "cloud", provider)
            case .local: return ("Local inference", "laptopcomputer", provider)
            case .unconfirmed: return ("Inference location unconfirmed", "questionmark.circle", provider)
            }
        }
    }

    /// The models Ollama has, without the `:latest` tag, plus the current choice if it isn't
    /// downloaded yet (so it can still be picked and downloaded).
    var modelChoices: [String] {
        let selected = Ollama.normalizedName(agentModel)
        var names = ollamaModels.map {
            let name = Ollama.normalizedName($0.name)
            return name == selected ? agentModel : name
        }
        if !names.contains(agentModel) { names.insert(agentModel, at: 0) }
        return names
    }

    func ollamaInputCapability(for name: String) -> ModelInputCapability? {
        if let installed = ollamaModels.first(where: { $0.matches(name) }) {
            return installedOllamaInputs[installed]
        }
        return libraryOllamaInputs[Ollama.normalizedName(name)] ?? Ollama.suggestedModelInput(for: name)
    }

    func rememberOllamaInput(_ capability: ModelInputCapability?, for name: String) {
        guard let capability else { return }
        let normalized = Ollama.normalizedName(name)
        if libraryOllamaInputs[normalized] != capability { libraryOllamaInputs[normalized] = capability }
    }

    func ollamaOptionTitle(_ name: String) -> String {
        let title = Ollama.displayName(for: name)
        return ollamaInputCapability(for: name).map { "\(title) · \($0.label)" } ?? title
    }

    func openRouterOptionTitle(_ id: String) -> String {
        let input = openRouterModels.first { $0.id == id }?.inputCapability?.label
        let zdr = openRouterZDRAvailability(for: id).label
        return [id, input, zdr].compactMap { $0 }.joined(separator: " · ")
    }

    // MARK: Actions

    func prepare() async {
        guard !needsFirstAgentSetup, !isSavingSandbox, !files.isPrepared, vm.machine == nil else { return }
        if case .preparing = phase { return }
        if case .booting = phase { return }
        phase = .preparing(.working("Preparing the sandbox…"))
        do {
            try await GuestImage.prepare(files: files) { [weak self] step in
                self?.phase = .preparing(step)
            }
            phase = .stopped
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func prepareAndStartSandbox() async {
        guard !needsFirstAgentSetup, !Task.isCancelled else { return }
        if !files.isPrepared {
            await prepare()
            guard phase == .stopped else { return }
        }
        guard !Task.isCancelled else { return }
        switch phase {
        case .stopped, .failed: await start()
        default: break
        }
    }

    func resumeSavedSandboxOnLaunch() {
        guard !attemptedSandboxRestore else { return }
        attemptedSandboxRestore = true
        guard hasSavedSandbox, !GuestFiles.legacyAppRunning else { return }
        Task { await start() }
    }

    func start(restoreSavedState: Bool = true) async {
        guard !isSavingSandbox, sandboxStartup == nil, vm.state != .stopping else { return }
        if case .preparing = phase { return }

        let task = Task { await startSandbox(restoreSavedState: restoreSavedState) }
        sandboxStartup = task
        await task.value
    }

    private func startSandbox(restoreSavedState: Bool) async {
        defer { if !Task.isCancelled { sandboxStartup = nil } }
        guard !Task.isCancelled else { return }
        let resuming: Bool
        if vm.state == .running {
            resuming = sandboxCanCheckpoint
        } else {
            guard vm.machine == nil else { return }
            let restoring = restoreSavedState && hasSavedSandbox
            phase = .booting(restoring ? "Resuming sandbox…" : "Starting the computer…")
            resuming = await vm.start(memory: computerMemory, networkMode: sandboxNetworkMode,
                                      restoreSavedState: restoreSavedState)
            guard !Task.isCancelled, vm.state == .running else { return }
            sandboxCanCheckpoint = resuming
        }
        guard vm.state == .running else { return }

        let shell = shell
        phase = .booting(resuming ? "Reconnecting to the sandbox…" : "Waiting for the computer to come online…")
        guard let ip = await shell.waitUntilReachable() else {
            if !Task.isCancelled, vm.state == .running {
                phase = .failed("The computer started but never answered on its network.")
            }
            return
        }
        guard !Task.isCancelled, vm.state == .running else { return }
        guestIP = ip

        let install: (command: String, step: String)?
        do {
            if resuming, !sandboxNeedsFullToolInstall, try await shell.isReady(on: ip) {
                guard !Task.isCancelled, vm.state == .running else { return }
                phase = .booting("Checking the sandbox tools…")
                do {
                    try await shell.run(GuestTools.deepSeekReadyCommand, on: ip, timeout: 20)
                    install = nil
                } catch let failure as GuestShell.CommandFailure where failure.status == GuestTools.deepSeekUpdateRequiredStatus {
                    install = (GuestTools.installDeepSeekCommand, "Updating DeepSeek Harness in the sandbox…")
                }
            } else {
                guard !Task.isCancelled, vm.state == .running else { return }
                phase = .booting("Waiting for the sandbox’s first-time setup…")
                let clock = ContinuousClock()
                let deadline = clock.now.advanced(by: .seconds(600))
                while (try? await shell.isReady(on: ip)) != true {
                    guard !Task.isCancelled, vm.state == .running else { return }
                    guard clock.now < deadline else {
                        phase = .failed("The sandbox’s first-time setup is taking longer than expected. Try again to check its progress.")
                        return
                    }
                    try await Task.sleep(for: .seconds(3))
                }
                guard !Task.isCancelled, vm.state == .running else { return }
                sandboxNeedsFullToolInstall = true
                let minutes = (GuestTools.installTimeoutSeconds + 59) / 60
                install = (GuestTools.installCommand, "Installing the sandbox tools… Allow about \(minutes) minutes")
            }
        } catch {
            guard !Task.isCancelled, vm.state == .running else { return }
            phase = .failed(GuestShell.commandTimedOut(error)
                            ? "Checking the sandbox tools timed out."
                            : "Couldn’t verify the sandbox tools.")
            return
        }

        guard !Task.isCancelled, vm.state == .running else { return }
        if let install {
            phase = .booting(install.step)
            do {
                // Give the guest time to stop the installer and report its exit status.
                let timeout = TimeInterval(GuestTools.installTimeoutSeconds + 15)
                try await shell.run(install.command, on: ip, timeout: timeout)
            } catch {
                guard !Task.isCancelled, vm.state == .running else { return }
                NSLog("Agent S: installing the sandbox tools failed: %@", "\(error)")
                phase = .failed(GuestShell.commandTimedOut(error)
                                ? "Installing the sandbox tools timed out. Check your internet connection and try again."
                                : "Couldn’t install the sandbox tools. Try again to finish setup.")
                return
            }
        }
        guard !Task.isCancelled, vm.state == .running else { return }
        sandboxNeedsFullToolInstall = false
        sandboxCanCheckpoint = true
        phase = .ready
    }

    func stop() async {
        guard !isSavingSandbox else { return }
        cancelSandboxStartup()
        agents.forEach { $0.stop() }
        await vm.stop()
    }

    func prepareToQuit() async -> Bool {
        guard vm.machine != nil || vm.state == .starting || vm.state == .stopping else {
            stopAppWork()
            return true
        }

        isSavingSandbox = true
        for (route, task) in accountSignIns where route.inSandbox { task.cancel() }
        agents.filter { $0.config.usesLocalComputer }.forEach { $0.stop() }

        do {
            // Let setup and active guest work finish before pausing the VM.
            let clock = ContinuousClock()
            let deadline = clock.now.advanced(by: .seconds(5))
            while sandboxIsBusyForSave {
                guard clock.now < deadline else { throw SandboxSaveFailure.stillBusy }
                try await Task.sleep(for: .milliseconds(100))
            }
            if sandboxCanCheckpoint, vm.supportsCheckpoint {
                try await vm.saveForQuit()
            } else if !(await vm.stop()) {
                isSavingSandbox = false
                return false
            }
        } catch {
            let alert = NSAlert()
            alert.alertStyle = .warning
            alert.messageText = "Couldn’t save the sandbox"
            alert.informativeText = "\(error.localizedDescription) Shutting down closes its open apps and windows."
            alert.addButton(withTitle: "Keep Agent S Open")
            alert.addButton(withTitle: "Shut Down and Quit")
            guard alert.runModal() == .alertSecondButtonReturn else {
                isSavingSandbox = false
                return false
            }
            guard await vm.stop() else {
                isSavingSandbox = false
                return false
            }
        }

        stopAppWork()
        return true
    }

    var hasWorkToFinishBeforeUpdate: Bool {
        if case .preparing = phase { return true }
        return isSavingSandbox || sandboxIsBusyForSave || agents.contains(where: \.isRunning)
    }

    private var sandboxIsBusyForSave: Bool {
        if case .booting = phase { return true }
        return vm.state == .starting || vm.state == .stopping || sandboxHasActiveTurn
            || accountSignIns.keys.contains(where: \.inSandbox)
    }

    private enum SandboxSaveFailure: LocalizedError {
        case stillBusy

        var errorDescription: String? { "Sandbox work is still in progress." }
    }

    func stopAppWork() {
        macTurnPanel.suppressAutomaticChatRestoration()
        cancelSandboxStartup()
        accountSignIns.values.forEach { $0.cancel() }
        agents.forEach { $0.stop() }
        DeepSeekRuntime.stopAll()
    }

    func refreshOllama(timeout: TimeInterval = 10) async {
        _ = await refreshOllamaState(timeout: timeout)
    }

    private func refreshOllamaState(timeout: TimeInterval) async -> [Ollama.Model] {
        ollamaRefreshGeneration += 1
        let generation = ollamaRefreshGeneration
        async let version = Ollama.version(timeout: timeout)
        async let models = Ollama.models(timeout: timeout)
        let (checkedVersion, checkedModels) = await (version, models)
        if !Task.isCancelled, generation > appliedOllamaRefreshGeneration {
            appliedOllamaRefreshGeneration = generation
            ollamaVersion = checkedVersion
            if ollamaModels != checkedModels {
                ollamaModels = checkedModels
                refreshInstalledOllamaInputs()
            }
            ollamaChecked = true
        }
        return checkedModels
    }

    private func refreshInstalledOllamaInputs() {
        ollamaInputsTask?.cancel()
        let local = ollamaModels.filter { !$0.usesRemoteInference && !Ollama.isCloudReference($0.name) }
        let current = Set(local)
        installedOllamaInputs = installedOllamaInputs.filter { current.contains($0.key) }
        let missing = local.filter { installedOllamaInputs[$0] == nil }
        ollamaInputsTask = Task { [weak self] in
            for installed in missing {
                guard !Task.isCancelled else { return }
                let capability = await Ollama.inputCapability(for: installed.name)
                guard !Task.isCancelled else { return }
                if let capability { self?.installedOllamaInputs[installed] = capability }
            }
        }
    }

    func downloadOllamaModel(_ name: String) async {
        guard !Task.isCancelled, !isDownloading else { return }
        guard ollamaReady(for: name) else {
            let message = ollamaUpgradeMessage(for: name) ?? "Open Ollama and check the connection before downloading a model."
            ollamaDownload = .failed(name: name, message: message)
            return
        }
        ollamaDownload = .downloading(name: name, progress: nil)
        do {
            try await Ollama.pull(name) { [weak self] _, fraction in
                self?.ollamaDownload = .downloading(name: name, progress: fraction)
            }
            ollamaDownload = .verifying(name: name)
            let downloadedModels = await refreshOllamaState(timeout: 10)
            try Task.checkCancellation()
            if downloadedModels.contains(where: { $0.matches(name) }) {
                ollamaDownload = nil
            } else {
                ollamaDownload = .failed(name: name, message: "Ollama didn’t list \(name) after the download. Refresh the list or try again.")
            }
        } catch {
            ollamaDownload = Task.isCancelled ? nil : .failed(name: name, message: error.localizedDescription)
        }
    }

    /// The site open in the computer's browser (from its address bar), e.g. "x.com".
    func currentSite() async -> String? {
        guard let ip = guestIP,
              let line = try? await shell.run("ui tree 2>/dev/null | grep -m1 'Address and search bar'", on: ip),
              let value = line.split(separator: "\"").dropFirst(3).first
        else { return nil }
        return OnePassword.host(of: String(value))
    }

    /// Types into whatever is focused in the computer. The text goes over stdin, so it never
    /// appears in a command line, the chat or the agent's context.
    func typeIntoComputer(_ text: String, thenTab: Bool = false) async throws {
        guard let ip = guestIP, !isSavingSandbox else { throw OnePassword.Failure(message: "The computer is unavailable.") }
        try await shell.run("DISPLAY=:0 xdotool type --delay 12 --file -", on: ip, input: text)
        if thenTab { try await shell.run("DISPLAY=:0 xdotool key Tab", on: ip) }
    }

    // MARK: Model provider

    /// The model the next message runs on, for the chosen harness and provider.
    var activeModel: String { activeModel(for: agent.config) }

    private func activeModel(for config: AgentConfig) -> String {
        if config.harness == .sai { return config.saiModel }
        if config.provider.usesAccount { return "Automatic" }
        return config.provider == .ollama ? config.ollamaModel : config.openRouterModel
    }

    /// Whether the selected computer, harness, and model are ready for a message.
    var canChat: Bool { chatBlocker == nil }

    var usesOllama: Bool { harness != .sai && provider == .ollama }

    var shouldCheckOllamaForChat: Bool { usesOllama && computerChatBlocker(for: agent) == nil }

    private func modelBlocker(for config: AgentConfig) -> String? {
        let provider = config.provider
        if provider.usesAccount {
            let status = accountStatus(for: config)
            if !status.isConnected {
                let name = provider == .codexAccount ? "Codex" : "Claude Code"
                return status == .checking ? "Checking sign-in…" : "Connect your \(name) account to chat"
            }
        }
        if provider == .ollama {
            if !ollamaChecked { return "Checking Ollama…" }
            if let message = ollamaUpgradeMessage(for: config.ollamaModel) { return message }
            if !hasOllamaModel(config.ollamaModel) {
                return ollamaReady ? "Download \(Ollama.displayName(for: config.ollamaModel)) to chat" : "Connect Ollama to chat"
            }
        }
        if provider == .openRouter {
            if !hasOpenRouterKey { return "Add your OpenRouter key in Settings first" }
            if config.openRouterModel.isEmpty { return "Choose an OpenRouter model in Settings first" }
        }
        return nil
    }

    /// Why a message can't be sent, for the composer.
    var chatBlocker: String? { chatBlocker(for: agent) }

    private func chatBlocker(for session: AgentSession) -> String? {
        computerChatBlocker(for: session) ?? (session.config.harness == .sai ? nil : modelBlocker(for: session.config))
    }

    private func computerChatBlocker(for session: AgentSession) -> String? {
        let config = session.config
        if config.privacy == .thisMac {
            switch config.harness {
            case .codex where CodexAccount.executable == nil:
                return "Install Codex on this Mac to use this agent"
            case .claudeCode where BoxAccount.claude == nil:
                return "Install Claude Code on this Mac to use this agent"
            case .deepSeek where !DeepSeekTurn.isAvailable:
                return "Install DeepSeek Harness on this Mac to use this agent"
            default: break
            }
            if MacAccess.simulang == nil { return "Install simulang first: npm install -g @simular-ai/simulang" }
            if !macAccessibility { return "Allow Accessibility in Settings to use this Mac" }
            return nil
        }
        if config.computer == .box {
            if !hasBoxKey { return "Connect Simular Box in Settings" }
            switch config.harness {
            case .claudeCode where BoxAccount.claude == nil: return "Install Claude Code on this Mac"
            case .codex where CodexAccount.executable == nil: return "Install Codex on this Mac"
            case .deepSeek where !DeepSeekTurn.isAvailable:
                return "Install DeepSeek Harness on this Mac"
            default: break
            }
            return nil
        }
        if config.harness == .sai {
            if !hasSaiKey { return "Add your Sai API key in Settings" }
            if config.saiMachine == SaiAPI.localMachine { return "This Mac's computer isn't available to Sai yet: pick a Sai computer" }
            if config.saiMachine.isEmpty { return saiMachines.isEmpty ? "Finding your Sai computers…" : "Choose a Sai computer above" }
            return nil
        }
        if isSavingSandbox { return "Saving the sandbox…" }
        switch phase {
        case .needsSetup: return "Set up the computer in Settings to chat"
        case .preparing, .booting, .stopping: return "Waiting for the sandbox…"
        case .failed where vm.state == .running: return "Finish sandbox setup to chat"
        case .ready:
            if agents.contains(where: { $0 !== session && $0.config.usesLocalComputer && $0.isRunning }) {
                return "Another agent is using the sandbox"
            }
            return nil
        default: return "Turn the computer on to chat"
        }
    }

    var selectedSaiComputer: SaiAPI.Machine? {
        saiMachines.first { $0.id == saiMachine }
    }

    var saiMachineName: String {
        selectedSaiComputer?.name ?? (saiMachine.isEmpty ? "No computer" : saiMachine)
    }

    var saiScreenState: SaiAPI.ScreenState {
        saiScreenStates[saiMachine] ?? selectedSaiComputer?.screenState ?? .unknown
    }

    func noteSaiScreenState(_ state: SaiAPI.ScreenState, for machine: String) {
        guard hasSaiKey, !machine.isEmpty, machine != SaiAPI.localMachine else { return }
        let observation: SaiAPI.ScreenState? = state == .unknown ? nil : state
        guard saiScreenStates[machine] != observation else { return }
        saiScreenStates[machine] = observation
    }

    func saveSaiKey(_ key: String) async -> Bool {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, SaiAPI.save(key: trimmed) else { return false }
        resetSaiKeyState(hasKey: true)
        await refreshSai()
        return true
    }

    func removeSaiKey() {
        SaiAPI.delete()
        resetSaiKeyState(hasKey: false)
    }

    private func resetSaiKeyState(hasKey: Bool) {
        hasSaiKey = hasKey
        saiKeyRevision += 1
        saiMachines = []
        saiModels = []
        saiStatus = nil
        saiScreenStates = [:]
    }

    /// Loads the Sai computers and models; picks the first online computer if none is chosen.
    func refreshSai() async {
        let revision = saiKeyRevision
        let session = agent
        guard await loadSaiComputers() else { return }
        let models = (try? await SaiAPI.models()) ?? []
        guard !Task.isCancelled, revision == saiKeyRevision else { return }
        saiModels = models
        guard session.config.harness == .sai, !session.isRunning, agents.contains(where: { $0 === session }) else { return }
        let machine = session.config.saiMachine
        if machine.isEmpty || (machine != SaiAPI.localMachine && !saiMachines.contains { $0.id == machine }) {
            session.config.saiMachine = (saiMachines.first { $0.online } ?? saiMachines.first)?.id ?? ""
        }
    }

    /// Refreshes status for the live screen without changing any agent's computer.
    func refreshSaiComputers() async {
        _ = await loadSaiComputers()
    }

    private func loadSaiComputers() async -> Bool {
        guard hasSaiKey, !Task.isCancelled else { return false }
        let revision = saiKeyRevision
        do {
            let machines = try await SaiAPI.machines()
            guard !Task.isCancelled, revision == saiKeyRevision else { return false }
            saiMachines = machines
            saiStatus = nil
            return true
        } catch {
            guard !Task.isCancelled, revision == saiKeyRevision else { return false }
            saiStatus = error.localizedDescription
            return false
        }
    }

    func saveOpenRouterKey(_ key: String) -> Bool {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, OpenRouter.save(key: trimmed) else { return false }
        hasOpenRouterKey = true
        return true
    }

    func removeOpenRouterKey() {
        OpenRouter.delete()
        hasOpenRouterKey = false
    }

    var openRouterZDRNeedsRefresh: Bool {
        guard let check = openRouterZDRCheck else { return true }
        return !check.isFresh || check.publicModelIDs == nil
    }

    func openRouterZDRAvailability(for id: String) -> OpenRouterZDRAvailability {
        guard let check = openRouterZDRCheck, check.isFresh else { return .unknown }
        return check.catalog.availability(for: id, isPublicModel: check.publicModelIDs?.contains(id) == true)
    }

    func refreshOpenRouterModels(force: Bool = false) async {
        guard !isRefreshingOpenRouterModels, force || openRouterModels.isEmpty || openRouterZDRNeedsRefresh else { return }
        isRefreshingOpenRouterModels = true
        defer { isRefreshingOpenRouterModels = false }

        async let models = OpenRouter.models()
        async let zdrCatalog = OpenRouter.zdrCatalog()
        let (loadedModels, loadedZDRCatalog) = await (models, zdrCatalog)
        guard !Task.isCancelled else { return }

        if let loadedModels { openRouterModels = loadedModels }
        openRouterZDRCheck = loadedZDRCatalog.map { catalog in
            OpenRouterZDRCheck(catalog: catalog, publicModelIDs: loadedModels.map { Set($0.map(\.id)) })
        }
    }

    // MARK: Harness accounts

    var accountStatus: AccountStatus { accountStatus(for: agent.config) }

    private func accountStatus(for config: AgentConfig) -> AccountStatus {
        accountStatuses[AccountRoute(config)] ?? .unchecked
    }

    var accountComputerReady: Bool {
        !privacy.usesLocalComputer || (phase == .ready && guestIP != nil && !isSavingSandbox)
    }

    var sandboxHasActiveTurn: Bool {
        agents.contains { $0.config.usesLocalComputer && $0.isRunning }
    }

    private struct AccountFailure: LocalizedError {
        let errorDescription: String?
    }

    private func accountPlan(_ route: AccountRoute) async throws -> HarnessAccountPlan? {
        let provider = route.provider
        if route.inSandbox {
            guard let ip = guestIP, phase == .ready else {
                throw AccountFailure(errorDescription: "Turn on the sandbox to check sign-in.")
            }
            let output = try await shell.run(SandboxAccount.statusCommand(for: provider), on: ip)
            return SandboxAccount.plan(for: provider, output: output)
        }
        if provider == .codexAccount { return try await CodexAccount.accountPlan(dataFolder: files.root) }
        guard let claude = BoxAccount.claude else {
            throw AccountFailure(errorDescription: "Install Claude Code on this Mac first.")
        }
        return await ClaudeAccount.accountPlan(claude: claude, profile: ClaudeAccount.profile(in: files.root))
    }

    func refreshAccountStatus() async {
        let route = AccountRoute(agent.config)
        guard availableProviders.contains(route.provider), route.provider.usesAccount, accountComputerReady else { return }
        if let pending = accountSignIns[route] {
            await confirmClaudeBrowserSignIn(route, pending: pending)
            return
        }
        let check = UUID()
        let previous = accountStatuses[route] ?? .unchecked
        accountChecks[route] = check
        if !previous.isConnected, previous != .awaitingSignIn { accountStatuses[route] = .checking }
        let status: AccountStatus
        do {
            let plan = try await accountPlan(route)
            status = plan.map(AccountStatus.connected) ?? (previous == .awaitingSignIn ? .awaitingSignIn : .signedOut)
        } catch {
            status = .failed(error.localizedDescription)
        }
        guard accountChecks[route] == check else { return }
        accountChecks[route] = nil
        accountStatuses[route] = Task.isCancelled ? (previous == .checking ? .unchecked : previous) : status
    }

    private func confirmClaudeBrowserSignIn(_ route: AccountRoute, pending: Task<Void, Never>) async {
        guard route.inSandbox, route.provider == .claudeAccount, let ip = guestIP,
              case .awaitingBrowserApproval = accountStatuses[route], accountChecks[route] == nil else { return }
        let check = UUID()
        let generation = sandboxAccountGeneration
        accountChecks[route] = check
        defer {
            if accountChecks[route] == check { accountChecks[route] = nil }
        }

        // The guest verifies the saved account only after its login command releases the profile lock.
        let output = try? await shell.run(SandboxAccount.claudeCompletedSignInStatusCommand, on: ip, timeout: 10)
        guard !Task.isCancelled, !pending.isCancelled, accountChecks[route] == check, generation == sandboxAccountGeneration,
              case .awaitingBrowserApproval = accountStatuses[route],
              let output, let plan = SandboxAccount.plan(for: route.provider, output: output) else { return }
        accountStatuses[route] = .connected(plan)
        pending.cancel()
    }

    func signInToAccount() async {
        let route = AccountRoute(agent.config)
        let provider = route.provider
        guard availableProviders.contains(provider), provider.usesAccount,
              accountComputerReady, accountSignIns[route] == nil else { return }
        if route.inSandbox, sandboxHasActiveTurn { return }
        let generation = sandboxAccountGeneration
        accountChecks[route] = nil
        accountStatuses[route] = .connecting
        let task = Task { @MainActor in
            defer { accountSignIns[route] = nil }
            do {
                if route.inSandbox {
                    try await signInToSandbox(route, generation: generation)
                } else if provider == .codexAccount {
                    let plan = try await CodexAccount.signIn(dataFolder: files.root)
                    try Task.checkCancellation()
                    accountStatuses[route] = .connected(plan)
                } else {
                    guard let claude = BoxAccount.claude else {
                        throw AccountFailure(errorDescription: "Install Claude Code on this Mac first.")
                    }
                    try await ClaudeAccount.signIn(claude: claude, profile: ClaudeAccount.profile(in: files.root))
                    accountStatuses[route] = .awaitingSignIn
                }
            } catch {
                guard !route.inSandbox || sandboxAccountGeneration == generation else { return }
                guard accountStatuses[route]?.isConnected != true else { return }
                accountStatuses[route] = Task.isCancelled || error is CancellationError
                    ? .signedOut : .failed(error.localizedDescription)
            }
        }
        accountSignIns[route] = task
        await task.value
    }

    private func signInToSandbox(_ route: AccountRoute, generation: UUID) async throws {
        try checkSandboxSignIn(generation)
        guard let ip = guestIP else { throw AccountFailure(errorDescription: "Turn on the sandbox to sign in.") }
        if route.provider == .claudeAccount {
            let plan = try await ClaudeAccount.signInInSandbox(shell: shell, ip: ip) { url in
                try self.checkSandboxSignIn(generation)
                self.accountStatuses[route] = .awaitingBrowserApproval(url)
                if AccountRoute(self.agent.config) == route {
                    NSWorkspace.shared.open(url)
                }
            }
            try checkSandboxSignIn(generation)
            accountStatuses[route] = .connected(plan)
            return
        }
        if route.provider == .codexAccount {
            do {
                let plan = try await CodexAccount.signInInSandbox(shell: shell, ip: ip) { code in
                    try self.checkSandboxSignIn(generation)
                    self.accountStatuses[route] = .awaitingDeviceCode(code)
                    if AccountRoute(self.agent.config) == route {
                        NSWorkspace.shared.open(code.verificationURL)
                    }
                }
                try checkSandboxSignIn(generation)
                accountStatuses[route] = .connected(plan)
                return
            } catch CodexAccount.DeviceSignInError.unsupported {
                try checkSandboxSignIn(generation)
            }
        }

        try await shell.run(SandboxAccount.signInCommand(for: route.provider), on: ip)
        try checkSandboxSignIn(generation)
        accountStatuses[route] = .awaitingSignIn
        if AccountRoute(agent.config) == route { sandboxSignInRequest = UUID() }
    }

    private func checkSandboxSignIn(_ generation: UUID) throws {
        try Task.checkCancellation()
        guard sandboxAccountGeneration == generation, vm.state == .running else { throw CancellationError() }
    }

    func cancelAccountSignIn() {
        let route = AccountRoute(agent.config)
        guard let pending = accountSignIns[route] else { return }
        accountChecks[route] = nil
        pending.cancel()
    }

    private var checkingFollowUpRoutes: Set<UUID> = []

    @discardableResult
    func queueFollowUp(_ prompt: String, for session: AgentSession) -> Bool {
        guard agents.contains(where: { $0 === session }) else { return false }
        return session.enqueueFollowUp(prompt, route: memoryRoute(for: session))
    }

    func canForceQueuedFollowUp(_ followUp: AgentSession.QueuedFollowUp, for session: AgentSession) -> Bool {
        agents.contains(where: { $0 === session }) && session.isRunning
            && !session.isStopping && !session.isSendingSteering
            && session.config == session.config.enforcingPrivacy && followUp.route == memoryRoute(for: session)
    }

    func forceQueuedFollowUp(_ followUp: AgentSession.QueuedFollowUp, for session: AgentSession) {
        guard canForceQueuedFollowUp(followUp, for: session) else { return }
        session.forceFollowUp(followUp)
    }

    func canSendQueuedFollowUp(for session: AgentSession) -> Bool {
        guard let followUp = session.queuedFollowUps.first else { return false }
        return canDispatchFollowUp(followUp, to: session)
    }

    private func canDispatchFollowUp(_ followUp: AgentSession.QueuedFollowUp, to session: AgentSession) -> Bool {
        !checkingFollowUpRoutes.contains(followUp.id) && canDispatch(to: session)
            && session.config == session.config.enforcingPrivacy && followUp.route == memoryRoute(for: session)
    }

    func sendQueuedFollowUp(for session: AgentSession, forcedID: UUID? = nil) {
        let next = session.queuedFollowUps.first { forcedID == nil || $0.id == forcedID }
        guard let followUp = next, !checkingFollowUpRoutes.contains(followUp.id) else { return }
        guard canDispatchFollowUp(followUp, to: session) else { session.pauseQueuedFollowUps(); return }
        if forcedID == nil { session.resumeQueuedFollowUps() }
        guard followUp.route.provider == .ollama else {
            dispatchFollowUp(followUp, to: session)
            return
        }

        // A model tag can change from local to hosted while the follow-up is waiting.
        checkingFollowUpRoutes.insert(followUp.id)
        Task { @MainActor [weak self, weak session] in
            let verified = await Self.verifyOllamaMemoryRoute(followUp.route)
            guard let self else { return }
            self.checkingFollowUpRoutes.remove(followUp.id)
            guard let session, !session.isRunning, forcedID != nil || !session.followUpsPaused else { return }
            let pending = session.queuedFollowUps.first { forcedID == nil || $0.id == forcedID }
            guard pending?.id == followUp.id else {
                if forcedID == nil { self.sendQueuedFollowUp(for: session) }
                return
            }
            guard verified == followUp.route, self.canDispatchFollowUp(followUp, to: session) else {
                session.pauseQueuedFollowUps()
                return
            }
            self.dispatchFollowUp(followUp, to: session, verifiedRoute: verified)
        }
    }

    private func dispatchFollowUp(_ followUp: AgentSession.QueuedFollowUp, to session: AgentSession,
                                  verifiedRoute: AgentMemoryRoute? = nil) {
        if send(followUp.text, to: session, verifiedRoute: verifiedRoute) { session.removeQueuedFollowUp(followUp.id) }
        else { session.pauseQueuedFollowUps() }
    }

    private func canDispatch(to session: AgentSession) -> Bool {
        agents.contains(where: { $0 === session }) && !session.isRunning && !session.isSendingSteering
            && chatBlocker(for: session) == nil
    }

    func canRunOpenWork(for session: AgentSession) -> Bool {
        canDispatch(to: session) && memoryRoute(for: session).allowsAutomaticRecall
    }

    func runOpenWork(ids: Set<UUID>, for session: AgentSession) async -> Bool {
        guard !ids.isEmpty, selectedAgentID == session.id, canRunOpenWork(for: session) else { return false }
        let route = memoryRoute(for: session)
        let selected = openWork(for: session).filter { ids.contains($0.id) }
        guard selected.count == ids.count else { return false }
        if route.provider == .ollama {
            guard await Self.verifyOllamaMemoryRoute(route) == route else { return false }
        }
        guard !Task.isCancelled, selectedAgentID == session.id, canRunOpenWork(for: session),
              memoryRoute(for: session) == route,
              openWork(for: session).filter({ ids.contains($0.id) }) == selected else { return false }
        let list = selected.map { "- " + $0.text.replacingOccurrences(of: "\n", with: "\n  ") }.joined(separator: "\n")
        return send("Work on the checked items below.\n\n\(list)", to: session, verifiedRoute: route)
    }

    @discardableResult
    func send(_ prompt: String, to session: AgentSession, verifiedRoute: AgentMemoryRoute? = nil) -> Bool {
        guard agents.contains(where: { $0 === session }), !session.isRunning, !session.isSendingSteering else { return false }
        // The private/cloud boundary is enforced here too, not only by what the settings offer.
        let config = session.config.enforcingPrivacy
        guard config == session.config else { session.config = config; return false }
        guard chatBlocker(for: session) == nil else { return false }
        let route = memoryRoute(for: session)
        guard verifiedRoute == nil || verifiedRoute == route else { return false }
        let verifyRoute = verifiedRoute == nil ? memoryRouteVerifier(for: config.provider) : nil
        let harness = config.harness
        if harness == .sai {
            session.sendToSai(prompt, machine: config.saiMachine, model: config.saiModel, memoryRoute: route)
            return true
        }
        if config.computer == .box {
            return sendToBox(prompt, to: session, memoryRoute: route)
        }
        if config.privacy == .thisMac {
            return sendToMac(prompt, to: session, memoryRoute: route, verifyRoute: verifyRoute)
        }
        guard config.usesLocalComputer, let ip = guestIP, let endpoint = localModelEndpoint(for: session) else { return false }
        let shell = shell
        let provider = config.provider
        let port: UInt16? = provider.usesAccount ? nil : (provider == .openRouter ? OpenRouter.proxyPort : Ollama.port)
        session.send(prompt, harness: harness, shell: shell, ip: ip, model: activeModel(for: config),
                     ollamaURL: endpoint.url, provider: provider, apiKey: provider == .openRouter ? endpoint.key : nil,
                     tunnelTo: port, memoryRoute: route, verifyRoute: verifyRoute) {
            await Self.screenNow(shell: shell, ip: ip)
        }
        return true
    }

    /// Box turns run the selected harness with only the remote Box's tool set.
    private func sendToBox(_ prompt: String, to session: AgentSession, memoryRoute: AgentMemoryRoute) -> Bool {
        guard let (baseURL, apiKey) = localModelEndpoint(for: session) else { return false }
        let config = session.config
        let model = activeModel(for: config)
        switch config.harness {
        case .claudeCode:
            guard let claude = BoxAccount.claude, let boxToken = BoxAccount.key() else { return false }
            let command = BoxClaudeCommand(prompt: prompt, resume: nil, model: model, provider: config.provider, apiKey: apiKey,
                                           boxName: config.boxName, baseURL: baseURL)
            session.sendOnBox(prompt, command: command, claude: claude, boxToken: boxToken, dataFolder: files.root, memoryRoute: memoryRoute)
        case .codex:
            session.sendWithCodex(prompt, dataFolder: files.root, model: model, baseURL: baseURL, apiKey: apiKey,
                                  memoryRoute: memoryRoute)
        case .deepSeek:
            session.sendWithDeepSeekOnBox(prompt, dataFolder: files.root, model: model, baseURL: baseURL, apiKey: apiKey,
                                         memoryRoute: memoryRoute)
        case .sai: return false // Sai runs through its own API on a Sai computer.
        }
        return true
    }

    // MARK: This Mac agents

    private let macTurnPanel = MacTurnPanelController()
    var openMainWindow: (() -> Void)?

    func showMacTurnPanel(for session: AgentSession, presentation: MacTurnPanelController.Presentation = .manual) {
        guard session.config.privacy == .thisMac else { return }
        let mainWindow = NSApp.mainWindow
        macTurnPanel.show(session, presentation: presentation) { [weak self, weak mainWindow] session in
            guard let self, self.agents.contains(where: { $0.id == session.id }) else { return }
            self.select(session)
            NSApp.unhide(nil)
            NSApp.activate()
            let windows = [mainWindow].compactMap { $0 } + NSApp.windows
            guard let window = windows.first(where: { $0.canBecomeMain && ($0.isVisible || $0.isMiniaturized) }) else {
                self.openMainWindow?()
                return
            }
            if window.isMiniaturized { window.deminiaturize(nil) }
            window.makeKeyAndOrderFront(nil)
        }
    }

    /// Listens for the approval hook; started with the first This Mac turn.
    let approvals = ApprovalServer()
    /// macOS grants these permissions to Agent S. Every This Mac agent shares the result.
    @Published private(set) var macAccessibility = MacAccess.hasAccessibility
    @Published private(set) var macScreenRecording = MacAccess.hasScreenRecording

    func refreshMacAccess() {
        let accessibility = MacAccess.hasAccessibility
        let screenRecording = MacAccess.hasScreenRecording
        if macAccessibility != accessibility { macAccessibility = accessibility }
        if macScreenRecording != screenRecording { macScreenRecording = screenRecording }
    }

    /// A This Mac agent's turn. The harness asks Agent S for approval before acting on the Mac.
    private func sendToMac(_ prompt: String, to session: AgentSession, memoryRoute: AgentMemoryRoute,
                           verifyRoute: ((AgentMemoryRoute) async -> AgentMemoryRoute)?) -> Bool {
        let config = session.config
        let model = activeModel(for: config)
        do {
            try MacAccess.install()
            if config.harness == .claudeCode { try approvals.start() }
        } catch {
            session.noticeFromApp("Couldn't get ready to work on this Mac: \(error.localizedDescription)")
            return false
        }
        if config.provider == .codexAccount {
            session.sendWithCodex(prompt, dataFolder: files.root, model: model, memoryRoute: memoryRoute)
            return true
        }
        guard let (baseURL, apiKey) = localModelEndpoint(for: session) else { return false }
        if config.harness == .deepSeek {
            session.sendWithDeepSeekOnMac(prompt, dataFolder: files.root, model: model, baseURL: baseURL, apiKey: apiKey,
                                         memoryRoute: memoryRoute, verifyRoute: verifyRoute)
            return true
        }
        guard let claude = BoxAccount.claude else { return false }
        approvals.decide = { [weak self] agentID, tool, input in
            guard let session = self?.agents.first(where: { $0.id.uuidString == agentID }),
                  session.config.privacy == .thisMac, !session.config.provider.usesAccount else { return false }
            return await session.requestApproval(tool: tool, input: input, macToolPath: MacAccess.macTool)
        }
        approvals.runCommand = { [weak self] agentID, turn, command in
            guard let session = self?.agents.first(where: { $0.id.uuidString == agentID }),
                  session.config.privacy == .thisMac, session.config.provider == .claudeAccount else {
                return ("This turn has ended.", false, nil)
            }
            return await session.runClaudeMacCommand(command, turn: turn)
        }
        let command = MacClaudeCommand(prompt: prompt, resume: nil, model: model, provider: config.provider, apiKey: apiKey,
                                       baseURL: baseURL, macToolPath: MacAccess.macTool)
        session.sendOnMac(prompt, command: command, claude: claude, dataFolder: files.root, settingsPath: MacAccess.settings,
                          path: MacAccess.path, approvalPort: approvals.port, approvalToken: approvals.token,
                          memoryRoute: memoryRoute, verifyRoute: verifyRoute)
        return true
    }

    private func localModelEndpoint(for session: AgentSession) -> (url: String, key: String)? {
        switch session.config.provider {
        case .ollama:
            return ("http://127.0.0.1:\(Ollama.port)", ModelProvider.ollama.placeholderKey)
        case .openRouter:
            do { try openRouterProxy.start() } catch {
                session.noticeFromApp("Couldn't start the OpenRouter connection: \(error.localizedDescription)")
                return nil
            }
            return ("http://127.0.0.1:\(OpenRouter.proxyPort)", openRouterProxy.token)
        case .claudeAccount, .codexAccount:
            return ("", "")
        }
    }

    /// The focused desktop window and any existing browser tab, from one guest query.
    static func screenNow(shell: GuestShell, ip: String) async -> String? {
        let command = #"""
        pid=$(DISPLAY=:0 xdotool getactivewindow getwindowpid 2>/dev/null)
        if [[ $pid =~ ^[0-9]+$ ]]; then
          ps -p "$pid" -o comm= 2>/dev/null | head -c 80 | base64 -w0
        fi
        printf '\n'
        curl -fs -m 2 http://127.0.0.1:9222/json/list || true
        """#
        guard let output = try? await shell.run(command, on: ip), let separator = output.firstIndex(of: "\n") else {
            return "Screen now: state unavailable. Inspect the desktop with ui tree or the browser with web snapshot."
        }
        let process = Data(base64Encoded: String(output[..<separator])).map { String(decoding: $0, as: UTF8.self) } ?? ""
        let json = output[output.index(after: separator)...]
        let tabs = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [[String: Any]]
        let foreground = screenLabel(process, limit: 80)
        let window = foreground.isEmpty ? "No focused app could be identified." : "Focused desktop app process: \(foreground)."
        guard let page = tabs?.first(where: { $0["type"] as? String == "page" }) else {
            return "Screen now: \(window) Use ui tree to inspect the desktop or web snapshot if the task uses a browser."
        }
        let address = screenLabel(page["url"] as? String ?? "", limit: 200)
        let pageTitle = screenLabel(page["title"] as? String ?? "", limit: 80)
        return "Screen now: \(window) Browser tab: \(address), page title: \(pageTitle). "
            + "Inspect the active app with ui tree or the page with web snapshot."
    }

    private static func screenLabel(_ value: String, limit: Int) -> String {
        value.prefix(limit).filter { $0 == " " || (!$0.isWhitespace && !"[]\"{}<>`".contains($0)) }
            .trimmingCharacters(in: .whitespaces)
    }

    private func vmChanged(_ state: LinuxVM.State) {
        switch state {
        case .starting, .running:
            return
        case .stopping:
            phase = .stopping
        case .stopped:
            if case .needsSetup = phase {} else { phase = .stopped }
        case let .failed(message):
            phase = .failed(message)
        }
        cancelSandboxStartup()
        sandboxCanCheckpoint = false
        sandboxNeedsFullToolInstall = false
        resetSandboxAccountStatus()
        guestIP = nil
    }

    private func cancelSandboxStartup() {
        sandboxStartup?.cancel()
        sandboxStartup = nil
    }

    private func resetSandboxAccountStatus() {
        sandboxAccountGeneration = UUID()
        accountChecks = accountChecks.filter { !$0.key.inSandbox }
        accountStatuses = accountStatuses.filter { !$0.key.inSandbox }
        for (route, task) in accountSignIns where route.inSandbox { task.cancel() }
    }
}
