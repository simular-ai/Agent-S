import Foundation
import AppKit
import AgentSCore

/// The app's state and the order things happen in: prepare the guest, boot it, find it, relay
/// Ollama to it, wait for its first-boot setup, then chat.
@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable {
        case needsSetup
        case preparing(String)
        case stopped
        case booting(String)
        case ready
        case failed(String)
    }

    enum AccountStatus: Equatable {
        case unchecked
        case checking
        case signedOut
        case connecting
        case awaitingSignIn
        case connected(HarnessAccountPlan)
        case failed(String)

        var isConnected: Bool {
            if case .connected = self { return true }
            return false
        }
    }

    /// Claude Code's model: tool calling and a long context matter most.
    static let defaultAgentModel = "gemma4"
    /// Ollama >= 0.14 serves the Anthropic Messages API Claude Code speaks.
    static let minimumOllama = "0.14.0"
    private static let computerMemoryKey = "computerMemoryGB"

    @Published private(set) var phase: Phase
    @Published private(set) var downloadProgress: Double?
    @Published private(set) var guestIP: String?
    @Published var computerMemory = LinuxVM.MemoryAllocation(rawValue: UserDefaults.standard.integer(forKey: AppModel.computerMemoryKey)) ?? .sixGB {
        didSet { UserDefaults.standard.set(computerMemory.rawValue, forKey: Self.computerMemoryKey) }
    }
    @Published private(set) var ollamaChecked = false
    @Published private(set) var ollamaVersion: String?
    @Published private(set) var ollamaModels: [Ollama.Model] = []
    @Published private(set) var pullStatus: String?
    @Published private(set) var openRouterModels: [OpenRouter.Model] = []
    @Published private(set) var hasOpenRouterKey = OpenRouter.hasKey
    // Sai harness (cloud agents): the user's Sai key, computers and models.
    @Published private(set) var hasSaiKey = SaiAPI.hasKey
    @Published private(set) var saiKeyRevision = 0
    @Published private(set) var saiMachines: [SaiAPI.Machine] = []
    @Published private(set) var saiModels: [SaiAPI.Model] = []
    @Published private(set) var saiStatus: String?
    @Published private var saiScreenStates: [String: SaiAPI.ScreenState] = [:]
    private let openRouterProxy = OpenRouterProxy()
    @Published private var accountStatuses: [ModelProvider: AccountStatus] = [:]
    private var accountChecks: [ModelProvider: UUID] = [:]
    private var accountSignIns: [ModelProvider: Task<Void, Never>] = [:]

    let files = GuestFiles.standard
    let vm: LinuxVM

    // MARK: Agents

    /// The user's agents, private ones (on this Mac) and cloud ones; each with its own context.
    @Published private(set) var agents: [AgentSession] = []
    @Published var selectedAgentID: UUID {
        didSet { UserDefaults.standard.set(selectedAgentID.uuidString, forKey: "selectedAgent") }
    }
    /// The selected agent, whose chat and settings the window shows.
    var agent: AgentSession { agents.first { $0.id == selectedAgentID } ?? agents[0] }

    func agents(at location: AgentLocation) -> [AgentSession] {
        agents.filter { $0.config.privacy.location == location }
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
            update { $0.provider = newValue }
        }
    }
    var availableProviders: [ModelProvider] {
        ModelProvider.allowed(in: privacy, harness: harness, computer: computer)
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

    /// Adds an agent with a fresh context and selects it; it gets the next headshot color.
    func createAgent(_ config: AgentConfig) {
        var config = config.enforcingPrivacy
        if config.avatarColor == nil { config.avatarColor = agents.count % AgentAvatar.palette.count }
        adopt(AgentSession(config: config))
        selectedAgentID = config.id
        saveAgents()
    }

    func rename(_ session: AgentSession, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        session.config.name = trimmed
    }

    /// Deletes an agent and its context; the last agent stays.
    func delete(_ session: AgentSession) {
        guard agents.count > 1 else { return }
        session.stop()
        macTurnPanel.remove(session)
        agents.removeAll { $0.id == session.id }
        if selectedAgentID == session.id { selectedAgentID = agents[0].id }
        saveAgents()
    }

    func select(_ session: AgentSession) { selectedAgentID = session.id }

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
        session.onChange = { [weak self] in
            self?.objectWillChange.send()
            self?.saveAgents()
        }
        session.onMacUIActivity = { [weak self] session, needsApproval in
            self?.showMacTurnPanel(for: session, presentation: .macUI(needsApproval: needsApproval))
        }
        agents.append(session)
    }

    private var agentsFile: URL { files.root.appendingPathComponent("agents.json") }

    /// Chats can hold page text and tool output: the file is readable by this user only.
    func saveAgents() {
        // While an earlier build runs, this copy only shows "quit it first" and writes nothing.
        guard !GuestFiles.legacyAppRunning, let data = try? JSONEncoder().encode(agents.map(\.saved)) else { return }
        try? files.ensureFolders()
        try? data.write(to: agentsFile, options: .atomic)
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: agentsFile.path)
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
        loadAgents()
        let saved = (UserDefaults.standard.string(forKey: "selectedAgent")
            ?? UserDefaults(suiteName: GuestFiles.legacy[0].0)?.string(forKey: "selectedAgent"))
            .flatMap(UUID.init(uuidString:))
        selectedAgentID = agents.first { $0.id == saved }?.id ?? agents[0].id
        vm.onStateChange = { [weak self] state in self?.vmChanged(state) }
    }

    var shell: GuestShell { GuestShell(files: files, mac: vm.macAddress) }

    var ollamaReady: Bool {
        guard let ollamaVersion else { return false }
        return Ollama.isAtLeast(ollamaVersion, Self.minimumOllama)
    }

    var hasAgentModel: Bool {
        ollamaModels.contains { $0.matches(agentModel) }
    }

    var hasOllamaAgents: Bool {
        agents.contains { $0.config.harness != .sai && $0.config.provider == .ollama }
    }

    /// Use the selected model's metadata so Ollama cloud models are identified as hosted.
    func sidebarInference(for config: AgentConfig) -> (label: String, help: String) {
        if config.harness == .sai { return ("Hosted", "Sai") }

        switch config.provider {
        case .openRouter: return ("Hosted", "OpenRouter · \(config.openRouterModel)")
        case .claudeAccount, .codexAccount:
            let confirmed = accountStatuses[config.provider] == .connected(.subscription)
            let provider = config.provider == .claudeAccount ? "Claude Code account" : "Codex account"
            return (confirmed ? "Subscription" : "Account", provider)
        case .ollama:
            let model = ollamaModels.first { $0.matches(config.ollamaModel) }
            let provider = "Ollama · \(config.ollamaModel)"
            if model?.usesRemoteInference == true || Ollama.isCloudReference(config.ollamaModel) {
                return ("Hosted", provider)
            }
            if model != nil, ollamaReady { return ("Local", provider) }
            return ("Ollama", "\(provider) · Model location unavailable")
        }
    }

    /// The models Ollama has, without the `:latest` tag, plus the current choice if it isn't
    /// downloaded yet (so it can still be picked and downloaded).
    var modelChoices: [String] {
        var names = ollamaModels.map { $0.name.hasSuffix(":latest") ? String($0.name.dropLast(7)) : $0.name }
        if !names.contains(agentModel) { names.insert(agentModel, at: 0) }
        return names
    }

    // MARK: Actions

    func prepare() async {
        downloadProgress = nil
        do {
            try await GuestImage.prepare(files: files,
                                         step: { [weak self] in self?.phase = .preparing($0) },
                                         progress: { [weak self] in self?.downloadProgress = $0 })
            downloadProgress = nil
            phase = .stopped
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func start() async {
        phase = .booting("Starting the computer…")
        await vm.start(memory: computerMemory)
        guard vm.state == .running else { return }

        phase = .booting("Waiting for the computer to come online…")
        guard let ip = await shell.waitUntilReachable() else {
            phase = .failed("The computer started but never answered on its network.")
            return
        }
        guestIP = ip

        phase = .booting("Installing the desktop, simulang and Claude Code (first start only, a few minutes)…")
        while !(await shell.isReady(on: ip)) {
            if vm.state != .running { return }
            try? await Task.sleep(nanoseconds: 3_000_000_000)
        }
        phase = .booting("Installing Sai's screen tools…")
        do { try await shell.run(GuestTools.installCommand, on: ip) } catch {
            NSLog("Agent S: installing the screen tools failed: %@", "\(error)")
        }
        phase = .ready
    }

    func stop() async {
        agents.forEach { $0.stop() }
        await vm.stop()
    }

    func stopAppWork() {
        accountSignIns.values.forEach { $0.cancel() }
        agents.forEach { $0.stop() }
    }

    func refreshOllama() async {
        async let version = Ollama.version()
        async let models = Ollama.models()
        let (checkedVersion, checkedModels) = await (version, models)
        guard !Task.isCancelled else { return }
        ollamaVersion = checkedVersion
        ollamaModels = checkedModels
        ollamaChecked = true
    }

    func pullAgentModel() async {
        pullStatus = "Starting…"
        do {
            try await Ollama.pull(agentModel) { [weak self] status, fraction in
                self?.pullStatus = fraction.map { "\(status) \(Int($0 * 100))%" } ?? status
            }
            pullStatus = nil
            await refreshOllama()
        } catch {
            pullStatus = "Couldn't pull \(agentModel): \(error.localizedDescription)"
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
        guard let ip = guestIP else { throw OnePassword.Failure(message: "The computer is off.") }
        try await shell.run("DISPLAY=:0 xdotool type --delay 12 --file -", on: ip, input: text)
        if thenTab { try await shell.run("DISPLAY=:0 xdotool key Tab", on: ip) }
    }

    // MARK: Model provider

    /// The model the next message runs on, for the chosen harness and provider.
    var activeModel: String {
        if harness == .sai { return saiModel }
        if provider.usesAccount { return "Automatic" }
        return provider == .ollama ? agentModel : openRouterModel
    }

    /// Whether the selected computer, harness, and model are ready for a message.
    var canChat: Bool { chatBlocker == nil }

    var usesOllama: Bool { harness != .sai && provider == .ollama }

    var shouldCheckOllamaForChat: Bool { usesOllama && computerChatBlocker == nil }

    private var modelBlocker: String? {
        if provider.usesAccount, !accountStatus.isConnected {
            return accountStatus == .checking ? "Checking sign-in…" : "Connect your \(accountName) account to chat"
        }
        if usesOllama {
            if !ollamaChecked { return "Checking Ollama…" }
            if !hasAgentModel { return "Download \(agentModel) in Settings first" }
        }
        if provider == .openRouter {
            if !hasOpenRouterKey { return "Add your OpenRouter key in Settings first" }
            if openRouterModel.isEmpty { return "Choose an OpenRouter model in Settings first" }
        }
        return nil
    }

    /// Why a message can't be sent, for the composer.
    var chatBlocker: String? {
        computerChatBlocker ?? (harness == .sai ? nil : modelBlocker)
    }

    private var computerChatBlocker: String? {
        if privacy == .thisMac {
            if harness == .codex {
                if CodexAccount.executable == nil { return "Install Codex on this Mac to use this agent" }
            } else if BoxAccount.claude == nil { return "Install Claude Code on this Mac to use This Mac agents" }
            if MacAccess.simulang == nil { return "Install simulang first: npm install -g @simular-ai/simulang" }
            if !macAccessibility { return "Allow Accessibility in Settings to use this Mac" }
            return nil
        }
        if computer == .box {
            if !hasBoxKey { return "Connect Simular Box in Settings" }
            switch harness {
            case .claudeCode where BoxAccount.claude == nil: return "Install Claude Code on this Mac"
            case .codex where CodexAccount.executable == nil: return "Install Codex on this Mac"
            case .deepSeek where DeepSeekBoxTurn.executable == nil:
                return "Install DeepSeek Harness on this Mac"
            default: break
            }
            return nil
        }
        if harness == .sai {
            if !hasSaiKey { return "Add your Sai API key in Settings" }
            if saiMachine == SaiAPI.localMachine { return "This Mac's computer isn't available to Sai yet: pick a Sai computer" }
            if saiMachine.isEmpty { return saiMachines.isEmpty ? "Finding your Sai computers…" : "Choose a Sai computer above" }
            return nil
        }
        switch phase {
        case .needsSetup: return "Set up the computer in Settings to chat"
        case .ready:
            if agents.contains(where: { $0 !== agent && $0.config.usesLocalComputer && $0.isRunning }) {
                return "Another agent is using the local sandbox"
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

    func refreshOpenRouterModels() async {
        if openRouterModels.isEmpty { openRouterModels = await OpenRouter.models() }
    }

    // MARK: Harness accounts

    var accountStatus: AccountStatus { accountStatuses[provider] ?? .unchecked }

    private var accountName: String { provider == .codexAccount ? "Codex" : "Claude Code" }

    private struct AccountFailure: LocalizedError {
        let errorDescription: String?
    }

    private func accountPlan(_ provider: ModelProvider) async throws -> HarnessAccountPlan? {
        if provider == .codexAccount { return try await CodexAccount.accountPlan(dataFolder: files.root) }
        guard let claude = BoxAccount.claude else {
            throw AccountFailure(errorDescription: "Install Claude Code on this Mac first.")
        }
        return await ClaudeAccount.accountPlan(claude: claude, profile: ClaudeAccount.profile(in: files.root))
    }

    func refreshAccountStatus() async {
        let provider = provider
        guard availableProviders.contains(provider), provider.usesAccount, accountSignIns[provider] == nil else { return }
        let check = UUID()
        let previous = accountStatuses[provider] ?? .unchecked
        accountChecks[provider] = check
        if !previous.isConnected, previous != .awaitingSignIn { accountStatuses[provider] = .checking }
        let status: AccountStatus
        do {
            let plan = try await accountPlan(provider)
            status = plan.map(AccountStatus.connected) ?? (previous == .awaitingSignIn ? .awaitingSignIn : .signedOut)
        } catch {
            status = .failed(error.localizedDescription)
        }
        guard accountChecks[provider] == check else { return }
        accountChecks[provider] = nil
        accountStatuses[provider] = Task.isCancelled ? (previous == .checking ? .unchecked : previous) : status
    }

    func signInToAccount() async {
        let provider = provider
        guard availableProviders.contains(provider), provider.usesAccount, accountSignIns[provider] == nil else { return }
        accountChecks[provider] = nil
        accountStatuses[provider] = .connecting
        let task = Task { @MainActor in
            defer { accountSignIns[provider] = nil }
            do {
                if provider == .codexAccount {
                    let plan = try await CodexAccount.signIn(dataFolder: files.root)
                    try Task.checkCancellation()
                    accountStatuses[provider] = .connected(plan)
                } else {
                    guard let claude = BoxAccount.claude else {
                        throw AccountFailure(errorDescription: "Install Claude Code on this Mac first.")
                    }
                    try await ClaudeAccount.signIn(claude: claude, profile: ClaudeAccount.profile(in: files.root))
                    accountStatuses[provider] = .awaitingSignIn
                }
            } catch {
                accountStatuses[provider] = Task.isCancelled ? .signedOut : .failed(error.localizedDescription)
            }
        }
        accountSignIns[provider] = task
        await task.value
    }

    func cancelAccountSignIn() { accountSignIns[provider]?.cancel() }

    func send(_ prompt: String) {
        // The private/cloud boundary is enforced here too, not only by what the settings offer.
        let config = agent.config.enforcingPrivacy
        guard config == agent.config else { agent.config = config; return }
        guard canChat, !agent.isRunning else { return }
        if harness == .sai {
            agent.sendToSai(prompt, machine: saiMachine, model: saiModel)
            return
        }
        if computer == .box {
            sendToBox(prompt)
            return
        }
        if privacy == .thisMac {
            sendToMac(prompt)
            return
        }
        guard config.usesLocalComputer, let ip = guestIP else { return }
        let shell = shell
        let port = provider == .openRouter ? OpenRouter.proxyPort : Ollama.port
        var apiKey: String?
        if provider == .openRouter {
            do { try openRouterProxy.start() } catch {
                agent.noticeFromApp("Couldn't start the OpenRouter connection: \(error.localizedDescription)")
                return
            }
            apiKey = openRouterProxy.token
        }
        agent.send(prompt, harness: harness, shell: shell, ip: ip, model: activeModel,
                   ollamaURL: "http://127.0.0.1:\(port)", provider: provider, apiKey: apiKey, tunnelTo: port) {
            await Self.screenNow(shell: shell, ip: ip)
        }
    }

    /// Box turns run the selected harness with only the remote Box's tool set.
    private func sendToBox(_ prompt: String) {
        let (baseURL, apiKey) = localModelEndpoint
        switch harness {
        case .claudeCode:
            guard let claude = BoxAccount.claude, let boxToken = BoxAccount.key() else { return }
            let command = BoxClaudeCommand(prompt: prompt, resume: nil, model: activeModel, provider: provider, apiKey: apiKey,
                                           boxName: agent.config.boxName, baseURL: baseURL)
            agent.sendOnBox(prompt, command: command, claude: claude, boxToken: boxToken, dataFolder: files.root)
        case .codex:
            agent.sendWithCodex(prompt, dataFolder: files.root, model: activeModel, baseURL: baseURL, apiKey: apiKey)
        case .deepSeek:
            agent.sendWithDeepSeekOnBox(prompt, dataFolder: files.root, model: activeModel, baseURL: baseURL, apiKey: apiKey)
        case .sai: break // Sai runs through its own API on a Sai computer.
        }
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
    private func sendToMac(_ prompt: String) {
        do {
            try MacAccess.install()
            if harness == .claudeCode { try approvals.start() }
        } catch {
            agent.noticeFromApp("Couldn't get ready to work on this Mac: \(error.localizedDescription)")
            return
        }
        if provider == .codexAccount {
            agent.sendWithCodex(prompt, dataFolder: files.root, model: activeModel)
            return
        }
        guard let claude = BoxAccount.claude else { return }
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
        let (baseURL, apiKey) = localModelEndpoint
        let command = MacClaudeCommand(prompt: prompt, resume: nil, model: activeModel, provider: provider, apiKey: apiKey,
                                       baseURL: baseURL, macToolPath: MacAccess.macTool)
        agent.sendOnMac(prompt, command: command, claude: claude, dataFolder: files.root, settingsPath: MacAccess.settings,
                        path: MacAccess.path, approvalPort: approvals.port, approvalToken: approvals.token)
    }

    private var localModelEndpoint: (url: String, key: String) {
        switch provider {
        case .ollama:
            return ("http://127.0.0.1:\(Ollama.port)", ModelProvider.ollama.placeholderKey)
        case .openRouter:
            do { try openRouterProxy.start() } catch {
                NSLog("Agent S: couldn't start the OpenRouter proxy: %@", "\(error)")
            }
            return ("http://127.0.0.1:\(OpenRouter.proxyPort)", openRouterProxy.token)
        case .claudeAccount, .codexAccount:
            return ("", "")
        }
    }

    /// One line on what the computer's browser shows right now, from Chromium's own tab list (fast).
    /// Small models otherwise answer follow-ups from what they saw turns ago. The page title is the
    /// page's own text, so it is cut short, stripped of brackets and quotes, and labelled as such.
    static func screenNow(shell: GuestShell, ip: String) async -> String? {
        let nothing = "Screen now: no website is open. Use web open <url> to start."
        guard let json = try? await shell.run("curl -s -m 2 http://127.0.0.1:9222/json/list", on: ip),
              let tabs = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]],
              let page = tabs.first(where: { $0["type"] as? String == "page" })
        else { return nothing }
        let address = (page["url"] as? String ?? "").prefix(200)
        let title = String((page["title"] as? String ?? "").prefix(80))
            .filter { !"[]\"{}<>`".contains($0) && !$0.isNewline }
        return "Screen now: the browser is at \(address), page title (from the page) \(title). "
            + "This is current; what you saw earlier in this chat may be out of date. Run web snapshot to see the page."
    }

    private func vmChanged(_ state: LinuxVM.State) {
        switch state {
        case .stopped:
            guestIP = nil
            if case .needsSetup = phase {} else { phase = .stopped }
        case let .failed(message):
            guestIP = nil
            phase = .failed(message)
        default:
            break
        }
    }
}
