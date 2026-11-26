import SwiftUI
import AgentSCore

/// The conversation: messages, replies, and expandable runs of tool steps.
/// The pill on top switches the model; the Computer button opens the screen beside the chat.
struct ChatView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var showComputer: Bool
    @State private var drafts: [UUID: String] = [:]
    @FocusState private var composerFocused: Bool
    @ScaledMetric(relativeTo: .callout) private var followUpRowHeight: CGFloat = 26
    @State private var followsLatest = true
    @State private var transcriptViewport: CGSize?
    @State private var showsOpenRouterCatalog = false
    @State private var showsOpenRouterSettings = false
    @State private var showsOllamaCatalog = false

    private enum TranscriptAnchor: Hashable {
        case row(String)
        case working
    }

    private struct TranscriptRevision: Equatable {
        let agentID: UUID
        let lastItem: ChatItem?
        let target: TranscriptAnchor?
    }

    private static let nearBottom: CGFloat = 40

    var body: some View {
        chat
            .task(id: "\(agent.id)-\(model.harness.rawValue)-\(model.hasSaiKey)") {
                if model.harness == .sai { await model.refreshSai() }
            }
            .task(id: model.shouldCheckOllamaForChat ? agent.id : nil) {
                if model.shouldCheckOllamaForChat { await model.refreshOllama() }
            }
            .task(id: model.provider == .openRouter) {
                if model.provider == .openRouter { await model.refreshOpenRouterModels() }
            }
            .task(id: accountSelection) {
                if accountSelection != nil { await model.refreshAccountStatus() }
            }
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                guard accountSelection != nil else { return }
                switch model.accountStatus {
                case .awaitingSignIn, .awaitingBrowserApproval:
                    Task { await model.refreshAccountStatus() }
                default:
                    break
                }
            }
            .sheet(isPresented: $showsOllamaCatalog) {
                OllamaCatalogView(selection: $model.agentModel)
            }
            .sheet(isPresented: $showsOpenRouterCatalog) {
                OpenRouterCatalogView(selection: $model.openRouterModel)
            }
            .sheet(isPresented: $showsOpenRouterSettings) {
                VStack(spacing: 0) {
                    Form { OpenRouterSettings() }.formStyle(.grouped)
                    Divider()
                    HStack {
                        Spacer()
                        Button("Done") { showsOpenRouterSettings = false }.keyboardShortcut(.defaultAction)
                    }
                    .padding(12)
                }
                .frame(width: 520, height: 420)
            }
    }

    private var accountSelection: String? {
        guard model.harness != .sai, model.provider.usesAccount, model.accountComputerReady else { return nil }
        return "\(agent.id)-\(model.provider.rawValue)"
    }

    private var needsAccountSignIn: Bool {
        guard accountSelection != nil else { return false }
        if case .connected = model.accountStatus { return false }
        return true
    }

    private var chat: some View {
        VStack(spacing: 0) {
            topBar
            transcript
            if let banner = computerBanner {
                banner
                    .frame(maxWidth: 820)
                    .padding(.horizontal, 24).padding(.top, 8)
            }
            if model.shouldCheckOllamaForChat, !model.hasAgentModel {
                VStack(alignment: .leading, spacing: 8) {
                    OllamaModelSetup(name: model.agentModel)
                }
                .padding(12)
                .frame(maxWidth: 820, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
                .padding(.horizontal, 24).padding(.top, 8)
            }
            if needsAccountSignIn {
                HarnessAccountSetup()
                    .padding(12)
                    .frame(maxWidth: 820, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
                    .padding(.horizontal, 24).padding(.top, 8)
            }
            VStack(spacing: 4) {
                if !agent.queuedFollowUps.isEmpty { queuedFollowUps }
                composer
            }
            .frame(maxWidth: 820)
            .padding(.horizontal, 24).padding(.bottom, 16).padding(.top, 8)
        }
    }

    private var transcript: some View {
        let transcript = ChatTranscript(agent.items)
        let rows = transcript.rows
        let activityAtEnd: Bool
        if let last = rows.last, last.isCurrentTurn, case .activity = last.content {
            activityAtEnd = true
        } else {
            activityAtEnd = false
        }
        let showsWorkingRow = agent.isRunning && !activityAtEnd
        let target: TranscriptAnchor? = showsWorkingRow ? .working : rows.last.map { .row($0.id) }
        let revision = TranscriptRevision(agentID: agent.id, lastItem: transcript.lastItem, target: target)
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 0) {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if rows.isEmpty { emptyState }
                        ForEach(rows) { row in
                            ChatTranscriptRow(row: row, isRunning: agent.isRunning,
                                              showsWorking: agent.isRunning && activityAtEnd && row.id == rows.last?.id)
                                .id(TranscriptAnchor.row(row.id))
                        }
                        if showsWorkingRow {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Working…").foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 16).padding(.vertical, 10)
                            .background(Capsule().fill(Theme.bubble))
                            .id(TranscriptAnchor.working)
                        }
                    }
                    .padding(.horizontal, 24).padding(.vertical, 16)
                    .frame(maxWidth: 820, alignment: .leading)
                    .frame(maxWidth: .infinity)
                }
                .background(TranscriptScrollObserver { scrollView in
                    if let document = scrollView.documentView {
                        followsLatest = document.bounds.maxY - scrollView.contentView.bounds.maxY <= Self.nearBottom
                    }
                })
                .onGeometryChange(for: CGSize?.self) { geometry in
                    geometry.bounds(of: NamedCoordinateSpace.scrollView(axis: .vertical))?.size
                } action: { size in
                    if let size, size.height > 0 { trackViewport(size, proxy: proxy, target: target) }
                }
            }
            .overlay(alignment: .bottom) {
                Group {
                    if !followsLatest, revision.lastItem != nil {
                        CircleButton(systemImage: "arrow.down", help: "Jump to latest message") {
                            followsLatest = true
                            withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.88)) {
                                scrollToLatest(proxy, target: target)
                            }
                        }
                        .accessibilityLabel("Jump to latest message")
                        .padding(.bottom, 8)
                        .transition(.opacity)
                    }
                }
                .animation(reduceMotion ? nil : .easeInOut(duration: 0.16), value: followsLatest)
            }
            .onChange(of: revision.agentID) {
                followsLatest = true
            }
            .onChange(of: agent.chatItemRequest) { _, request in
                guard let request else { return }
                followsLatest = false
                withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.88)) {
                    proxy.scrollTo(TranscriptAnchor.row(request.itemID), anchor: .top)
                }
            }
            .task(id: revision) {
                // Text can stream into the last item without changing the number of rows.
                await Task.yield()
                guard !Task.isCancelled, followsLatest, revision.lastItem != nil else { return }
                scrollToLatest(proxy, target: target)
            }
        }
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy, target: TranscriptAnchor?) {
        // Target a rendered row so an estimate of earlier Markdown heights cannot hide the newest message.
        if let target { proxy.scrollTo(target, anchor: .bottom) }
    }

    private func trackViewport(_ size: CGSize, proxy: ScrollViewProxy, target: TranscriptAnchor?) {
        let previous = transcriptViewport
        transcriptViewport = size
        // Follow window and split-pane resizing without reacting to lazy content estimates.
        if followsLatest, let previous, previous != size {
            Task { @MainActor in
                await Task.yield()
                guard followsLatest, transcriptViewport == size else { return }
                scrollToLatest(proxy, target: target)
            }
        }
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(spacing: 12) {
            // Centred while there is room; slides left rather than under the Computer button.
            WindowHeaderSpace(minWidth: 16)
            modelPill
            WindowHeaderSpace(minWidth: 12)
            HStack {
                if model.privacy == .thisMac {
                    CircleButton(systemImage: "pip", help: "Show floating chat") {
                        model.showMacTurnPanel(for: agent)
                    }
                }
                // While the panel is open, the button lives at the top of the panel instead.
                if !showComputer { ComputerToggle(showComputer: $showComputer) }
            }
            .padding(.trailing, 16)
        }
        .frame(height: Theme.titleBarHeight)
    }

    /// Picks the harness, provider and model for the next message.
    private var modelPill: some View {
        Menu {
            Section("Harness") {
                ForEach(Harness.allowed(in: model.privacy, computer: model.computer)) { choice in
                    Button { model.harness = choice } label: {
                        if choice == model.harness { Label(choice.displayName, systemImage: "checkmark") } else { Text(choice.displayName) }
                    }
                }
            }
            if model.harness == .sai {
                Section("Sai computer") {
                    ForEach(model.saiMachines) { machine in
                        Button { model.saiMachine = machine.id } label: {
                            let name = machine.name + (machine.online ? "" : " (offline)")
                            if machine.id == model.saiMachine { Label(name, systemImage: "checkmark") } else { Text(name) }
                        }
                    }
                    if model.saiMachines.isEmpty { Text(model.hasSaiKey ? "No Sai computers" : "Add your Sai key in Settings") }
                }
                Section("Sai model · relative credit cost") {
                    ForEach(model.saiModels.filter(\.allowed)) { choice in
                        Button { model.saiModel = choice.id } label: {
                            let title = choice.optionTitle()
                            if choice.id == model.saiModel { Label(title, systemImage: "checkmark") } else { Text(title) }
                        }
                    }
                }
            }
            if model.harness != .sai {
                Section("Provider") {
                    if model.availableProviders.count == 1 {
                        Text(model.provider.displayName)
                    } else {
                        ForEach(model.availableProviders) { choice in
                            Button { model.provider = choice } label: {
                                if choice == model.provider { Label(choice.displayName, systemImage: "checkmark") } else { Text(choice.displayName) }
                            }
                        }
                    }
                }
                Section("Model") {
                    if model.provider == .ollama {
                        ForEach(model.modelChoices, id: \.self) { name in
                            Button { model.agentModel = name } label: {
                                let title = model.ollamaOptionTitle(name)
                                if name == model.agentModel { Label(title, systemImage: "checkmark") } else { Text(title) }
                            }
                        }
                        Button("Browse Ollama models…") { showsOllamaCatalog = true }
                        Button("Refresh model list") { Task { await model.refreshOllama() } }
                    } else if model.provider == .openRouter {
                        Button("Browse OpenRouter models…") { showsOpenRouterCatalog = true }
                        ForEach(openRouterChoices, id: \.self) { name in
                            Button { model.openRouterModel = name } label: {
                                let title = model.openRouterOptionTitle(name)
                                if name == model.openRouterModel { Label(title, systemImage: "checkmark") } else { Text(title) }
                            }
                        }
                        Button("OpenRouter settings…") { showsOpenRouterSettings = true }
                        Toggle("Request Zero Data Retention (ZDR)", isOn: $model.openRouterRequiresZDR)
                    } else if let notice = model.provider.accountDataNotice {
                        Text("Automatic")
                        if let billing = model.provider.accountBillingLabel { Text(billing) }
                        Text(notice)
                    }
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: model.privacy.location.symbol).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text("\(model.harness.displayName) · \(model.activeModel)").foregroundStyle(.secondary)
                Image(systemName: "chevron.down").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16).padding(.vertical, 9)
            .background(Capsule().fill(Theme.card).shadow(color: .black.opacity(0.08), radius: 6, y: 2))
            .overlay(Capsule().stroke(Theme.hairline))
            .contentShape(Capsule())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(agent.isRunning)
        .help(agent.isRunning ? "Switch after this answer" : "Switch the harness or the model")
    }

    /// The suggested OpenRouter models, plus the current one if it was picked in Settings.
    private var openRouterChoices: [String] {
        OpenRouter.suggested.contains(model.openRouterModel)
            ? OpenRouter.suggested : [model.openRouterModel] + OpenRouter.suggested
    }

    // MARK: Computer state, inline

    /// Shows the local computer's status and available controls above the composer.
    private var computerBanner: AnyView? {
        guard model.agent.config.usesLocalComputer else { return nil }
        if model.isSavingSandbox {
            return AnyView(bannerRow(symbol: nil, text: "Saving sandbox…") { EmptyView() })
        }
        switch model.phase {
        case .ready:
            return nil
        case .stopped:
            if model.hasSavedSandbox {
                return AnyView(bannerRow(symbol: "pause.circle", text: "Sandbox saved") {
                    Button("Resume") { Task { await model.start() } }
                        .buttonStyle(.borderedProminent).tint(Theme.accent)
                })
            }
            return AnyView(MemoryBanner())
        case let .preparing(step):
            if case let .downloadingDebian(progress) = step {
                return AnyView(downloadBanner(label: step.label, progress: progress))
            }
            return AnyView(bannerRow(symbol: nil, text: step.label) { EmptyView() })
        case let .booting(step):
            return AnyView(bannerRow(symbol: nil, text: step) { EmptyView() })
        case .stopping:
            return AnyView(bannerRow(symbol: nil, text: model.status.label) { EmptyView() })
        case let .failed(message):
            return AnyView(VStack(alignment: .trailing, spacing: 6) {
                bannerRow(symbol: "exclamationmark.triangle", text: message) {
                    if !model.files.isPrepared {
                        Button("Try again") { Task { await model.prepareAndStartSandbox() } }
                    } else if model.vm.machine != nil, model.vm.state != .running {
                        Button("Turn off") { Task { await model.stop() } }
                    } else {
                        let title = model.hasSavedSandbox ? "Resume" : model.vm.state == .running ? "Try again" : "Turn on"
                        Button(title) { Task { await model.start() } }
                    }
                }
                if model.hasSavedSandbox, model.vm.machine == nil {
                    DiscardSavedSandboxButton().padding(.trailing, 14)
                }
            })
        case .needsSetup:
            return AnyView(bannerRow(symbol: "gearshape", text: "Set up the sandbox to chat.") {
                Button("Set up") { Task { await model.prepareAndStartSandbox() } }
                    .buttonStyle(.borderedProminent).tint(Theme.accent)
            })
        }
    }

    private func downloadBanner(label: String, progress: Double?) -> some View {
        DownloadProgressView(label: label, progress: progress)
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
    }

    private func bannerRow<Action: View>(symbol: String?, text: String, @ViewBuilder action: () -> Action) -> some View {
        HStack(spacing: 10) {
            if let symbol { Image(systemName: symbol).foregroundStyle(.secondary) } else { ProgressView().controlSize(.small) }
            Text(text).font(.callout).foregroundStyle(.secondary).lineLimit(2)
            Spacer(minLength: 8)
            action()
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
    }

    // MARK: Composer

    private var draft: Binding<String> {
        let id = agent.id
        return Binding(get: { drafts[id, default: ""] }, set: { drafts[id] = $0 })
    }

    private var hasDraft: Bool {
        !draft.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canSend: Bool {
        !agent.isStopping && hasDraft && (agent.isRunning || (model.canChat && !agent.isSendingSteering))
    }

    private var queuedFollowUps: some View {
        let remaining = agent.queuedFollowUps.dropFirst()
        let rows = ForEach(remaining) { queuedFollowUpRow($0) }
        return VStack(alignment: .leading, spacing: 0) {
            if let next = agent.queuedFollowUps.first { queuedFollowUpRow(next, isNext: true) }
            if remaining.count > 2 {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) { rows }
                }
                .frame(height: 2 * followUpRowHeight)
            } else {
                rows
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.bubble))
    }

    private func queuedFollowUpRow(_ followUp: AgentSession.QueuedFollowUp, isNext: Bool = false) -> some View {
        let sending = agent.steeringFollowUpID == followUp.id
        let stopping = agent.forcedFollowUpID == followUp.id
        let status = if sending { "Steering…" }
            else if stopping { "Stopping…" }
            else if agent.followUpsPaused { "Paused" }
            else if agent.waitingForSteering(followUp) { "Waiting to steer" }
            else { "Next turn" }
        return HStack(spacing: 8) {
            if isNext || sending || stopping {
                Label(status, systemImage: agent.followUpsPaused && !sending && !stopping ? "pause.circle" : "clock")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize()
            }
            Text(verbatim: followUp.text)
                .font(.callout).lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(followUp.text)
            if agent.isRunning, !agent.isStopping, !sending {
                Button("Stop & send") { model.forceQueuedFollowUp(followUp, for: agent) }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .fixedSize()
                    .disabled(!model.canForceQueuedFollowUp(followUp, for: agent))
                    .help("Stop this turn and send this follow-up once it has stopped")
            } else if isNext, agent.followUpsPaused, !stopping, !sending {
                Button("Send next") { model.sendQueuedFollowUp(for: agent) }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .fixedSize()
                    .disabled(!model.canSendQueuedFollowUp(for: agent))
                    .help("Send the first queued follow-up")
            }
            Button { agent.removeQueuedFollowUp(followUp.id) } label: {
                Image(systemName: "xmark").font(.caption2.weight(.semibold))
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .disabled(sending)
            .accessibilityLabel("Remove queued follow-up")
            .help("Remove queued follow-up")
        }
        .frame(height: followUpRowHeight)
    }

    private var composer: some View {
        let stopsTurn = agent.isRunning && !hasDraft
        let submissionLabel = if !agent.isRunning { "Send" }
            else if agent.supportsSteering { "Steer the current turn" }
            else { "Queue follow-up" }
        let stopLabel = agent.forcedFollowUpID == nil ? "Stop" : "Cancel follow-up after stopping"
        let actionSymbol = stopsTurn ? "stop.fill" : "arrow.up"
        let actionEnabled = canSend || (stopsTurn && (!agent.isStopping || agent.forcedFollowUpID != nil))
        return HStack(alignment: .bottom, spacing: 10) {
            TextField(placeholder, text: draft, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...6)
                .focused($composerFocused)
                .onSubmit(send)
                .disabled(!model.canChat && !agent.isRunning && !agent.config.usesLocalComputer)
                .padding(.vertical, 9)
            roundButton(actionSymbol, help: stopsTurn ? stopLabel : submissionLabel) {
                if stopsTurn { agent.stop() }
                else { send() }
            }
            .keyboardShortcut(stopsTurn ? nil : KeyboardShortcut(.return, modifiers: .command))
            .disabled(!actionEnabled)
            .opacity(actionEnabled ? 1 : 0.4)
        }
        .padding(.leading, 18).padding(.trailing, 6).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 24).fill(Theme.card))
        .overlay(RoundedRectangle(cornerRadius: 24).stroke(Theme.hairline))
    }

    private func roundButton(_ systemImage: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Theme.onAccent)
                .frame(width: 34, height: 34)
                .background(Circle().fill(Theme.accent))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(help)
        .help(help)
    }

    private var placeholder: String {
        if agent.isStopping { return "Stopping the current turn…" }
        if agent.isRunning {
            return agent.supportsSteering ? "Steer the current turn…" : "Queue a follow-up…"
        }
        return model.chatBlocker ?? (model.harness == .sai
            ? "Ask \(model.agent.config.name) to do something on \(model.saiMachineName)…"
            : model.computer == .box ? "Ask \(model.agent.config.name) to do something on your Simular Box…" : "Ask \(model.agent.config.name) to do something on its computer…")
    }

    /// Helps set up the selected agent's computer and provider.
    @ViewBuilder private var emptyState: some View {
        VStack(spacing: 12) {
            if model.harness == .sai {
                Text("Sai works on your Sai computer").font(.title3.weight(.medium))
                Text("All in one: Sai's agent, model and cloud computer. Nothing runs on this Mac.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
                if !model.hasSaiKey {
                    Form { KeySetup.sai(model) }.formStyle(.grouped).frame(maxWidth: 480, maxHeight: 200)
                } else if model.saiMachines.isEmpty {
                    if let status = model.saiStatus {
                        Text(status).font(.callout).foregroundStyle(.red)
                    } else {
                        ProgressView("Finding your Sai computers…").controlSize(.small)
                    }
                    Button("Try again") { Task { await model.refreshSai() } }
                } else {
                    Picker("Computer", selection: $model.saiMachine) {
                        ForEach(model.saiMachines) { Text($0.name + ($0.online ? "" : " (offline)")).tag($0.id) }
                        if model.saiMachine.isEmpty { Text("Choose…").tag("") }
                    }
                    .frame(maxWidth: 320)
                }
            } else if model.privacy == .thisMac {
                Text("Works on this Mac").font(.title3.weight(.medium))
                Text("It uses your Mac's apps, Terminal and files. Before each step it shows you exactly what it will do, and runs it only when you allow it.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 440)
                MacAccessList().frame(maxWidth: 480)
                if model.provider == .openRouter, !model.hasOpenRouterKey {
                    Form { KeySetup.openRouter(model) }.formStyle(.grouped).frame(maxWidth: 480, maxHeight: 200)
                }
            } else if model.computer == .box {
                Text("Working on your Simular Box").font(.title3.weight(.medium))
                Text("The agent runs in this app and uses only your box’s tools.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
                if !model.hasBoxKey || (model.provider == .openRouter && !model.hasOpenRouterKey) {
                    Form {
                        if !model.hasBoxKey { KeySetup.box(model) }
                        if model.provider == .openRouter, !model.hasOpenRouterKey { KeySetup.openRouter(model) }
                    }
                    .formStyle(.grouped).frame(maxWidth: 480, maxHeight: 260)
                }
            } else {
                let usesOpenRouter = model.provider == .openRouter
                Text(model.privacy.location.displayName).font(.title3.weight(.medium))
                Text(usesOpenRouter
                     ? "The sandbox runs on this Mac. OpenRouter processes model requests."
                     : "The agent works in a Linux VM on this Mac. Open the computer to watch or take over.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
                if agent.config.usesLocalComputer, model.canChat { sandboxStarterPrompts }
                if usesOpenRouter, !model.hasOpenRouterKey {
                    Form { KeySetup.openRouter(model) }.formStyle(.grouped).frame(maxWidth: 480, maxHeight: 200)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, agent.config.usesLocalComputer ? 24 : 80)
    }

    private var sandboxStarterPrompts: some View {
        VStack(spacing: 6) {
            ForEach([
                "Open example.com and summarize the page.",
                "Compare the Moon and Mars Wikipedia pages in two tabs.",
                "Save a short checklist to ~/work/checklist.txt and open it in the sandbox browser.",
            ], id: \.self) { prompt in
                Button {
                    draft.wrappedValue = prompt
                    composerFocused = true
                } label: {
                    HStack(spacing: 8) {
                        Text(prompt).multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "arrow.turn.down.right")
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                    }
                    .font(.callout)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Theme.bubble))
                }
                .buttonStyle(.plain)
                .help("Use this prompt as your draft")
                .accessibilityHint("Loads this prompt in the draft")
            }
        }
        .frame(maxWidth: 420)
    }

    private func send() {
        guard canSend else { return }
        let session = agent
        let field = draft
        let original = field.wrappedValue
        let text = original.trimmingCharacters(in: .whitespacesAndNewlines)
        let wasRunning = session.isRunning
        let accepted = wasRunning ? model.queueFollowUp(text, for: session) : model.send(text, to: session)
        if accepted {
            if field.wrappedValue == original { field.wrappedValue = "" }
            if !wasRunning || session.supportsSteering { followsLatest = true }
        }
    }
}

/// Listens to user scrolling in the transcript, separately from programmatic layout changes.
private struct TranscriptScrollObserver: NSViewRepresentable {
    let onScroll: (NSScrollView) -> Void

    func makeNSView(context: Context) -> ObserverView { ObserverView() }
    func updateNSView(_ view: ObserverView, context: Context) { view.onScroll = onScroll }

    final class ObserverView: NSView {
        var onScroll: ((NSScrollView) -> Void)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            NotificationCenter.default.removeObserver(self, name: NSScrollView.didLiveScrollNotification, object: nil)
            guard window != nil, let scrollView = enclosingScrollView else { return }
            NotificationCenter.default.addObserver(self, selector: #selector(didScroll(_:)),
                                                   name: NSScrollView.didLiveScrollNotification, object: scrollView)
        }

        @objc private func didScroll(_ notification: Notification) {
            if let scrollView = notification.object as? NSScrollView { onScroll?(scrollView) }
        }

        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}

struct ChatRow: View {
    let item: ChatItem
    var isToolActive = true
    @State private var open = false

    var body: some View {
        switch item.kind {
        case .user:
            HStack {
                Spacer(minLength: 120)
                VStack(alignment: .trailing, spacing: 4) {
                    if item.isSteering == true {
                        Text("Steering").font(.caption).foregroundStyle(.secondary)
                    }
                    UserPromptBubble(prompt: item.text)
                }
            }
        case .agent:
            ChatMarkdown(markdown: item.text)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
        case let .step(name, done, failed, output):
            toolStep(name: name, done: done, failed: failed, output: output)
        case let .approval(id, linkOnly, url, answer):
            ApprovalCard(id: id, text: item.text, linkOnly: linkOnly, url: url, answer: answer)
        case let .notice(isError):
            Text(item.text).font(.callout).foregroundStyle(isError ? Color.red : Color.secondary)
                .padding(.horizontal, 4)
        }
    }

    private func toolStep(name: String, done: Bool, failed: Bool, output: String) -> some View {
        let title = ToolActivity.title(tool: name, summary: item.text, detail: item.toolInput)
        let status = failed ? "Failed" : done ? "Completed" : isToolActive ? "Running" : "Unfinished"
        let symbol = failed ? "exclamationmark.circle" : done ? "checkmark.circle" : isToolActive ? "circle.dotted" : "clock"
        let input = item.toolInput ?? (item.text == name || item.text == title ? "" : item.text)

        return VStack(alignment: .leading, spacing: 6) {
            Button { open.toggle() } label: {
                HStack(spacing: 8) {
                    Image(systemName: symbol)
                        .foregroundStyle(failed ? Color.red : done ? Color.green : Color.secondary)
                    Text(title).font(.callout).lineLimit(1)
                    Image(systemName: open ? "chevron.down" : "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(RoundedRectangle(cornerRadius: 14).stroke(Theme.hairline))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(title)
            .accessibilityValue(status)
            .accessibilityHint(open ? "Hide step details" : "Show step details")

            if open {
                let details = VStack(alignment: .leading, spacing: 8) {
                    stepDetail("Tool", name)
                    stepDetail("Input", input)
                    stepDetail("Output", output)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                ViewThatFits(in: .vertical) {
                    details
                    ScrollView { details }
                }
                .frame(maxHeight: 180)
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 10).fill(Theme.bubble))
            }
        }
    }

    @ViewBuilder private func stepDetail(_ label: String, _ value: String) -> some View {
        if !value.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                Text(label).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(verbatim: value).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
        }
    }
}

/// Sai paused for the user: approve or deny, or open the link it needs a person for.
struct ApprovalCard: View {
    @EnvironmentObject private var agent: AgentSession
    let id: String
    let text: String
    let linkOnly: Bool
    let url: URL?
    let answer: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(text, systemImage: "hand.raised").font(.callout)
            if let answer {
                Text(answer == "deny" ? "Denied" : "Approved").font(.caption).foregroundStyle(.secondary)
            } else if linkOnly {
                if let url { Link("Open to continue", destination: url) }
                Text("Sai continues once it's done.").font(.caption).foregroundStyle(.secondary)
            } else {
                HStack {
                    Button("Approve") { agent.answer(approval: id, with: "approve") }
                        .buttonStyle(.borderedProminent).tint(Theme.accent)
                    Button("Approve for this task") { agent.answer(approval: id, with: "approve_for_task") }
                    Button("Deny") { agent.answer(approval: id, with: "deny") }
                }
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 16).stroke(Theme.hairline))
    }
}

/// Compares the memory needed to turn on the computer with what the Mac has free.
private struct MemoryBanner: View {
    @EnvironmentObject private var model: AppModel
    @State private var free: UInt64?

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: short ? "exclamationmark.triangle.fill" : "power")
                .foregroundStyle(short ? Color.orange : Color.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(short ? "Your Mac may not have enough memory free." : "The computer is off.")
                    .font(.callout).foregroundStyle(short ? Color.primary : Color.secondary)
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer()
            Button("Turn on") { Task { await model.start() } }
                .buttonStyle(.borderedProminent).tint(Theme.accent)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
        .task { await watch() }
    }

    private var short: Bool { free.map { $0 < model.computerMemory.bytes } ?? false }

    private var detail: String {
        let allocation = "The computer is configured for \(SystemMemory.format(model.computerMemory.bytes))."
        guard let free else { return allocation }
        let summary = allocation + " \(SystemMemory.format(free)) of \(SystemMemory.format(SystemMemory.total)) is free."
        return summary + (short ? " Quit some apps first, or it may be slow." : "")
    }

    /// Rechecks every few seconds while the banner shows (apps open and close).
    private func watch() async {
        while !Task.isCancelled {
            free = SystemMemory.available()
            try? await Task.sleep(for: .seconds(5))
        }
    }
}

/// What a This Mac agent needs from macOS and this Mac, each with a way to fix it.
struct MacAccessList: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            MacAccessRows(showsHarnessSetup: true)
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
        .onAppear { model.refreshMacAccess() }
    }
}

/// Shared permission controls that also fit inside Settings' grouped form.
struct MacAccessRows: View {
    @EnvironmentObject private var model: AppModel
    var showsHarnessSetup = false

    var body: some View {
        Group {
            row("Accessibility", detail: "To click and type in your apps", ok: model.macAccessibility) {
                Button("Allow…") {
                    model.refreshMacAccess()
                    if !model.macAccessibility { MacAccess.askForAccessibility() }
                }
            }
            row("Screen Recording", detail: "For screenshots (optional)", ok: model.macScreenRecording) {
                Button("Allow…") {
                    model.refreshMacAccess()
                    if !model.macScreenRecording { MacAccess.askForScreenRecording() }
                }
            }
            if showsHarnessSetup {
                row("simulang", detail: "npm install -g @simular-ai/simulang", ok: MacAccess.simulang != nil) { EmptyView() }
                if model.harness == .claudeCode {
                    row("Claude Code", detail: "claude.com/claude-code", ok: BoxAccount.claude != nil) { EmptyView() }
                }
            }
            if !model.macAccessibility || !model.macScreenRecording {
                Text("All This Mac agents share these permissions. After updating them in System Settings, return to Agent S. Screen Recording may require reopening the app.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Check permissions") { model.refreshMacAccess() }
                    .controlSize(.small)
            }
        }
    }

    private func row<Action: View>(_ title: String, detail: String, ok: Bool, @ViewBuilder action: () -> Action) -> some View {
        HStack(spacing: 10) {
            Image(systemName: ok ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(ok ? Color.green : Color.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.callout)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if !ok { action() }
        }
    }
}
