import SwiftUI
import AgentSCore

/// The conversation: the user's messages, the agent's answers, and its steps folded to one line each.
/// The pill on top switches the model; the Computer button opens the screen beside the chat.
struct ChatView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var showComputer: Bool
    @State private var draft = ""
    @State private var followsLatest = true
    @State private var transcriptPosition: TranscriptPosition?

    private enum TranscriptAnchor { case bottom }

    private struct TranscriptRevision: Equatable {
        let agentID: UUID
        let lastItem: ChatItem?
        let isRunning: Bool
    }

    private struct TranscriptPosition: Equatable, Sendable {
        let content: CGRect
        let viewportHeight: CGFloat

        var distanceFromBottom: CGFloat { max(0, content.maxY - viewportHeight) }
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
            .task(id: accountSelection) {
                if accountSelection != nil { await model.refreshAccountStatus() }
            }
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
                if accountSelection != nil, case .awaitingSignIn = model.accountStatus {
                    Task { await model.refreshAccountStatus() }
                }
            }
    }

    private var accountSelection: String? {
        guard model.harness != .sai, model.provider.usesAccount else { return nil }
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
            if needsAccountSignIn {
                HarnessAccountSetup()
                    .padding(12)
                    .frame(maxWidth: 820, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.bubble))
                    .padding(.horizontal, 24).padding(.top, 8)
            }
            composer
                .frame(maxWidth: 820)
                .padding(.horizontal, 24).padding(.bottom, 16).padding(.top, 8)
        }
    }

    private var transcript: some View {
        let revision = TranscriptRevision(agentID: agent.id, lastItem: agent.items.last, isRunning: agent.isRunning)
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 0) {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if agent.items.isEmpty { emptyState }
                        ForEach(agent.items) { ChatRow(item: $0).id($0.id) }
                        if agent.isRunning {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Working…").foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 16).padding(.vertical, 10)
                            .background(Capsule().fill(Theme.bubble))
                        }
                    }
                    .padding(.horizontal, 24).padding(.vertical, 16)
                    .frame(maxWidth: 820, alignment: .leading)
                    .frame(maxWidth: .infinity)

                    Color.clear.frame(height: 1).id(TranscriptAnchor.bottom).accessibilityHidden(true)
                }
                .onGeometryChange(for: TranscriptPosition?.self) { geometry in
                    let coordinateSpace = NamedCoordinateSpace.scrollView(axis: .vertical)
                    guard let viewport = geometry.bounds(of: coordinateSpace), viewport.height > 0 else { return nil }
                    return TranscriptPosition(content: geometry.frame(in: coordinateSpace), viewportHeight: viewport.height)
                } action: { position in
                    if let position { trackTranscript(position, proxy: proxy) }
                }
            }
            .overlay(alignment: .bottom) {
                Group {
                    if !followsLatest, revision.lastItem != nil {
                        CircleButton(systemImage: "arrow.down", help: "Jump to latest message") {
                            followsLatest = true
                            withAnimation(reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.88)) {
                                proxy.scrollTo(TranscriptAnchor.bottom, anchor: .bottom)
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
                transcriptPosition = nil
            }
            .task(id: revision) {
                // Text can stream into the last item without changing the number of rows.
                await Task.yield()
                guard !Task.isCancelled, followsLatest, revision.lastItem != nil else { return }
                proxy.scrollTo(TranscriptAnchor.bottom, anchor: .bottom)
            }
        }
    }

    private func trackTranscript(_ position: TranscriptPosition, proxy: ScrollViewProxy) {
        let previous = transcriptPosition
        transcriptPosition = position
        if position.distanceFromBottom <= Self.nearBottom {
            followsLatest = true
        } else if let previous, position.content.minY > previous.content.minY {
            // Growing messages move the end down. A scroll toward older messages moves the top down.
            followsLatest = false
        }

        // The content can resize after the text update, especially when Markdown reflows.
        if followsLatest, position.distanceFromBottom > 1, let previous,
           position.content.height != previous.content.height || position.viewportHeight != previous.viewportHeight {
            proxy.scrollTo(TranscriptAnchor.bottom, anchor: .bottom)
        }
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(spacing: 12) {
            // Centred while there is room; slides left rather than under the Computer button.
            Spacer(minLength: 16)
            modelPill
            Spacer(minLength: 12)
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
        .frame(height: Theme.titleBarHeight + 8)
        .padding(.top, 4)
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
                Section("Sai model") {
                    ForEach(model.saiModels.filter(\.allowed)) { choice in
                        Button { model.saiModel = choice.id } label: {
                            if choice.id == model.saiModel { Label(choice.name, systemImage: "checkmark") } else { Text(choice.name) }
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
                                if name == model.agentModel { Label(name, systemImage: "checkmark") } else { Text(name) }
                            }
                        }
                        Button("Refresh model list") { Task { await model.refreshOllama() } }
                    } else if model.provider == .openRouter {
                        ForEach(openRouterChoices, id: \.self) { name in
                            Button { model.openRouterModel = name } label: {
                                if name == model.openRouterModel { Label(name, systemImage: "checkmark") } else { Text(name) }
                            }
                        }
                        Text("Any other model: right-click the agent → Agent Settings")
                    } else if let notice = model.provider.accountDataNotice {
                        Text("Automatic")
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

    /// When this agent needs the local computer and it isn't on: say so above the composer, with
    /// the one action that unblocks the chat.
    private var computerBanner: AnyView? {
        guard model.agent.config.usesLocalComputer else { return nil }
        switch model.phase {
        case .ready:
            return nil
        case .stopped:
            return AnyView(MemoryBanner())
        case let .booting(step), let .preparing(step):
            return AnyView(bannerRow(symbol: nil, text: step) { EmptyView() })
        case let .failed(message):
            return AnyView(bannerRow(symbol: "exclamationmark.triangle", text: message) {
                Button("Try again") { Task { await model.start() } }
            })
        case .needsSetup:
            return AnyView(bannerRow(symbol: "gearshape", text: "Set up the computer in Settings (the gear, bottom left) to chat.") {
                EmptyView()
            })
        }
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

    private var canSend: Bool {
        model.canChat && !agent.isRunning
            && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField(placeholder, text: $draft, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...6)
                .onSubmit(send)
                .disabled(!model.canChat)
                .padding(.vertical, 9)
            if agent.isRunning {
                roundButton("stop.fill", help: "Stop") { agent.stop() }
            } else {
                roundButton("arrow.up", help: "Send") { send() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!canSend)
                    .opacity(canSend ? 1 : 0.4)
            }
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
        .help(help)
    }

    private var placeholder: String {
        model.chatBlocker ?? (model.harness == .sai
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
                let hosted = model.privacy == .localVM
                Text(model.privacy.location.displayName).font(.title3.weight(.medium))
                Text(hosted
                     ? "The sandbox runs on this Mac. OpenRouter processes model requests."
                     : "The agent works in a Linux VM on this Mac. Open the computer to watch or take over.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
                if hosted, !model.hasOpenRouterKey {
                    Form { KeySetup.openRouter(model) }.formStyle(.grouped).frame(maxWidth: 480, maxHeight: 200)
                }
                if model.phase != .ready, model.files.isPrepared, !isBusy {
                    Button("Turn on the computer") { Task { await model.start() } }
                        .buttonStyle(.borderedProminent)
                        .tint(Theme.accent)
                        .padding(.top, 4)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 80)
    }

    private var isBusy: Bool {
        switch model.phase {
        case .booting, .preparing: true
        default: false
        }
    }

    private func send() {
        guard canSend else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        draft = ""
        followsLatest = true
        model.send(text)
    }
}

struct ChatRow: View {
    let item: ChatItem
    @State private var open = false

    var body: some View {
        switch item.kind {
        case .user:
            HStack {
                Spacer(minLength: 120)
                Text(item.text)
                    .textSelection(.enabled)
                    .foregroundStyle(Theme.onAccent)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(RoundedRectangle(cornerRadius: 20).fill(Theme.accent))
            }
        case .agent:
            ChatMarkdown(markdown: item.text)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
        case let .step(name, done, failed, output):
            VStack(alignment: .leading, spacing: 6) {
                Button { open.toggle() } label: {
                    HStack(spacing: 8) {
                        Image(systemName: failed ? "exclamationmark.circle" : done ? "checkmark.circle" : "circle.dotted")
                            .foregroundStyle(failed ? Color.red : done ? Color.green : Color.secondary)
                        Text(item.text).lineLimit(1).foregroundStyle(.secondary)
                        Text(name).font(.caption).foregroundStyle(.tertiary)
                        Image(systemName: open ? "chevron.down" : "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                    }
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(RoundedRectangle(cornerRadius: 14).stroke(Theme.hairline))
                }
                .buttonStyle(.plain)
                if open, !output.isEmpty {
                    ScrollView {
                        Text(output).font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 180)
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Theme.bubble))
                }
            }
        case let .approval(id, linkOnly, url, answer):
            ApprovalCard(id: id, text: item.text, linkOnly: linkOnly, url: url, answer: answer)
        case let .notice(isError):
            Text(item.text).font(.callout).foregroundStyle(isError ? Color.red : Color.secondary)
                .padding(.horizontal, 4)
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
