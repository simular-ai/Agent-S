import SwiftUI
import AgentSCore

struct AgentMemoryView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var agent: AgentSession
    @State private var editor: MemoryEditorRequest?
    @State private var sharing: MemoryShareRequest?
    @State private var actionError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            if let error = agent.memoryError {
                Text(error).font(.caption).foregroundStyle(.red)
            }
            if agent.memory.notes.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    Text("No memory yet").font(.callout.weight(.medium))
                    Text("Add a note about work, preferences, decisions, or workflows.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ForEach(AgentMemoryKind.allCases, id: \.self) { kind in
                    let notes = agent.memory.notes(in: kind)
                    if !notes.isEmpty { section(kind, notes: notes) }
                }
            }
        }
        .sheet(item: $editor) { MemoryNoteEditor(request: $0) }
        .confirmationDialog("Share memory?", isPresented: Binding(
            get: { sharing != nil }, set: { if !$0 { sharing = nil } }
        ), presenting: sharing) { request in
            Button("Share") { change { try request.agent.shareMemoryNote(id: request.id, with: request.route) } }
        } message: { request in
            Text("Agent S can include this note in future turns using \(request.route.displayName).")
        }
        .alert("Couldn’t update memory", isPresented: Binding(
            get: { actionError != nil }, set: { if !$0 { actionError = nil } }
        )) {
            Button("OK", role: .cancel) { actionError = nil }
        } message: {
            Text(actionError ?? "")
        }
    }

    private var header: some View {
        HStack {
            Text("Memory").font(.headline).accessibilityAddTraits(.isHeader)
            Spacer()
            Button { edit(nil) } label: { Label("Add note", systemImage: "plus") }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    private func section(_ kind: AgentMemoryKind, notes: [AgentMemoryNote]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(kind.displayName).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                .accessibilityAddTraits(.isHeader)
            VStack(spacing: 0) {
                ForEach(notes) { note in
                    noteRow(note)
                    if note.id != notes.last?.id { Divider().padding(.horizontal, 10) }
                }
            }
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.hairline).allowsHitTesting(false))
        }
    }

    private func noteRow(_ note: AgentMemoryNote) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                if note.isPinned {
                    Image(systemName: "pin.fill").font(.caption).foregroundStyle(.secondary)
                        .accessibilityLabel("Pinned")
                }
                Text((try? AttributedString(markdown: note.text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
                     ?? AttributedString(note.text))
                    .font(.callout).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .layoutPriority(1)
                noteMenu(note)
            }
            HStack(spacing: 8) {
                if note.kind == .activeWork, note.status == .resolved {
                    Label("Resolved", systemImage: "checkmark.circle")
                }
                if let source = note.sourceChatItemID, agent.items.contains(where: { $0.id == source }) {
                    Button { model.revealChatItem(source, in: agent) } label: {
                        Label("Source message", systemImage: "arrow.up.left")
                    }
                    .buttonStyle(.link)
                    .foregroundStyle(.primary)
                }
                Spacer(minLength: 0)
                scopeMenu(note)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func noteMenu(_ note: AgentMemoryNote) -> some View {
        Menu {
            Button("Edit…", systemImage: "pencil") { edit(note) }
            Button(note.isPinned ? "Unpin" : "Pin", systemImage: note.isPinned ? "pin.slash" : "pin") {
                change { try agent.setMemoryPinned(!note.isPinned, id: note.id) }
            }
            if note.kind == .activeWork {
                Button(note.status == .resolved ? "Reopen" : "Mark resolved", systemImage: "checkmark.circle") {
                    change { try agent.setMemoryStatus(note.status == .resolved ? .open : .resolved, id: note.id) }
                }
            }
            Divider()
            Button("Forget", role: .destructive) { change { try agent.forgetMemoryNote(id: note.id) } }
        } label: {
            Image(systemName: "ellipsis").font(.callout)
                .accessibilityLabel("Memory note actions")
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Memory note actions")
    }

    @ViewBuilder private func scopeMenu(_ note: AgentMemoryNote) -> some View {
        let route = model.memoryRoute(for: agent)
        let isPrivate = note.scope == .private
        if !isPrivate || route.allowsAutomaticRecall {
            Menu {
                if !isPrivate {
                    Button("Keep private", systemImage: "lock") {
                        change { try agent.keepMemoryNotePrivate(id: note.id) }
                    }
                }
                if route.allowsAutomaticRecall, note.scope != .route(route) {
                    Button("Share with current route…", systemImage: "arrow.up.right") {
                        sharing = MemoryShareRequest(id: note.id, agent: agent, route: route)
                    }
                }
            } label: {
                scopeLabel(note)
            }
            .menuStyle(.borderlessButton)
            .controlSize(.small)
            .fixedSize()
            .help(note.scope.displayName)
        } else {
            scopeLabel(note).help("Choose a confirmed model and computer route to share this note.")
        }
    }

    @ViewBuilder private func scopeLabel(_ note: AgentMemoryNote) -> some View {
        switch note.scope {
        case .private:
            Label("Private", systemImage: "lock")
        case let .route(route):
            Label(route.allowsAutomaticRecall ? "Shared" : "Route only",
                  systemImage: route.allowsAutomaticRecall ? "arrow.up.right" : "lock")
                .accessibilityLabel(route.allowsAutomaticRecall
                    ? "Shared with \(route.displayName)" : "Saved for \(route.displayName). Recall unavailable.")
        }
    }

    private func edit(_ note: AgentMemoryNote?) {
        editor = MemoryEditorRequest(agent: agent, route: model.memoryRoute(for: agent), note: note)
    }

    private func change(_ action: () throws -> Void) {
        do { try action() } catch { actionError = error.localizedDescription }
    }
}

private struct MemoryShareRequest {
    let id: UUID
    let agent: AgentSession
    let route: AgentMemoryRoute
}

private struct MemoryEditorRequest: Identifiable {
    let id = UUID()
    let agent: AgentSession
    let route: AgentMemoryRoute
    let note: AgentMemoryNote?
}

private struct MemoryNoteEditor: View {
    private enum SharingChoice: Hashable { case keepPrivate, currentRoute, existingRoute }

    @Environment(\.dismiss) private var dismiss
    let request: MemoryEditorRequest
    @State private var kind: AgentMemoryKind
    @State private var text: String
    @State private var sharing: SharingChoice
    @State private var error: String?

    init(request: MemoryEditorRequest) {
        self.request = request
        _kind = State(initialValue: request.note?.kind ?? .activeWork)
        _text = State(initialValue: request.note?.text ?? "")
        let scope = request.note?.scope ?? .private
        _sharing = State(initialValue: scope == .private ? .keepPrivate
            : request.route.allowsAutomaticRecall && scope == .route(request.route) ? .currentRoute : .existingRoute)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(request.note == nil ? "Add memory note" : "Edit memory note").font(.headline)
            Picker("Section", selection: $kind) {
                ForEach(AgentMemoryKind.allCases, id: \.self) { kind in Text(kind.displayName).tag(kind) }
            }
            TextEditor(text: $text)
                .font(.body)
                .accessibilityLabel("Memory note")
                .scrollContentBackground(.hidden)
                .padding(5)
                .frame(height: 112)
                .background(Theme.card, in: RoundedRectangle(cornerRadius: 7))
                .overlay(RoundedRectangle(cornerRadius: 7).stroke(Theme.hairline).allowsHitTesting(false))
            VStack(alignment: .leading, spacing: 5) {
                Picker("Sharing", selection: $sharing) {
                    Text("Private").tag(SharingChoice.keepPrivate)
                    if request.route.allowsAutomaticRecall { Text("Share with current route").tag(SharingChoice.currentRoute) }
                    if keepsOtherRoute {
                        Text("Keep existing route").tag(SharingChoice.existingRoute)
                    }
                }
                Text(sharingDescription).font(.caption).foregroundStyle(.secondary)
            }
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save", action: save)
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 430)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var keepsOtherRoute: Bool {
        guard let note = request.note, note.scope != .private else { return false }
        return !request.route.allowsAutomaticRecall || note.scope != .route(request.route)
    }

    private var sharingDescription: String {
        switch sharing {
        case .currentRoute:
            "Agent S can include this note in future turns using \(request.route.displayName)."
        case .existingRoute:
            request.note?.scope.displayName ?? ""
        case .keepPrivate:
            if let note = request.note, note.scope != .private {
                "Future turns will keep this note private. Earlier conversation history may still contain it."
            } else {
                "Agent S keeps private notes out of model requests."
            }
        }
    }

    private func save() {
        do {
            try request.agent.saveMemoryNote(id: request.note?.id, kind: kind, text: text,
                                             route: sharing == .currentRoute ? request.route : nil,
                                             keepPrivate: sharing == .keepPrivate)
            dismiss()
        } catch {
            self.error = error.localizedDescription
        }
    }
}
