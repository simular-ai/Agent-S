import SwiftUI

/// Pick a login from the Mac's 1Password and type it into the computer's focused field. Logins for
/// the site open in the computer come first. Secrets go straight from 1Password to the computer's
/// keyboard: never into the chat, the agent or the context panel.
struct OnePasswordSheet: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var logins: [OnePassword.Login] = []
    @State private var site: String?
    @State private var query = ""
    @State private var status: String?
    @State private var isError = false
    @State private var loading = true
    @State private var filling = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "key.fill")
                Text("Fill from 1Password").font(.headline)
                Spacer()
                if let site { Text(site).font(.callout).foregroundStyle(.secondary) }
            }
            Text("Click into the field in the computer first (Take over), then pick what to type.")
                .font(.callout).foregroundStyle(.secondary)
            TextField("Search logins", text: $query).textFieldStyle(.roundedBorder)

            Group {
                if loading {
                    HStack { ProgressView().controlSize(.small); Text("Asking 1Password…").foregroundStyle(.secondary) }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if shown.isEmpty {
                    Text(logins.isEmpty ? "No logins." : "No logins match.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(shown) { login in row(login) }
                        .listStyle(.inset)
                }
            }
            .frame(maxHeight: .infinity)

            HStack {
                if let status {
                    Text(status).font(.callout).foregroundStyle(isError ? Color.red : Color.secondary).lineLimit(3)
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
        }
        .padding(16)
        .frame(width: 520, height: 480)
        .task { await load() }
    }

    /// Show which account a login is in when there is more than one.
    private var accountLabels: Bool { Set(logins.compactMap { $0.account }).count > 1 }

    /// Matches for the open site first, then the rest; filtered by the search box.
    private var shown: [OnePassword.Login] {
        let filtered = query.isEmpty ? logins : logins.filter {
            $0.title.localizedCaseInsensitiveContains(query)
                || ($0.username ?? "").localizedCaseInsensitiveContains(query)
                || $0.hosts.contains { $0.localizedCaseInsensitiveContains(query) }
        }
        guard let site else { return filtered }
        return filtered.filter { OnePassword.matches($0, site: site) }
            + filtered.filter { !OnePassword.matches($0, site: site) }
    }

    private func row(_ login: OnePassword.Login) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(login.title)
                    if let site, OnePassword.matches(login, site: site) {
                        Text("this site").font(.caption2).padding(.horizontal, 6).padding(.vertical, 1)
                            .background(Capsule().fill(Theme.bubble))
                    }
                }
                Text([login.username ?? login.hosts.first, accountLabels ? login.account?.label : nil]
                        .compactMap { $0 }.joined(separator: " · "))
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            Button("Username") { fill(login, username: true, password: false) }
            Button("Password") { fill(login, username: false, password: true) }
            Button("Both") { fill(login, username: true, password: true) }
                .help("Types the username, presses Tab, then types the password")
        }
        .disabled(filling)
        .padding(.vertical, 2)
    }

    private func load() async {
        loading = true
        async let siteTask = model.currentSite()
        do {
            logins = try await OnePassword.logins().sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
        } catch {
            say(error.localizedDescription, error: true)
        }
        site = await siteTask
        loading = false
    }

    private func fill(_ login: OnePassword.Login, username: Bool, password: Bool) {
        filling = true
        say("Filling \(login.title)…", error: false)
        Task {
            defer { filling = false }
            do {
                let secret = try await OnePassword.credentials(login)
                if username {
                    guard let value = secret.username else { throw OnePassword.Failure(message: "\(login.title) has no username.") }
                    try await model.typeIntoComputer(value, thenTab: password)
                }
                if password {
                    guard let value = secret.password else { throw OnePassword.Failure(message: "\(login.title) has no password.") }
                    try await model.typeIntoComputer(value)
                }
                say("Filled \(username && password ? "username and password" : username ? "username" : "password") from \(login.title).", error: false)
            } catch {
                say(error.localizedDescription, error: true)
            }
        }
    }

    private func say(_ text: String, error: Bool) {
        status = text
        isError = error
    }
}
