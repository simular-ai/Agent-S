import Foundation

/// The Mac's 1Password, through its command-line tool (`op`) and the app's CLI integration
/// (Touch ID). Agent S uses it to fill sign-in forms in the computer: secrets go from 1Password
/// straight to the computer's keyboard and never into the chat or the model's context.
enum OnePassword {
    struct Login: Identifiable, Hashable {
        let id: String
        let title: String
        let username: String?
        let hosts: [String]
        /// The 1Password account it lives in (the CLI needs it when several are signed in).
        let account: Account?
    }

    struct Account: Hashable {
        let id: String
        let label: String
    }

    /// The accounts signed in to the 1Password app on this Mac.
    static func accounts() async throws -> [Account] {
        let data = try await run(["account", "list", "--format", "json"])
        let list = (try JSONSerialization.jsonObject(with: data) as? [[String: Any]]) ?? []
        return list.compactMap { item in
            guard let id = (item["user_uuid"] ?? item["account_uuid"]) as? String else { return nil }
            let label = (item["email"] as? String).map { "\($0) · \(item["url"] as? String ?? "")" } ?? (item["url"] as? String ?? id)
            return Account(id: id, label: label)
        }
    }

    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// `op`, from Homebrew or the 1Password installer.
    static var tool: URL? {
        ["/opt/homebrew/bin/op", "/usr/local/bin/op", "/usr/bin/op"]
            .map(URL.init(fileURLWithPath:))
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// Every Login item in every signed-in account (1Password asks for Touch ID the first time).
    /// An account that fails is skipped as long as another one answers.
    static func logins() async throws -> [Login] {
        let accounts = (try? await accounts()) ?? []
        if accounts.isEmpty { return try await logins(in: nil) }
        var all: [Login] = []
        var firstError: Error?
        for account in accounts {
            do { all += try await logins(in: account) } catch { firstError = firstError ?? error }
        }
        if all.isEmpty, let firstError { throw firstError }
        return all
    }

    private static func logins(in account: Account?) async throws -> [Login] {
        let data = try await run(["item", "list", "--categories", "Login", "--format", "json"] + accountFlag(account))
        guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return items.compactMap { item in
            guard let id = item["id"] as? String else { return nil }
            let urls = (item["urls"] as? [[String: Any]] ?? []).compactMap { $0["href"] as? String }
            return Login(id: id,
                         title: item["title"] as? String ?? "Untitled",
                         username: item["additional_information"] as? String,
                         hosts: urls.compactMap(host(of:)),
                         account: account)
        }
    }

    private static func accountFlag(_ account: Account?) -> [String] {
        account.map { ["--account", $0.id] } ?? []
    }

    /// The username and password of one item.
    static func credentials(_ login: Login) async throws -> (username: String?, password: String?) {
        let data = try await run(["item", "get", login.id, "--fields", "label=username,label=password", "--reveal",
                                  "--format", "json"] + accountFlag(login.account))
        let object = try JSONSerialization.jsonObject(with: data)
        let fields = (object as? [[String: Any]]) ?? [(object as? [String: Any])].compactMap { $0 }
        func value(_ label: String) -> String? {
            fields.first { ($0["label"] as? String)?.lowercased() == label || ($0["id"] as? String) == label }?["value"] as? String
        }
        return (value("username"), value("password"))
    }

    /// `x.com` from `https://x.com/i/flow/login` or `x.com/home`.
    static func host(of url: String) -> String? {
        let withScheme = url.contains("://") ? url : "https://" + url
        guard let host = URL(string: withScheme)?.host?.lowercased(), !host.isEmpty else { return nil }
        return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
    }

    /// Whether a login belongs to the site: same host, or one is a subdomain of the other.
    static func matches(_ login: Login, site: String) -> Bool {
        login.hosts.contains { $0 == site || $0.hasSuffix("." + site) || site.hasSuffix("." + $0) }
    }

    private static func run(_ arguments: [String]) async throws -> Data {
        guard let tool else {
            throw Failure(message: "1Password's command-line tool isn't installed. Install it with: brew install 1password-cli")
        }
        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = tool
            process.arguments = arguments
            let out = Pipe(), err = Pipe()
            process.standardOutput = out
            process.standardError = err
            process.terminationHandler = { process in
                let data = out.fileHandleForReading.readDataToEndOfFile()
                let message = String(decoding: err.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                if process.terminationStatus == 0 { continuation.resume(returning: data); return }
                let hint = message.contains("not currently signed in") || message.contains("connect")
                    ? " In the 1Password app, turn on Settings → Developer → Integrate with 1Password CLI."
                    : ""
                continuation.resume(throwing: Failure(message: message.trimmingCharacters(in: .whitespacesAndNewlines) + hint))
            }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
    }
}
