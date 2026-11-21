import Foundation
import Security

/// One API key in the login Keychain, under a name a person recognizes when macOS asks
/// ("Agent S · Sai API key"). Keys saved by earlier builds under their old names move over.
/// Each key is read from the Keychain once per launch and then kept in memory, so macOS asks
/// at most once per launch rather than on every request.
struct KeychainKey {
    let name: String
    let legacyService: String

    /// What this launch already read: a key, or nil for "there is none".
    private static var cache: [String: String?] = [:]
    private static let lock = NSLock()

    func read() -> String? {
        Self.lock.lock()
        defer { Self.lock.unlock() }
        if let cached = Self.cache[name] { return cached }
        var value = Self.read(service: name)
        if value == nil, let old = Self.read(service: legacyService) {
            if Self.store(name, old) { Self.delete(service: legacyService) }
            value = old
        }
        Self.cache[name] = .some(value)
        return value
    }

    /// Whether a key is saved, without reading it: asking only for the item's attributes doesn't
    /// need the secret, so macOS doesn't ask. The secret is read when a request needs it.
    var exists: Bool {
        Self.lock.lock()
        defer { Self.lock.unlock() }
        if let cached = Self.cache[name] { return cached != nil }
        return Self.exists(service: name) || Self.exists(service: legacyService)
    }

    private static func exists(service: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
            kSecReturnAttributes as String: true, kSecMatchLimit as String: kSecMatchLimitOne,
            kSecUseAuthenticationUI as String: kSecUseAuthenticationUISkip,
        ]
        var result: AnyObject?
        return SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess
    }

    @discardableResult
    func save(_ key: String) -> Bool {
        Self.lock.lock()
        defer { Self.lock.unlock() }
        let saved = Self.store(name, key)
        Self.cache[name] = .some(saved ? key : nil)
        return saved
    }

    private static func store(_ name: String, _ key: String) -> Bool {
        delete(service: name)
        let item: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: name,
            kSecAttrLabel as String: name,
            kSecAttrAccount as String: "api-key",
            kSecValueData as String: Data(key.utf8),
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        return SecItemAdd(item as CFDictionary, nil) == errSecSuccess
    }

    @discardableResult
    func delete() -> Bool {
        Self.lock.lock()
        defer { Self.lock.unlock() }
        Self.cache[name] = .some(nil)
        let removed = Self.delete(service: name)
        return Self.delete(service: legacyService) || removed
    }

    private static func read(service: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
            kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    private static func delete(service: String) -> Bool {
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service] as CFDictionary) == errSecSuccess
    }
}
