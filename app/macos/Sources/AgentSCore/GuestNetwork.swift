import Foundation

/// Finding the guest on the VM's NAT network.
///
/// Virtualization.framework's NAT gives the guest an address from macOS's own DHCP server, which
/// records every lease in `/var/db/dhcpd_leases`. The guest is found there by the MAC address the
/// app gave its network device, or else by its hostname.
public enum GuestNetwork {
    public static let leasesPath = "/var/db/dhcpd_leases"

    /// The newest IPv4 address leased to `mac`, else the newest leased to `hostname`, or nil.
    ///
    /// The file is a list of `{ … }` blocks with `name=`, `ip_address=` and `hw_address=` lines.
    /// A client that sends its MAC is listed as `hw_address=1,<mac>`, with leading zeros dropped
    /// from the octets (`2:4b:…`), so both sides are normalized first. Debian's systemd-networkd
    /// sends a DUID instead (`hw_address=ff,<duid>`), so the hostname the guest is given in its
    /// cloud-init `meta-data` is the fallback.
    public static func address(for mac: String, hostname: String? = nil, inLeases text: String) -> String? {
        let want = normalize(mac)
        var byMAC: String?
        var byName: String?
        var ip: String?
        var hw: String?
        var name: String?
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line == "{" { ip = nil; hw = nil; name = nil }
            else if line.hasPrefix("ip_address=") { ip = String(line.dropFirst("ip_address=".count)) }
            else if line.hasPrefix("name=") { name = String(line.dropFirst("name=".count)) }
            else if line.hasPrefix("hw_address=") {
                let value = line.dropFirst("hw_address=".count)
                hw = value.split(separator: ",", maxSplits: 1).last.map(String.init)
            } else if line == "}", let ip {
                // Newest leases come first in the file; keep the first match of each kind.
                if byMAC == nil, let hw, normalize(hw) == want { byMAC = ip }
                if byName == nil, let hostname, name == hostname { byName = ip }
            }
        }
        return byMAC ?? byName
    }

    /// The host as the guest sees it: the NAT gateway, `.1` on the guest's /24.
    public static func hostAddress(forGuest ip: String) -> String? {
        var parts = ip.split(separator: ".").map(String.init)
        guard parts.count == 4 else { return nil }
        parts[3] = "1"
        return parts.joined(separator: ".")
    }

    static func normalize(_ mac: String) -> String {
        mac.lowercased().split(separator: ":").map { octet in
            let trimmed = octet.drop { $0 == "0" }
            return trimmed.isEmpty ? "0" : String(trimmed)
        }.joined(separator: ":")
    }
}
