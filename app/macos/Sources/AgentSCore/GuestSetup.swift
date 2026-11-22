import Foundation

/// The guest's first boot: cloud-init `user-data` and `meta-data` for Debian's generic cloud image.
///
/// It makes the `sai` user (the app's SSH key, passwordless sudo), installs a minimal desktop
/// (Xorg + Openbox, logged in automatically on the VM's screen), Chromium, the accessibility bus
/// simulang reads on Linux, Node 22 (simulang needs >= 22.18; Debian 13 ships 20), simulang and
/// Claude Code, and mounts the folder the app shares with the guest.
public enum GuestSetup {
    public static let user = "sai"
    /// The virtio-fs tag of the folder the app shares, and where the guest mounts it.
    public static let shareTag = "sai"
    public static let shareMount = "/mnt/sai"

    /// The guest's hostname; it also names the guest's DHCP lease, which is how the app finds it.
    public static let hostname = "saige"

    public static func metaData(instanceId: String, hostname: String = GuestSetup.hostname) -> String {
        "instance-id: \(instanceId)\nlocal-hostname: \(hostname)\n"
    }

    /// cloud-init `network-config`: DHCP on the virtio NIC, identified by its MAC so macOS records
    /// the lease under that MAC (systemd-networkd otherwise sends a DUID the app can't match).
    public static let networkConfig = """
        version: 2
        ethernets:
          primary:
            match:
              name: "en*"
            dhcp4: true
            dhcp-identifier: mac
        """

    public static func userData(sshPublicKey: String) -> String {
        let key = sshPublicKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return """
        #cloud-config
        users:
          - name: \(user)
            groups: [sudo, audio, video, input]
            shell: /bin/bash
            sudo: "ALL=(ALL) NOPASSWD:ALL"
            lock_passwd: true
            ssh_authorized_keys:
              - \(key)
        ssh_pwauth: false
        package_update: true
        packages:
          - xserver-xorg
          - xinit
          - x11-xserver-utils
          - openbox
          - tint2
          - chromium
          - xdotool
          - scrot
          - spice-vdagent
          - dbus-x11
          - at-spi2-core
          - fonts-noto-core
          - curl
          - ca-certificates
          - git
        write_files:
          - path: /etc/systemd/system/getty@tty1.service.d/autologin.conf
            content: |
              [Service]
              ExecStart=
              ExecStart=-/sbin/agetty --autologin \(user) --noclear %I $TERM
          - path: /home/\(user)/.bash_profile
            owner: \(user):\(user)
            defer: true
            content: |
              [ -f ~/.bashrc ] && . ~/.bashrc
              # The VM's screen: Openbox on tty1, logged in at boot.
              if [ -z "$DISPLAY" ] && [ "$(tty)" = /dev/tty1 ]; then exec startx; fi
          - path: /home/\(user)/.xinitrc
            owner: \(user):\(user)
            defer: true
            content: |
              xset s off -dpms
              # Openbox draws nothing on an empty desktop; a colour shows the screen is alive.
              xsetroot -solid "#3b4252"
              exec dbus-launch --exit-with-session openbox-session
          - path: /home/\(user)/.config/openbox/autostart
            owner: \(user):\(user)
            defer: true
            content: |
              # Openbox has no taskbar of its own.
              tint2 &
              # Shares the clipboard with the Mac (through the app's SPICE agent port).
              spice-vdagent
        mounts:
          - [\(shareTag), \(shareMount), virtiofs, "rw,nofail", "0", "0"]
        runcmd:
          - curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
          - apt-get install -y nodejs
          - npm install -g @simular-ai/simulang @anthropic-ai/claude-code @openai/codex
          - mkdir -p \(shareMount) /home/\(user)/work
          - chown -R \(user):\(user) /home/\(user)/work
          - chown -R \(user):\(user) /home/\(user)/.config
          - systemctl restart getty@tty1
          - touch /var/lib/sai-ready
        """
    }

    /// Present once `runcmd` has finished: the guest is ready for Claude Code.
    public static let readyMarker = "/var/lib/sai-ready"
}
