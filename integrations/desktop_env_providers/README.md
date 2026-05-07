# OSWorld desktop_env providers: KVM and Webtop

These are downstream additions to OSWorld's `desktop_env/providers/` package.
OSWorld lives outside this repo, so this directory ships them as drop-ins.

## Install

```bash
# inside your OSWorld checkout
cp -r kvm    OSWorld/desktop_env/providers/kvm
cp -r webtop OSWorld/desktop_env/providers/webtop
```

Then patch `desktop_env/providers/__init__.py` (the `create_vm_manager_and_provider`
factory) to add two new branches:

```python
elif provider_name == "kvm":
    from desktop_env.providers.kvm.manager import KVMVMManager
    from desktop_env.providers.kvm.provider import KVMProvider
    return KVMVMManager(virsh_uri=region), KVMProvider(region)
elif provider_name == "webtop":
    from desktop_env.providers.webtop.manager import WebtopVMManager
    from desktop_env.providers.webtop.provider import WebtopProvider
    return WebtopVMManager(), WebtopProvider(region)
```

Then run with `--provider_name kvm` or `--provider_name webtop`.

## Notes

### KVM

- `path_to_vm` is interpreted as a libvirt **domain name**, not a disk path.
- `region` (when set) is used as the libvirt connection URI, e.g. `qemu:///system`.
- IP discovery uses `virsh domifaddr --source agent`, so the guest needs
  `qemu-guest-agent` installed.
- Snapshots use libvirt's internal snapshot mechanism (`virsh snapshot-create-as`).
  For external/disk snapshots, swap in `--disk-only --diskspec` as needed.
- Pre-create domains with `virt-install` and register them once via
  `KVMVMManager().add_vm("domain-name")`.

### Webtop

- `path_to_vm` is interpreted as a container image (default
  `ghcr.io/dothyt/webtop:ubuntu-kde`, override with `OSWORLD_WEBTOP_IMAGE`).
- Set `OSWORLD_WEBTOP_RUNTIME=podman` to use podman instead of docker.
- `get_ip_address` returns `localhost:<sidecar>:<sidecar>:<vnc>:<sidecar>`,
  matching the four-port `localhost:server:chromium:vnc:vlc` shape that the
  upstream Docker provider uses. Webtop has no separate chromium/VLC ports,
  so the sidecar is repeated; downstream code that only reads vnc/server is
  unaffected.
- `save_state` does a `docker commit`; `revert_to_snapshot` stops the current
  container and re-runs from that committed image.
