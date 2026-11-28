#Requires -Version 5.1
#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $VhdxPath,
    # Set up the Private computer for development builds (Agent S Dev) instead of the release app.
    [switch] $Dev,
    # The computer's memory, as on macOS. Settings in Agent S can change it later.
    [ValidateSet(2, 4, 6)]
    [int] $MemoryGB = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)

$hostOs = Get-CimInstance Win32_OperatingSystem
if ($hostOs.ProductType -ne 1 -or [int]$hostOs.BuildNumber -lt 22000 -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'Use Windows 11 x64 with the Hyper-V role enabled.'
}
Import-Module Hyper-V

$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$keygen = Join-Path $env:WINDIR 'System32\OpenSSH\ssh-keygen.exe'
if (-not (Test-Path -LiteralPath $ssh -PathType Leaf) -or -not (Test-Path -LiteralPath $keygen -PathType Leaf)) {
    throw 'Install the Windows OpenSSH Client optional feature first.'
}

$source = (Resolve-Path -LiteralPath $VhdxPath).ProviderPath
if ([IO.Path]::GetExtension($source) -ine '.vhdx' -or -not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw 'Supply a trusted, fresh Debian amd64 generic cloud VHDX.'
}
$sourceDisk = Get-VHD -Path $source
if ($sourceDisk.Attached -or $sourceDisk.VhdType -eq 'Differencing') {
    throw 'Supply a detached, standalone Debian cloud VHDX.'
}

# Agent S and Agent S Dev each get their own folder, VM, switch and internal network.
$appName = if ($Dev) { 'Agent S Dev' } else { 'Agent S' }
$subnet = if ($Dev) { '192.168.248' } else { '192.168.247' }
$appDir = Join-Path $env:LOCALAPPDATA "Simular\$appName"
$configPath = Join-Path $appDir 'private-vm.json'
$vmName = "$appName Private Linux"
$switchName = "$appName Internal"
$bootstrapSwitch = 'Default Switch'
$bootstrapNicName = 'Provisioning'
$runtimeNicName = 'Private'
$hostAddress = "$subnet.1"
$guestAddress = "$subnet.2"
$runtimePrefix = "$subnet.0/30"

if ((Test-Path -LiteralPath $configPath) -or (Get-VM -Name $vmName -ErrorAction SilentlyContinue) -or
    (Get-VMSwitch -Name $switchName -ErrorAction SilentlyContinue)) {
    throw 'An Agent S private VM, switch, or configuration already exists. Existing setup was left intact.'
}
if (-not (Get-VMSwitch -Name $bootstrapSwitch -ErrorAction SilentlyContinue)) {
    throw 'Hyper-V Default Switch is needed temporarily to install Debian packages.'
}
if ((Get-NetRoute -DestinationPrefix $runtimePrefix -ErrorAction SilentlyContinue) -or
    (Get-NetIPAddress -IPAddress $hostAddress, $guestAddress -ErrorAction SilentlyContinue)) {
    throw 'The Agent S internal network overlaps an existing interface or route.'
}

function New-Ed25519Key([string] $Path, [string] $Comment) {
    # Start-Process takes a single Windows command line. Explicit quotes preserve the empty password.
    $arguments = '-q -t ed25519 -N "" -C {0} -f "{1}"' -f $Comment, $Path
    $process = Start-Process -FilePath $keygen -ArgumentList $arguments -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw 'Could not generate an SSH key.' }
}

function Remove-InheritedKeyPermissions([string] $Path) {
    $owner = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl = New-Object Security.AccessControl.FileSecurity
    $acl.SetOwner($owner)
    $acl.SetAccessRuleProtection($true, $false)
    $rule = New-Object Security.AccessControl.FileSystemAccessRule($owner, 'FullControl', 'Allow')
    $acl.AddAccessRule($rule)
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Write-SeedDisk([string] $Path, [hashtable] $Files) {
    New-VHD -Path $Path -Dynamic -SizeBytes 128MB | Out-Null
    $mountedDisk = Mount-VHD -Path $Path -Passthru
    try {
        $disk = $mountedDisk | Get-Disk
        Initialize-Disk -Number $disk.Number -PartitionStyle MBR
        $partition = New-Partition -DiskNumber $disk.Number -UseMaximumSize -AssignDriveLetter
        $partition | Format-Volume -FileSystem FAT32 -NewFileSystemLabel CIDATA -Confirm:$false -Force | Out-Null
        $root = '{0}:\' -f $partition.DriveLetter
        foreach ($name in $Files.Keys) {
            [IO.File]::WriteAllText((Join-Path $root $name), $Files[$name], $utf8)
        }
    }
    finally {
        Dismount-VHD -Path $Path
    }
}

function Convert-GuestAsset([string] $Text, [string] $Path, [string] $Permissions) {
    $textLf = $Text.Replace("`r`n", "`n").Replace("`r", "`n")
    $encoded = [Convert]::ToBase64String($utf8.GetBytes($textLf))
    @"
  - path: $Path
    permissions: '$Permissions'
    encoding: b64
    content: $encoded
"@
}

function Format-MacAddress([string] $MacAddress) {
    (([regex]::Matches($MacAddress, '..') | ForEach-Object { $_.Value }) -join ':').ToLowerInvariant()
}

function Wait-PrivateDesktop([int] $TimeoutSeconds) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $savedErrorPreference = $ErrorActionPreference
    try {
        # An expected failed connection is a native-command error in Windows PowerShell.
        $ErrorActionPreference = 'Continue'
        while ([DateTime]::UtcNow -lt $deadline) {
            $png = & $ssh -F $setupSshConfig agent-s-setup '/usr/local/bin/agent-s-guest screenshot' 2>$null
            if ($LASTEXITCODE -eq 0 -and $png -is [string] -and $png.StartsWith('iVBORw0KGgo') -and
                $png.Length -le 4194304) { return }
            Start-Sleep -Seconds 3
        }
    }
    finally {
        $ErrorActionPreference = $savedErrorPreference
    }
    throw 'The private desktop did not become reachable over authenticated SSH.'
}

New-Item -ItemType Directory -Path $appDir -Force | Out-Null
$workDir = Join-Path $appDir ('private-vm-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $workDir | Out-Null

$privateKey = Join-Path $workDir 'client-key'
$knownHosts = Join-Path $workDir 'known_hosts'
$hostKey = Join-Path $workDir 'guest-host-key'
$setupSshConfig = Join-Path $workDir 'setup-ssh-config'
$systemDisk = Join-Path $workDir 'debian.vhdx'
$seedDisk = Join-Path $workDir 'seed.vhdx'
$vm = $null
$switch = $null
$keepForInspection = $false

try {
    Write-Host 'Preparing a private copy of the Debian system disk.'
    Copy-Item -LiteralPath $source -Destination $systemDisk
    if ((Get-VHD -Path $systemDisk).Size -lt 24GB) {
        Resize-VHD -Path $systemDisk -SizeBytes 24GB
    }

    New-Ed25519Key $privateKey 'agent-s-client'
    New-Ed25519Key $hostKey 'agent-s-guest'
    Remove-InheritedKeyPermissions $privateKey
    Remove-InheritedKeyPermissions $hostKey
    $clientPublic = [IO.File]::ReadAllText("$privateKey.pub").Trim()
    $hostPublic = [IO.File]::ReadAllText("$hostKey.pub").Trim()
    $hostPrivate = ([IO.File]::ReadAllLines($hostKey) | ForEach-Object { "    $_" }) -join "`n"
    [IO.File]::WriteAllText($knownHosts, "$guestAddress $hostPublic`n", $utf8)

    $sshIdentity = $privateKey.Replace('\', '/')
    $sshKnownHosts = $knownHosts.Replace('\', '/')
    $sshSettings = @"
Host agent-s-setup
    HostName $guestAddress
    User agent
    BatchMode yes
    RequestTTY no
    IdentitiesOnly yes
    IdentityFile "$sshIdentity"
    UserKnownHostsFile "$sshKnownHosts"
    StrictHostKeyChecking yes
    ConnectTimeout 4
    ServerAliveInterval 3
    ServerAliveCountMax 2
"@
    [IO.File]::WriteAllText($setupSshConfig, $sshSettings + "`n", $utf8)

    Write-Host 'Creating the local Hyper-V VM and internal network.'
    $switch = New-VMSwitch -Name $switchName -SwitchType Internal
    $adapter = $null
    for ($attempt = 0; $attempt -lt 20 -and $null -eq $adapter; $attempt++) {
        $adapter = Get-NetAdapter -Name "vEthernet ($switchName)" -ErrorAction SilentlyContinue
        if ($null -eq $adapter) { Start-Sleep -Milliseconds 500 }
    }
    if ($null -eq $adapter) { throw 'The Hyper-V internal network adapter did not appear.' }
    New-NetIPAddress -InterfaceIndex $adapter.ifIndex -IPAddress $hostAddress -PrefixLength 30 | Out-Null

    $vm = New-VM -Name $vmName -Generation 2 -MemoryStartupBytes ($MemoryGB * 1GB) `
        -Path $workDir -VHDPath $systemDisk -SwitchName $bootstrapSwitch
    Set-VM -VM $vm -ProcessorCount 2 -AutomaticStartAction Nothing -AutomaticStopAction ShutDown `
        -AutomaticCheckpointsEnabled $false -CheckpointType Standard -Notes 'Agent S Private computer'
    Set-VMFirmware -VM $vm -EnableSecureBoot On -SecureBootTemplate MicrosoftUEFICertificateAuthority

    $bootstrapNic = Get-VMNetworkAdapter -VM $vm
    Rename-VMNetworkAdapter -VMNetworkAdapter $bootstrapNic -NewName $bootstrapNicName
    $bootstrapMac = '02' + [Guid]::NewGuid().ToString('N').Substring(0, 10)
    Set-VMNetworkAdapter -VMNetworkAdapter $bootstrapNic -StaticMacAddress $bootstrapMac
    $runtimeMac = '02' + [Guid]::NewGuid().ToString('N').Substring(0, 10)
    Add-VMNetworkAdapter -VM $vm -Name $runtimeNicName -SwitchName $switchName -StaticMacAddress $runtimeMac

    $guestSsh = @"
AddressFamily inet
ListenAddress $guestAddress
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers agent
DisableForwarding yes
PermitTTY no
PermitUserEnvironment no
PermitUserRC no
ForceCommand /usr/local/bin/agent-s-guest
"@
    $desktopUnit = @'
[Unit]
Description=Agent S private desktop

[Service]
User=agent
WorkingDirectory=/home/agent
Environment=HOME=/home/agent
ExecStart=/usr/local/bin/agent-s-desktop
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
'@
    $assets = @(
        (Convert-GuestAsset ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'agent-s-guest.sh'))) '/usr/local/bin/agent-s-guest' '0755')
        (Convert-GuestAsset ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'agent-s-desktop.sh'))) '/usr/local/bin/agent-s-desktop' '0755')
        (Convert-GuestAsset ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'agent-s-demo.html'))) '/usr/local/share/agent-s/demo.html' '0644')
        (Convert-GuestAsset $guestSsh '/etc/ssh/sshd_config.d/00-agent-s.conf' '0644')
        (Convert-GuestAsset $desktopUnit '/etc/systemd/system/agent-s-desktop.service' '0644')
    ) -join "`n"

    $guestData = @"
#cloud-config
users:
  - name: agent
    shell: /bin/bash
    lock_passwd: true
    ssh_authorized_keys:
      - 'restrict,command="/usr/local/bin/agent-s-guest" $clientPublic'
disable_root: true
ssh_pwauth: false
ssh_deletekeys: true
ssh_keys:
  ed25519_private: |
$hostPrivate
  ed25519_public: $hostPublic
package_update: true
packages:
  - openssh-server
  - xvfb
  - xauth
  - x11-xserver-utils
  - openbox
  - chromium
  - xdotool
  - xclip
  - scrot
  - dbus-x11
  - fonts-dejavu-core
write_files:
$assets
runcmd:
  - [ /usr/sbin/sshd, -t ]
  - [ systemctl, restart, ssh.service ]
  - [ systemctl, daemon-reload ]
  - [ systemctl, enable, --now, agent-s-desktop.service ]
"@
    $instanceId = [Guid]::NewGuid().ToString('N')
    $metaData = "instance-id: agent-s-$instanceId`nlocal-hostname: agent-s-private`n"
    $bootstrapMacAddress = Format-MacAddress $bootstrapMac
    $runtimeMacAddress = Format-MacAddress $runtimeMac
    $networkConfig = @"
version: 2
ethernets:
  provisioning:
    match:
      macaddress: "$bootstrapMacAddress"
    set-name: provision0
    dhcp4: true
    optional: true
  private:
    match:
      macaddress: "$runtimeMacAddress"
    set-name: private0
    dhcp4: false
    dhcp6: false
    accept-ra: false
    link-local: []
    addresses: [$guestAddress/30]
"@
    Write-SeedDisk $seedDisk @{
        'user-data' = $guestData + "`n"
        'meta-data' = $metaData
        'network-config' = $networkConfig + "`n"
    }
    $seedDrive = Add-VMHardDiskDrive -VM $vm -Path $seedDisk -ControllerType SCSI -Passthru

    Write-Host 'Installing the guest desktop. This one-time step needs internet access.'
    Start-VM -VM $vm | Out-Null
    Wait-PrivateDesktop 1200

    Write-Host 'Disconnecting the provisioning NIC and verifying the isolated guest.'
    Stop-VM -VM $vm -Confirm:$false
    Get-VMNetworkAdapter -VM $vm -Name $bootstrapNicName | Remove-VMNetworkAdapter
    $runtimeAdapters = @(Get-VMNetworkAdapter -VM $vm)
    if ($runtimeAdapters.Count -ne 1 -or $runtimeAdapters[0].SwitchName -ne $switchName) {
        throw 'Could not confirm that only the private VM adapter remains.'
    }
    Remove-VMHardDiskDrive -VMHardDiskDrive $seedDrive
    Remove-Item -LiteralPath $seedDisk, $hostKey, "$hostKey.pub"
    Start-VM -VM $vm | Out-Null
    Wait-PrivateDesktop 150

    Write-Host 'Saving the clean checkpoint.'
    Stop-VM -VM $vm -Confirm:$false
    Checkpoint-VM -VM $vm -SnapshotName 'Agent S clean' | Out-Null

    $config = @{
        vmId = $vm.Id.ToString()
        address = $guestAddress
        privateKeyPath = $privateKey
        knownHostsPath = $knownHosts
        memoryGB = $MemoryGB
    } | ConvertTo-Json
    $stagedConfig = Join-Path $workDir 'private-vm.json'
    [IO.File]::WriteAllText($stagedConfig, $config + "`n", $utf8)
    Remove-Item -LiteralPath $setupSshConfig
    [IO.File]::Move($stagedConfig, $configPath)
    Write-Host "Private VM configured for this Windows account. Configuration is in $configPath"
}
catch {
    # Keep only this run's VM and its diagnostics for an administrator to inspect.
    # Detach the temporary provisioning NIC so a failed setup does not leave a cloud route.
    if ($null -ne $vm) {
        Stop-VM -VM $vm -TurnOff -Force -ErrorAction SilentlyContinue
        Get-VMNetworkAdapter -VM $vm -ErrorAction SilentlyContinue |
            Where-Object SwitchName -eq $bootstrapSwitch |
            Remove-VMNetworkAdapter -ErrorAction SilentlyContinue
        $keepForInspection = $true
        Write-Warning "Setup stopped. The VM '$vmName' and its files were kept in $workDir for diagnosis."
    }
    throw
}
finally {
    if (-not $keepForInspection -and -not (Test-Path -LiteralPath $configPath)) {
        if ($null -ne $switch) { Remove-VMSwitch -VMSwitch $switch -Force -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $workDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
