param(
    [Parameter(Mandatory)][string]$Setup,
    [Parameter(Mandatory)][string]$EvidenceDirectory,
    [string]$UpgradeSetup,
    [switch]$Development,
    [switch]$RequireSigned
)

$ErrorActionPreference = 'Stop'
$profile = if ($Development) { 'Agent S Dev' } else { 'Agent S' }
$install = Join-Path $env:RUNNER_TEMP "agent-s-install-$([Guid]::NewGuid())"
$executable = Join-Path $install 'current/AgentS.Windows.exe'
$data = Join-Path $env:LOCALAPPDATA "Simular/$profile"
$marker = Join-Path $data "installer-smoke-$([Guid]::NewGuid()).txt"

function Install([string]$path) {
    $setupProcess = Start-Process -FilePath $path -ArgumentList @('--silent', '--installto', "`"$install`"") -Wait -PassThru
    if ($setupProcess.ExitCode -ne 0 -or !(Test-Path -LiteralPath $executable)) {
        throw "Installer did not produce a runnable executable (exit $($setupProcess.ExitCode))."
    }
}

New-Item -Path $data -ItemType Directory -Force | Out-Null
Set-Content -LiteralPath $marker -Value 'Existing Agent S user data'
try {
    Install $Setup
    if ($UpgradeSetup) {
        $before = Get-Content -LiteralPath (Join-Path $install 'current/sq.version') -Raw
        Install $UpgradeSetup
        $after = Get-Content -LiteralPath (Join-Path $install 'current/sq.version') -Raw
        if ($before -eq $after -or $after -notmatch '<version>0\.0\.2</version>') {
            throw 'The second installer did not upgrade the installed version.'
        }
    }
    if ((Get-Content -LiteralPath $marker -Raw).Trim() -ne 'Existing Agent S user data') {
        throw 'Installer changed the existing user data directory.'
    }
    if ($RequireSigned) {
        foreach ($file in @($executable, (Join-Path $install 'Update.exe'))) {
            $signature = Get-AuthenticodeSignature -LiteralPath $file
            if ($signature.Status -ne 'Valid' -or !$signature.TimeStamperCertificate -or
                $signature.SignerCertificate.Subject -notmatch '(?:^|,\s*)O="Simular, Inc\."(?:,|$)') {
                throw "Invalid or unexpected installed signature: $file"
            }
        }
    }
    & powershell.exe -NoProfile -STA -File (Join-Path $PSScriptRoot 'smoke.ps1') `
        -Executable $executable -EvidenceDirectory $EvidenceDirectory
    if ($LASTEXITCODE -ne 0) { throw 'The installed app failed its startup smoke test.' }
}
finally { Remove-Item -LiteralPath $marker -Force }
