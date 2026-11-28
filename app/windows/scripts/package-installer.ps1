param(
    [Parameter(Mandatory)][string]$PublishDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [string]$SigningMetadata,
    [switch]$Development
)

$ErrorActionPreference = 'Stop'
$packageId = if ($Development) { 'AgentSDev' } else { 'AgentS' }
$title = if ($Development) { 'Agent S Dev' } else { 'Agent S' }
$vpkArguments = @(
    'pack', '--packId', $packageId, '--packTitle', $title, '--packAuthors', 'Simular, Inc.',
    '--packVersion', $Version, '--packDir', $PublishDirectory, '--outputDir', $OutputDirectory,
    '--runtime', 'win-x64', '--channel', 'win', '--mainExe', 'AgentS.Windows.exe',
    '--icon', (Join-Path $PSScriptRoot '../AgentS.Windows/Assets/AgentS.ico'),
    '--shortcuts', 'StartMenuRoot', '--delta', 'None', '--noPortable'
)
if ($SigningMetadata) {
    # The release job already signed the payload. Sign the generated installer, updater and stub.
    $vpkArguments += @('--azureTrustedSignFile', $SigningMetadata, '--signExclude',
        '(?i)[\/\\](?!Squirrel\.exe$|[^\/\\]+_ExecutionStub\.exe$)[^\/\\]+$')
}
& vpk @vpkArguments
if ($LASTEXITCODE -ne 0) { throw "Velopack failed to package Agent S ($LASTEXITCODE)." }

$setups = @(Get-ChildItem -LiteralPath $OutputDirectory -Filter '*Setup.exe' -File)
if ($setups.Count -ne 1) { throw 'Expected exactly one Velopack Setup executable.' }
Rename-Item -LiteralPath $setups[0].FullName -NewName 'AgentS-Windows-x64-Setup.exe'
$feed = Join-Path $OutputDirectory 'releases.win.json'
$packages = @((Get-Content -LiteralPath $feed -Raw | ConvertFrom-Json).Assets |
    Where-Object { $_.Type -in @('Full', 1) -and $_.Version -eq $Version -and $_.PackageId -eq $packageId })
if ($packages.Count -ne 1 -or $packages[0].FileName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]+-full\.nupkg$') {
    throw 'Expected one full package for the requested release.'
}
$package = Join-Path $OutputDirectory $packages[0].FileName
$sha256 = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash
if ($sha256 -ine $packages[0].SHA256) { throw 'The update feed SHA-256 does not match its package.' }

$catalogInput = Join-Path $OutputDirectory 'catalog-input'
New-Item -ItemType Directory -Path $catalogInput | Out-Null
try {
    Copy-Item -LiteralPath $package -Destination $catalogInput
    $catalog = "$package.cat"
    New-FileCatalog -Path $catalogInput -CatalogFilePath $catalog -CatalogVersion 2.0 | Out-Null
    if ((Test-FileCatalog -Path $catalogInput -CatalogFilePath $catalog) -ne 'Valid') {
        throw 'The update catalog does not match the full package.'
    }
}
finally { Remove-Item -LiteralPath $catalogInput -Recurse -Force }
