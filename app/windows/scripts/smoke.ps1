param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory
)

# Run with Windows PowerShell -STA on a fresh Windows CI desktop. No API key or tasks.
$ErrorActionPreference = 'Stop'
$tokens = $null
$privateVmParseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'setup-private-vm.ps1'), [ref]$tokens, [ref]$privateVmParseErrors)
if ($privateVmParseErrors.Count -gt 0) {
    $privateVmParseErrors | Format-List | Out-Host
    throw 'Private VM provisioning script has parse errors.'
}
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null
$checks = New-Object System.Collections.Generic.List[string]
$process = $null
$root = $null

function Find-Element([string]$id) {
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, $id)
    while ([DateTime]::UtcNow -lt $deadline) {
        $element = $script:root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
        if ($null -ne $element) { return $element }
        Start-Sleep -Milliseconds 200
    }
    throw "UI Automation element not found: $id"
}
function Assert-State([bool]$condition, [string]$message) {
    if (-not $condition) { throw "FAIL: $message" }
    $script:checks.Add($message)
    Write-Host "PASS: $message"
}
function Reveal-Control($element) {
    if ($element.Current.IsOffscreen) {
        # Scroll the dialog to it (focus alone doesn't scroll a ContentDialog), then focus it.
        $scroll = $null
        if ($element.TryGetCurrentPattern([System.Windows.Automation.ScrollItemPattern]::Pattern, [ref]$scroll)) { $scroll.ScrollIntoView() }
        $element.SetFocus()
        for ($attempt = 0; $attempt -lt 20 -and $element.Current.IsOffscreen; $attempt++) {
            Start-Sleep -Milliseconds 100
        }
    }
    return -not $element.Current.IsOffscreen
}
function Capture([string]$name) {
    # Capture the real rendered window, not an HTML reproduction or mock image.
    Start-Sleep -Milliseconds 500
    $bounds = $script:root.Current.BoundingRectangle
    $area = [System.Drawing.Rectangle]::FromLTRB(
        [int][Math]::Max(0, $bounds.Left), [int][Math]::Max(0, $bounds.Top),
        [int][Math]::Min([System.Windows.Forms.SystemInformation]::VirtualScreen.Right, $bounds.Right),
        [int][Math]::Min([System.Windows.Forms.SystemInformation]::VirtualScreen.Bottom, $bounds.Bottom))
    if ($area.Width -le 0 -or $area.Height -le 0) { throw 'No visible desktop bounds for screenshot.' }
    $bitmap = New-Object System.Drawing.Bitmap($area.Width, $area.Height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($area.Location, [System.Drawing.Point]::Empty, $area.Size)
        $bitmap.Save((Join-Path $EvidenceDirectory "$name.png"), [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally { $graphics.Dispose(); $bitmap.Dispose() }
}
try {
    $env:AGENT_S_STARTUP_DIAGNOSTICS = Join-Path $EvidenceDirectory 'startup-error.txt'
    # A fresh install opens on first-launch setup; this run checks the main window, so it starts past setup
    # with its own empty data folder (no conversations, no keys).
    $env:AGENT_S_DATA_DIR = Join-Path $EvidenceDirectory 'data'
    New-Item -ItemType Directory -Force -Path $env:AGENT_S_DATA_DIR | Out-Null
    Set-Content -Path (Join-Path $env:AGENT_S_DATA_DIR 'settings.json') -Value '{"FinishedFirstAgentSetup":true}' -Encoding ascii
    $startedAt = Get-Date
    $process = Start-Process -FilePath $Executable -WorkingDirectory (Split-Path $Executable) -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    while ([DateTime]::UtcNow -lt $deadline) {
        $process.Refresh()
        if ($process.HasExited) { throw "Agent S exited during startup (code $($process.ExitCode))." }
        if ($process.MainWindowHandle -ne [IntPtr]::Zero) {
            $root = [System.Windows.Automation.AutomationElement]::FromHandle($process.MainWindowHandle)
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if ($null -eq $root) { throw 'Agent S did not expose a window within 45 seconds.' }
    $accounts = Find-Element 'AccountsButton'
    Assert-State $accounts.Current.IsEnabled 'Native window starts and Accounts is available'
    Assert-State (-not $accounts.Current.IsOffscreen) 'Accounts is visible inside the desktop work area'
    Assert-State (-not (Find-Element 'SendButton').Current.IsEnabled) 'Send stays disabled without credentials'
    # Like the macOS composer, Stop replaces Send only while a turn runs; when idle it is hidden.
    $stopCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, 'StopButton')
    $stop = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $stopCondition)
    Assert-State ($null -eq $stop -or -not $stop.Current.IsEnabled) 'Stop stays disabled without a running turn'
    $newAgent = Find-Element 'NewAgentButton'
    Assert-State $newAgent.Current.IsEnabled 'New agent is available without a Sai account'
    Capture 'startup'
    $newAgent.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    # With a saved Sai key and no Private computer, the dialog starts on Cloud; choose This PC explicitly.
    $privateTarget = Find-Element 'TargetPrivate'
    $privateTarget.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    # A fresh PC shows the one-button setup checklist; a ready PC shows the local model picker.
    $setupCondition = New-Object System.Windows.Automation.OrCondition(
        (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::AutomationIdProperty, 'PrivateSetup')),
        (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::AutomationIdProperty, 'PrivateModel')))
    $localSetup = $null
    for ($i = 0; $i -lt 75 -and $null -eq $localSetup; $i++) {
        $localSetup = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $setupCondition)
        if ($null -eq $localSetup) { Start-Sleep -Milliseconds 200 }
    }
    Assert-State ($null -ne $localSetup -and (Reveal-Control $localSetup)) 'Private setup or local model is reachable in the dialog'
    Capture 'private-agent'
    $thisPcTarget = Find-Element 'TargetThisPc'
    $thisPcTarget.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    $harness = Find-Element 'PcHarness'
    Assert-State (Reveal-Control $harness) 'This PC has a reachable harness selector'
    $picker = $harness.GetCurrentPattern([System.Windows.Automation.ExpandCollapsePattern]::Pattern)
    $picker.Expand()
    foreach ($harnessName in 'Claude Code', 'Codex', 'DeepSeek') {
        $namedHarness = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty, $harnessName)
        $option = $null
        for ($attempt = 0; $attempt -lt 20 -and $null -eq $option; $attempt++) {
            $option = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $namedHarness)
            if ($null -eq $option) { Start-Sleep -Milliseconds 100 }
        }
        Assert-State ($null -ne $option) "This PC offers $harnessName"
    }
    $picker.Collapse()
    $boxTarget = Find-Element 'TargetBox'
    $boxTarget.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
    Assert-State (Reveal-Control (Find-Element 'BoxName')) 'Simular Box is selectable and its box name field is reachable'
    $cancelCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'Cancel')
    $cancel = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cancelCondition)
    if ($null -eq $cancel) { throw 'New agent dialog has no Cancel control.' }
    $cancel.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 300
    $invoke = $accounts.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $invoke.Invoke()
    # Keys live in Windows Credential Locker: a clean runner shows masked key fields; a PC with saved keys shows "Saved".
    function Find-KeyState([string]$id) {
        $either = New-Object System.Windows.Automation.OrCondition(
            (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::AutomationIdProperty, $id)),
            (New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::AutomationIdProperty, "${id}Saved")))
        for ($i = 0; $i -lt 75; $i++) {
            $found = $script:root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $either)
            if ($null -ne $found) { return $found }
            Start-Sleep -Milliseconds 200
        }
        throw "UI Automation element not found: $id"
    }
    $key = Find-KeyState 'SaiApiKey'
    Assert-State ($key.Current.AutomationId -eq 'SaiApiKeySaved' -or $key.Current.IsPassword) 'Settings shows a masked Sai API-key field (or the saved key)'
    $boxKey = Find-KeyState 'BoxApiKey'
    Assert-State (($boxKey.Current.AutomationId -eq 'BoxApiKeySaved' -or $boxKey.Current.IsPassword) -and (Reveal-Control $boxKey)) 'Settings shows a reachable masked Simular Box key field (or the saved key)'
    Capture 'accounts'
    $cancelCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'Done')
    $cancel = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cancelCondition)
    if ($null -eq $cancel) { throw 'Settings dialog has no Done control.' }
    $cancel.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 300
    Assert-State (Find-Element 'AccountsButton').Current.IsEnabled 'Settings closes back to the native window'
    $null = $process.CloseMainWindow()
    if (-not $process.WaitForExit(10000)) { throw 'Agent S did not close cleanly.' }
    Assert-State ($process.ExitCode -eq 0) 'Native window exits cleanly'
    $report = @{ result = 'passed'; checks = @($checks); revision = $env:GITHUB_SHA; os = [Environment]::OSVersion.VersionString }
    $report | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $EvidenceDirectory 'smoke.json')
}
catch {
    Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $startedAt } -ErrorAction SilentlyContinue |
        Where-Object { $_.ProviderName -in @('Application Error', '.NET Runtime', 'Windows Error Reporting') } |
        Select-Object TimeCreated, ProviderName, Id, Message | Format-List | Out-String |
        Set-Content (Join-Path $EvidenceDirectory 'windows-events.txt')
    if (Test-Path (Join-Path $EvidenceDirectory 'startup-error.txt')) {
        Get-Content (Join-Path $EvidenceDirectory 'startup-error.txt') | Write-Host
    }
    @{ result = 'failed'; error = $_.Exception.Message; checks = @($checks) } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $EvidenceDirectory 'smoke.json')
    throw
}
finally {
    if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}
