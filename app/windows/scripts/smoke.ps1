param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory
)

# Run with Windows PowerShell -STA on a fresh Windows CI desktop. No API key or tasks.
$ErrorActionPreference = 'Stop'
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
    Assert-State (-not (Find-Element 'SendButton').Current.IsEnabled) 'Send stays disabled without credentials'
    Assert-State (-not (Find-Element 'StopButton').Current.IsEnabled) 'Stop stays disabled without a running turn'
    Assert-State (-not (Find-Element 'NewAgentButton').Current.IsEnabled) 'Agent creation stays disabled without an account'
    Capture 'startup'
    $invoke = $accounts.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $invoke.Invoke()
    $key = Find-Element 'SaiApiKey'
    Assert-State $key.Current.IsPassword 'Accounts opens a masked API-key field'
    Capture 'accounts'
    $cancelCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'Cancel')
    $cancel = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cancelCondition)
    if ($null -eq $cancel) { throw 'Accounts dialog has no Cancel control.' }
    $cancel.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 300
    Assert-State (Find-Element 'AccountsButton').Current.IsEnabled 'Accounts cancels back to the native window'
    $null = $process.CloseMainWindow()
    if (-not $process.WaitForExit(10000)) { throw 'Agent S did not close cleanly.' }
    Assert-State ($process.ExitCode -eq 0) 'Native window exits cleanly'
    $report = @{ result = 'passed'; checks = @($checks); revision = $env:GITHUB_SHA; os = [Environment]::OSVersion.VersionString }
    $report | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $EvidenceDirectory 'smoke.json')
}
catch {
    @{ result = 'failed'; error = $_.Exception.Message; checks = @($checks) } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $EvidenceDirectory 'smoke.json')
    throw
}
finally {
    if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}
