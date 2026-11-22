# The agent's way to use this PC's apps, through Windows UI Automation (like `mac` on macOS).
#   pc apps                          the windows that are open
#   pc open <app|path|url> [arg]     open an app, a file or a page
#   pc focus <title>                 switch to an open window by part of its title
#   pc tree [--all]                  the front window, each element with [ref=N]
#   pc click <ref>                   click an element
#   pc type <ref> <text> [--enter]   type into a text field (replacing its text), optionally press Enter
#   pc key <keys...>                 press keys, e.g. pc key ctrl+l   pc key enter
#   pc screenshot                    save a screenshot and print its path
# Refs number the elements in tree order, so they stay valid until the window changes.
param([Parameter(Position = 0)][string]$Verb = 'tree', [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest = @())
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Drawing
Add-Type -Namespace AgentS -Name Native -MemberDefinition @'
[DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
[DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
[DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
[DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
[DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
'@
# Physical pixels for UI Automation bounds, the cursor and screenshots on scaled displays.
[void][AgentS.Native]::SetProcessDPIAware()
$A = [System.Windows.Automation.AutomationElement]
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Fail([string]$message, [int]$code = 1) { [Console]::Error.WriteLine($message); exit $code }

# ---- Keys ----------------------------------------------------------------------------------------
$vk = @{ ctrl = 0x11; control = 0x11; alt = 0x12; shift = 0x10; win = 0x5B; windows = 0x5B; cmd = 0x11
    enter = 0x0D; return = 0x0D; tab = 0x09; esc = 0x1B; escape = 0x1B; backspace = 0x08; delete = 0x2E; del = 0x2E
    space = 0x20; up = 0x26; down = 0x28; left = 0x25; right = 0x27; home = 0x24; end = 0x23
    pageup = 0x21; pagedown = 0x22; insert = 0x2D }
for ($i = 1; $i -le 12; $i++) { $vk["f$i"] = 0x6F + $i }
function KeyCode([string]$name) {
    $n = $name.ToLowerInvariant()
    if ($vk.ContainsKey($n)) { return [byte]$vk[$n] }
    if ($n.Length -eq 1 -and $n -match '[a-z0-9]') { return [byte][char]$n.ToUpperInvariant() }
    Fail "Unknown key '$name'. Use letters, digits, ctrl, alt, shift, win, enter, tab, esc, arrows, home, end, pageup, pagedown, f1-f12." 2
}
function Press([string]$combo) {
    $codes = @($combo.Split('+') | Where-Object { $_ } | ForEach-Object { KeyCode $_ })
    foreach ($c in $codes) { [AgentS.Native]::keybd_event($c, 0, 0, [UIntPtr]::Zero) }
    [array]::Reverse($codes)
    foreach ($c in $codes) { [AgentS.Native]::keybd_event($c, 0, 2, [UIntPtr]::Zero) }
    Start-Sleep -Milliseconds 60
}

# ---- Tree ----------------------------------------------------------------------------------------
$actionable = @('Button', 'Hyperlink', 'Edit', 'ComboBox', 'CheckBox', 'RadioButton', 'TabItem', 'MenuItem',
    'ListItem', 'TreeItem', 'Slider', 'Spinner', 'SplitButton', 'Document', 'DataItem')
$maxNodes = 2500

function Front {
    $handle = [AgentS.Native]::GetForegroundWindow()
    if ($handle -eq [IntPtr]::Zero) { Fail 'No window is in front. Open or focus one with pc open or pc focus, then run pc tree.' }
    return $A::FromHandle($handle)
}

# Walks the front window in a fixed order; calls $visit for each element with its ref and depth.
function Walk($root, [scriptblock]$visit) {
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $script:ref = 0
    $stack = New-Object System.Collections.Stack
    $stack.Push(@($root, 0))
    while ($stack.Count -gt 0 -and $script:ref -lt $maxNodes) {
        $item = $stack.Pop(); $element = $item[0]; $depth = $item[1]
        $stop = & $visit $element $script:ref $depth
        $script:ref++
        if ($stop) { return }
        $children = New-Object System.Collections.ArrayList
        try { $child = $walker.GetFirstChild($element) } catch { $child = $null }
        while ($null -ne $child) { [void]$children.Add($child); try { $child = $walker.GetNextSibling($child) } catch { $child = $null } }
        for ($i = $children.Count - 1; $i -ge 0; $i--) { $stack.Push(@($children[$i], ($depth + 1))) }
    }
}

function Describe($element) {
    $current = $element.Current
    $role = $current.ControlType.ProgrammaticName -replace '^ControlType\.', ''
    $name = ($current.Name -replace '\s+', ' ').Trim()
    if ($name.Length -gt 100) { $name = $name.Substring(0, 99) + '…' }
    $value = ''
    try {
        $pattern = $element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
        $value = ($pattern.Current.Value -replace '\s+', ' ').Trim()
        if ($value.Length -gt 60) { $value = $value.Substring(0, 59) + '…' }
    } catch { }
    return @{ Role = $role; Name = $name; Value = $value; Enabled = $current.IsEnabled }
}

function Find([string]$text) {
    $n = 0
    if (-not [int]::TryParse($text, [ref]$n) -or $n -lt 0) { Fail "Expected a ref number from pc tree, got '$text'." 2 }
    $script:found = $null
    Walk (Front) { param($element, $ref, $depth) if ($ref -eq $n) { $script:found = $element; return $true } }
    if ($null -eq $script:found) { Fail "No element $n in the front window now. Run pc tree to get fresh refs." }
    return $script:found
}

function PointerClick($element) {
    $r = $element.Current.BoundingRectangle
    if ($r.IsEmpty -or $r.Width -le 0) { Fail 'FAILED: that element has no position on screen. Nothing was clicked. Run pc tree and pick another.' }
    [void][AgentS.Native]::SetCursorPos([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2))
    [AgentS.Native]::mouse_event(0x02, 0, 0, 0, [UIntPtr]::Zero); [AgentS.Native]::mouse_event(0x04, 0, 0, 0, [UIntPtr]::Zero)
}

function TryPattern($element, $pattern) { try { return $element.GetCurrentPattern($pattern) } catch { return $null } }

# ---- Verbs ---------------------------------------------------------------------------------------
switch ($Verb) {
    'apps' {
        Get-Process | Where-Object { $_.MainWindowTitle } | Sort-Object ProcessName |
            ForEach-Object { "{0}  ({1})" -f $_.MainWindowTitle, $_.ProcessName }
    }
    'open' {
        if ($Rest.Count -lt 1) { Fail 'Usage: pc open <app|path|url> [argument]' 2 }
        try {
            if ($Rest.Count -gt 1) { Start-Process -FilePath $Rest[0] -ArgumentList ($Rest[1..($Rest.Count - 1)] -join ' ') }
            else { Start-Process -FilePath $Rest[0] }
        } catch { Fail "FAILED: could not open $($Rest[0]): $($_.Exception.Message)" }
        Start-Sleep -Milliseconds 1500
        "Opened $($Rest[0]). Run pc tree to see its window."
    }
    'focus' {
        $title = $Rest -join ' '
        if (-not $title) { Fail 'Usage: pc focus <part of a window title>' 2 }
        $process = Get-Process | Where-Object { $_.MainWindowTitle -like "*$title*" } | Select-Object -First 1
        if (-not $process) { Fail "No open window's title contains '$title'. Run pc apps to see them." }
        if ([AgentS.Native]::IsIconic($process.MainWindowHandle)) { [void][AgentS.Native]::ShowWindow($process.MainWindowHandle, 9) }
        # Windows only lets the foreground app hand over focus; a tap of Alt counts as input.
        [AgentS.Native]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero); [AgentS.Native]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
        [void][AgentS.Native]::SetForegroundWindow($process.MainWindowHandle)
        Start-Sleep -Milliseconds 400
        "Focused $($process.MainWindowTitle). Run pc tree to see it."
    }
    'tree' {
        $all = $Rest -contains '--all'
        $root = Front
        $lines = New-Object System.Collections.Generic.List[string]
        $lines.Add("Window: $($root.Current.Name)")
        $script:depthShown = @{}
        Walk $root {
            param($element, $ref, $depth)
            $d = Describe $element
            if ($all -or $d.Name -or $d.Value -or ($actionable -contains $d.Role)) {
                $parts = @(('  ' * [Math]::Min($depth, 12)) + "- $($d.Role)")
                if ($d.Name) { $parts += "`"$($d.Name)`"" }
                if ($d.Value -and $d.Value -ne $d.Name) { $parts += "value=`"$($d.Value)`"" }
                if (-not $d.Enabled) { $parts += '[disabled]' }
                $parts += "[ref=$ref]"
                $lines.Add($parts -join ' ')
            }
        }
        $max = 400
        if ($lines.Count -gt $max) {
            $more = $lines.Count - $max
            $lines = $lines.GetRange(0, $max); $lines.Add("… $more more elements not shown. Act on what is shown, or narrow the window first.")
        }
        $lines -join "`n"
    }
    'click' {
        $element = Find $Rest[0]
        $done = $false
        foreach ($p in @([System.Windows.Automation.InvokePattern]::Pattern, [System.Windows.Automation.TogglePattern]::Pattern,
                         [System.Windows.Automation.SelectionItemPattern]::Pattern, [System.Windows.Automation.ExpandCollapsePattern]::Pattern)) {
            $pattern = TryPattern $element $p
            if ($null -eq $pattern) { continue }
            try {
                switch ($p.ProgrammaticName) {
                    { $_ -like '*Invoke*' } { $pattern.Invoke() }
                    { $_ -like '*Toggle*' } { $pattern.Toggle() }
                    { $_ -like '*SelectionItem*' } { $pattern.Select() }
                    default { if ($pattern.Current.ExpandCollapseState -eq 'Collapsed') { $pattern.Expand() } else { $pattern.Collapse() } }
                }
                $done = $true; break
            } catch { }
        }
        if (-not $done) { PointerClick $element }
        Start-Sleep -Milliseconds 400
        "Clicked ref $($Rest[0]). Run pc tree to see the result."
    }
    'type' {
        if ($Rest.Count -lt 2) { Fail 'Usage: pc type <ref> <text> [--enter]' 2 }
        $element = Find $Rest[0]
        $enter = $Rest -contains '--enter'
        $text = ($Rest[1..($Rest.Count - 1)] | Where-Object { $_ -ne '--enter' }) -join ' '
        $value = TryPattern $element ([System.Windows.Automation.ValuePattern]::Pattern)
        $typed = $false
        if ($null -ne $value -and -not $value.Current.IsReadOnly) { try { $value.SetValue($text); $typed = $true } catch { } }
        if (-not $typed) {
            try { $element.SetFocus() } catch { PointerClick $element }
            Press 'ctrl+a'
            # Paste through the clipboard so any text (Unicode, symbols) arrives as typed.
            Set-Clipboard -Value $text
            Press 'ctrl+v'
        }
        if ($enter) { try { $element.SetFocus() } catch { }; Press 'enter' }
        Start-Sleep -Milliseconds 300
        "Typed into ref $($Rest[0])$(if ($enter) { ' and pressed Enter' }). Run pc tree to see the result."
    }
    'key' {
        if ($Rest.Count -lt 1) { Fail 'Usage: pc key <keys...>   e.g. pc key ctrl+l   pc key enter' 2 }
        foreach ($combo in $Rest) { Press $combo }
        "Pressed $($Rest -join ' ')."
    }
    'screenshot' {
        Add-Type -AssemblyName System.Windows.Forms
        $screen = [System.Windows.Forms.SystemInformation]::VirtualScreen
        $bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.CopyFromScreen($screen.X, $screen.Y, 0, 0, $bitmap.Size)
        $scale = [Math]::Min(1.0, 1600.0 / [Math]::Max($screen.Width, $screen.Height))
        $out = New-Object System.Drawing.Bitmap([int]($screen.Width * $scale), [int]($screen.Height * $scale))
        $g2 = [System.Drawing.Graphics]::FromImage($out)
        $g2.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g2.DrawImage($bitmap, 0, 0, $out.Width, $out.Height)
        $file = Join-Path $here 'agent-s-screen.png'
        $out.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose(); $g2.Dispose(); $bitmap.Dispose(); $out.Dispose()
        $file
    }
    default {
        Fail "Unknown command '$Verb'. Use: pc apps | pc open <app|url> | pc focus <title> | pc tree | pc click <ref> | pc type <ref> <text> [--enter] | pc key <keys> | pc screenshot" 2
    }
}
