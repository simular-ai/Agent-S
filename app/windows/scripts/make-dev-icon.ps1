# Makes the Agent S Dev icon (Assets\AgentS-Dev.ico and .png) from the release icon by adding an
# orange DEV badge, so a development build is easy to tell apart in the taskbar and title bar.
# Run it again after changing AgentS.ico or AgentS.png: .\app\windows\scripts\make-dev-icon.ps1
#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$assets = Join-Path $PSScriptRoot '..\AgentS.Windows\Assets' | Resolve-Path | Select-Object -ExpandProperty ProviderPath
$orange = [System.Drawing.Color]::FromArgb(255, 245, 124, 0)

function Add-Badge([byte[]] $png) {
    $stream = New-Object IO.MemoryStream(, $png)
    $source = [System.Drawing.Image]::FromStream($stream)
    $size = $source.Width
    $bitmap = New-Object System.Drawing.Bitmap($size, $size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bitmap)
    $g.SmoothingMode = 'AntiAlias'
    $g.TextRenderingHint = 'AntiAliasGridFit'
    $g.DrawImage($source, 0, 0, $size, $size)
    $brush = New-Object System.Drawing.SolidBrush($orange)
    if ($size -ge 32) {
        # A banner across the bottom with "DEV".
        $height = [Math]::Round($size * 0.34)
        $g.FillRectangle($brush, 0, $size - $height, $size, $height)
        $font = New-Object System.Drawing.Font('Segoe UI', [float]($height * 0.62), [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
        $format = New-Object System.Drawing.StringFormat
        $format.Alignment = 'Center'; $format.LineAlignment = 'Center'
        $area = New-Object System.Drawing.RectangleF(0, ($size - $height), $size, $height)
        $g.DrawString('DEV', $font, [System.Drawing.Brushes]::White, $area, $format)
        $font.Dispose()
    } else {
        # Too small for text: an orange bar along the bottom.
        $height = [Math]::Max(3, [Math]::Round($size * 0.25))
        $g.FillRectangle($brush, 0, $size - $height, $size, $height)
    }
    $g.Dispose(); $brush.Dispose()
    # Keep the badge inside the rounded tile: take each pixel's opacity from the original.
    $original = New-Object System.Drawing.Bitmap($source)
    for ($y = 0; $y -lt $size; $y++) {
        for ($x = 0; $x -lt $size; $x++) {
            $alpha = $original.GetPixel($x, $y).A
            $pixel = $bitmap.GetPixel($x, $y)
            if ($pixel.A -ne $alpha) { $bitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($alpha, $pixel.R, $pixel.G, $pixel.B)) }
        }
    }
    $original.Dispose(); $source.Dispose()
    $out = New-Object IO.MemoryStream
    $bitmap.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
    return , $out.ToArray()
}

# The PNG logo.
$logo = Add-Badge ([IO.File]::ReadAllBytes((Join-Path $assets 'AgentS.png')))
[IO.File]::WriteAllBytes((Join-Path $assets 'AgentS-Dev.png'), $logo)

# The ICO: every frame of AgentS.ico (PNG-compressed frames) gets the badge.
$ico = [IO.File]::ReadAllBytes((Join-Path $assets 'AgentS.ico'))
$count = [BitConverter]::ToUInt16($ico, 4)
$frames = @()
for ($i = 0; $i -lt $count; $i++) {
    $entry = 6 + 16 * $i
    $length = [BitConverter]::ToInt32($ico, $entry + 8)
    $offset = [BitConverter]::ToInt32($ico, $entry + 12)
    $data = New-Object byte[] $length
    [Array]::Copy($ico, $offset, $data, 0, $length)
    if ($data[0] -ne 0x89 -or $data[1] -ne 0x50) { throw "Frame $i of AgentS.ico is not PNG data." }
    $frames += , @{ Header = $ico[$entry..($entry + 7)]; Data = (Add-Badge $data) }
}
$out = New-Object IO.MemoryStream
$writer = New-Object IO.BinaryWriter($out)
$writer.Write([UInt16]0); $writer.Write([UInt16]1); $writer.Write([UInt16]$count)
$offset = 6 + 16 * $count
foreach ($frame in $frames) {
    $writer.Write([byte[]]$frame.Header)
    $writer.Write([int]$frame.Data.Length)
    $writer.Write([int]$offset)
    $offset += $frame.Data.Length
}
foreach ($frame in $frames) { $writer.Write([byte[]]$frame.Data) }
$writer.Flush()
[IO.File]::WriteAllBytes((Join-Path $assets 'AgentS-Dev.ico'), $out.ToArray())
Write-Host "Wrote AgentS-Dev.ico ($count frames) and AgentS-Dev.png to $assets"
