# Builds the Windows release icon and logo from the macOS production artwork.
# The black development artwork is checked in separately.
#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$sourcePath = (Resolve-Path (Join-Path $PSScriptRoot '..\..\macos\Resources\Icon\AppIcon.png')).Path
$assets = (Resolve-Path (Join-Path $PSScriptRoot '..\AgentS.Windows\Assets')).Path
$source = [Drawing.Bitmap]::new($sourcePath)
try {
    if ($source.Width -ne 1024 -or $source.Height -ne 1024) {
        throw 'The macOS icon dimensions changed. Update the Windows crop for the new artwork.'
    }
    # Crop the transparent macOS canvas to the 826-pixel tile.
    $tile = $source.Clone([Drawing.Rectangle]::new(99, 99, 826, 826), [Drawing.Imaging.PixelFormat]::Format32bppArgb)
}
finally { $source.Dispose() }

function ConvertTo-IconFrame([Drawing.Image]$image, [int]$size) {
    $bitmap = [Drawing.Bitmap]::new($size, $size, [Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $png = [IO.MemoryStream]::new()
    try {
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.CompositingMode = [Drawing.Drawing2D.CompositingMode]::SourceCopy
            $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $graphics.DrawImage($image, 0, 0, $size, $size)
        }
        finally { $graphics.Dispose() }
        $bitmap.Save($png, [Drawing.Imaging.ImageFormat]::Png)
        return ,$png.ToArray()
    }
    finally { $png.Dispose(); $bitmap.Dispose() }
}

$sizes = 16, 20, 24, 32, 40, 48, 64, 128, 256
$frames = [Collections.Generic.List[byte[]]]::new()
try {
    foreach ($size in $sizes) { $frames.Add((ConvertTo-IconFrame $tile $size)) }
}
finally { $tile.Dispose() }

$iconPath = Join-Path $assets 'AgentS.ico'
$writer = [IO.BinaryWriter]::new([IO.File]::Create($iconPath))
try {
    $writer.Write([UInt16]0)
    $writer.Write([UInt16]1)
    $writer.Write([UInt16]$sizes.Count)
    $offset = 6 + 16 * $sizes.Count
    for ($i = 0; $i -lt $sizes.Count; $i++) {
        # In an ICO directory, zero represents a dimension of 256.
        $dimension = [byte]($sizes[$i] % 256)
        $writer.Write($dimension)
        $writer.Write($dimension)
        $writer.Write([UInt16]0)
        $writer.Write([UInt16]1)
        $writer.Write([UInt16]32)
        $writer.Write([int]$frames[$i].Length)
        $writer.Write([int]$offset)
        $offset += $frames[$i].Length
    }
    foreach ($frame in $frames) { $writer.Write($frame) }
}
finally { $writer.Dispose() }

[IO.File]::WriteAllBytes((Join-Path $assets 'AgentS.png'), $frames[$frames.Count - 1])
