using System.Globalization;
using System.Text.RegularExpressions;

namespace AgentS.Core;

/// <summary>
/// Reads the WSL 2 memory limit from %USERPROFILE%\.wslconfig. WSL has one limit for every distro on
/// the PC, so Agent S never changes it; its own computer's 2/4/6 GB cap sits below it. Without a
/// <c>memory=</c> line under <c>[wsl2]</c>, WSL allows half of the PC's memory.
/// </summary>
public static class WslConfig
{
    private static readonly Regex Size = new(@"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)B?\s*$", RegexOptions.IgnoreCase);

    /// <summary>The limit in bytes, and whether it was set in .wslconfig (false: WSL's default of half the PC's memory).</summary>
    public static (ulong Bytes, bool Configured) MemoryLimit(string? wslconfig, ulong totalMemory)
    {
        var fallback = (totalMemory / 2, false);
        if (wslconfig is null) return fallback;
        var section = "";
        ulong? value = null;
        foreach (var raw in wslconfig.Split('\n'))
        {
            var line = raw.Split('#', ';')[0].Trim();
            if (line.StartsWith('[') && line.EndsWith(']')) { section = line[1..^1].Trim(); continue; }
            if (!section.Equals("wsl2", StringComparison.OrdinalIgnoreCase)) continue;
            var equals = line.IndexOf('=');
            if (equals <= 0 || !line[..equals].Trim().Equals("memory", StringComparison.OrdinalIgnoreCase)) continue;
            var match = Size.Match(line[(equals + 1)..]);
            if (!match.Success) continue;
            var number = double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
            var scale = match.Groups[2].Value.ToUpperInvariant() switch
            {
                "K" => 1UL << 10, "M" => 1UL << 20, "G" => 1UL << 30, "T" => 1UL << 40, _ => 1UL,
            };
            value = (ulong)(number * scale);
        }
        return value is > 0 ? (Math.Min(value.Value, totalMemory == 0 ? value.Value : totalMemory), true) : fallback;
    }

    /// <summary>"8 GB", "7.5 GB".</summary>
    public static string FormatGiB(ulong bytes) =>
        (bytes / (double)(1UL << 30)).ToString("0.#", CultureInfo.InvariantCulture) + " GB";
}
