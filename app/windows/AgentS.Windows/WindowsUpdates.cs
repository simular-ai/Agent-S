using System.Diagnostics;
using System.Text.RegularExpressions;
using Velopack;
using Velopack.Logging;
using Velopack.Sources;

namespace AgentS.Windows;

internal sealed class WindowsUpdates : GithubSource
{
    private const string PackageId = "AgentS";
    private const string Repository = "https://github.com/simular-ai/Agent-S";
    private static readonly Regex PackageName = new(@"\AAgentS-[a-zA-Z0-9._-]+-full\.nupkg\z");

    private WindowsUpdates() : base(Repository, accessToken: null, prerelease: false) { }

    internal static UpdateManager? ForInstalledApp()
    {
        if (!AppIdentity.IsRelease) return null;
        var manager = new UpdateManager(new WindowsUpdates(), new UpdateOptions
        {
            MaximumDeltasBeforeFallback = -1,
        });
        return manager.IsInstalled && !manager.IsPortable && manager.AppId == PackageId ? manager : null;
    }

    public override async Task<VelopackAssetFeed> GetReleaseFeed(IVelopackLogger logger, string? appId,
        string channel, Guid? stagingId = null, VelopackAsset? latestLocalRelease = null)
    {
        var feed = await base.GetReleaseFeed(logger, appId, channel, stagingId, latestLocalRelease);
        return feed with
        {
            Assets = feed.Assets.Where(asset => asset is GitBaseAsset entry
                && asset.PackageId == PackageId && asset.Type == VelopackAssetType.Full
                && asset.FileName is { Length: < 160 } name && PackageName.IsMatch(name)
                && name.StartsWith($"{PackageId}-{asset.Version}-", StringComparison.Ordinal)
                && asset.SHA256 is { Length: 64 } hash && hash.All(Uri.IsHexDigit)
                && entry.Release.Assets.Any(file => file.Name == name + ".cat")).ToArray(),
        };
    }

    public override async Task DownloadReleaseEntry(IVelopackLogger logger, VelopackAsset release,
        string localFile, Action<int> progress, CancellationToken cancelToken)
    {
        if (release is not GitBaseAsset entry || !PackageName.IsMatch(release.FileName))
            throw new InvalidDataException("Invalid Agent S update package.");

        // Validate the entire package before Velopack can extract its updater or any bundled code.
        var staging = Directory.CreateTempSubdirectory("agent-s-update-");
        try
        {
            var payload = staging.CreateSubdirectory("payload");
            var package = Path.Combine(payload.FullName, release.FileName);
            var catalog = Path.Combine(staging.FullName, "update.cat");
            var catalogUrl = GetAssetUrlFromName(entry.Release, release.FileName + ".cat");
            await Downloader.DownloadFile(catalogUrl, catalog, _ => { },
                GetRequestHeaders("application/octet-stream"), cancelToken: cancelToken);
            await base.DownloadReleaseEntry(logger, release, package, progress, cancelToken);
            await VerifyCatalog(payload.FullName, catalog, cancelToken);
            File.Move(package, localFile);
        }
        finally { staging.Delete(recursive: true); }
    }

    private static async Task VerifyCatalog(string payload, string catalog, CancellationToken cancelToken)
    {
        var powerShell = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory,
            "WindowsPowerShell", "v1.0", "powershell.exe"))
        {
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var arg in new[] { "-NoProfile", "-NonInteractive", "-Command" })
            powerShell.ArgumentList.Add(arg);
        powerShell.Environment["AGENT_S_UPDATE_CATALOG"] = catalog;
        powerShell.Environment["AGENT_S_UPDATE_PAYLOAD"] = payload;
        powerShell.ArgumentList.Add("""
            $ErrorActionPreference = 'Stop'
            $signature = Get-AuthenticodeSignature -LiteralPath $env:AGENT_S_UPDATE_CATALOG
            if ($signature.Status -ne 'Valid' -or !$signature.TimeStamperCertificate -or
                $signature.SignerCertificate.Subject -notmatch '(?:^|,\s*)O="Simular, Inc\."(?:,|$)') { exit 1 }
            if ((Test-FileCatalog -Path $env:AGENT_S_UPDATE_PAYLOAD -CatalogFilePath $env:AGENT_S_UPDATE_CATALOG) -ne 'Valid') { exit 1 }
            exit 0
            """);
        using var process = Process.Start(powerShell)
            ?? throw new InvalidOperationException("Could not verify the update signature.");
        try { await process.WaitForExitAsync(cancelToken); }
        catch (OperationCanceledException) { if (!process.HasExited) process.Kill(); throw; }
        if (process.ExitCode != 0) throw new InvalidDataException("Could not verify the signed Agent S update.");
    }
}
