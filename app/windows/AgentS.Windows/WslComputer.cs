using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace AgentS.Windows;

/// <summary>
/// The Private computer as an app-owned WSL 2 distro. Works on Windows 11 Home and on ARM64.
/// Windows drives are not mounted and Windows programs cannot be launched from the guest.
/// Ollama keeps running on the Windows host; the guest only provides the desktop.
/// </summary>
internal static class WslComputer
{
    public const string Distro = "AgentS-Private";
    private const string DistributionList = "https://raw.githubusercontent.com/microsoft/WSL/master/distributions/DistributionInfo.json";

    private static string Folder => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Simular", "Agent S", "wsl");
    private static string ConfigPath => Path.Combine(Folder, "private-wsl.json");
    private static string CleanImage => Path.Combine(Folder, "clean.tar");
    private static string DiskFolder => Path.Combine(Folder, "disk");
    private static string WslExe => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "wsl.exe");

    private static Process? desktop;

    public static bool IsConfigured => File.Exists(ConfigPath) && File.Exists(CleanImage);

    // ---- Runtime -------------------------------------------------------------------------------

    public static Task<string> Guest(string command, int outputLimit, CancellationToken cancellation) =>
        PrivateComputer.Run(WslExe,
            ["-d", Distro, "-u", "agent", "--cd", "~", "--exec", "/usr/local/bin/agent-s-guest",
             .. command.Split(' ', StringSplitOptions.RemoveEmptyEntries)],
            outputLimit, TimeSpan.FromSeconds(20),
            "Could not reach the Private computer in WSL. Try Stop computer, then send again.", cancellation);

    /// <summary>Starts the guest desktop if it is not already answering.</summary>
    public static void LaunchDesktop()
    {
        if (desktop is { HasExited: false }) return;
        var start = new ProcessStartInfo(WslExe)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true,
        };
        foreach (var argument in new[] { "-d", Distro, "-u", "agent", "--cd", "~", "--exec", "/usr/local/bin/agent-s-desktop" })
            start.ArgumentList.Add(argument);
        // This wsl.exe stays open while the desktop runs; it also keeps the WSL distro alive.
        desktop = Process.Start(start) ?? throw new PrivateComputerException("Could not start WSL.");
        _ = desktop.StandardOutput.BaseStream.CopyToAsync(Stream.Null);
        _ = desktop.StandardError.BaseStream.CopyToAsync(Stream.Null);
    }

    public static async Task Stop(CancellationToken cancellation)
    {
        if (desktop is { HasExited: false })
        {
            try { desktop.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
        }
        desktop = null;
        HarnessRunner.StopRelay();
        await Exec(["--terminate", Distro], TimeSpan.FromSeconds(60), cancellation);
    }

    public static async Task Reset(CancellationToken cancellation)
    {
        await Stop(cancellation);
        // The clean image predates the harness install; the next harness turn installs again.
        HarnessRunner.Invalidate();
        await Exec(["--unregister", Distro], TimeSpan.FromMinutes(2), cancellation);
        Directory.CreateDirectory(DiskFolder);
        await Exec(["--import", Distro, DiskFolder, CleanImage, "--version", "2"], TimeSpan.FromMinutes(10), cancellation, required: true,
            failure: "Could not restore the clean Private computer.");
    }

    // ---- Setup ---------------------------------------------------------------------------------

    public enum Readiness { Ready, NeedsWsl, NeedsRestart }

    /// <summary>
    /// Whether WSL 2 can actually start a VM. `wsl --status` succeeds even when the Virtual Machine Platform
    /// feature isn't active yet, so check for its compute service (vmcompute) and a pending restart instead.
    /// </summary>
    public static Task<Readiness> CheckWsl(CancellationToken cancellation = default)
    {
        if (!File.Exists(WslExe)) return Task.FromResult(Readiness.NeedsWsl);
        if (RegistryKeyExists(@"SYSTEM\CurrentControlSet\Services\vmcompute")) return Task.FromResult(Readiness.Ready);
        return Task.FromResult(RebootPending ? Readiness.NeedsRestart : Readiness.NeedsWsl);
    }

    private static bool RebootPending =>
        RegistryKeyExists(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending");

    private static bool RegistryKeyExists(string path)
    {
        try { using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(path); return key is not null; }
        catch (Exception error) when (error is System.Security.SecurityException or UnauthorizedAccessException) { return false; }
    }

    /// <summary>Asks Windows (with an administrator prompt) to install WSL. A restart usually follows.</summary>
    public static async Task InstallWsl()
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(WslExe, "--install --no-distribution")
            { UseShellExecute = true, Verb = "runas" });
            if (process is not null) await process.WaitForExitAsync();
        }
        catch (System.ComponentModel.Win32Exception)
        {
            throw new PrivateComputerException("WSL was not installed because the administrator prompt was declined.");
        }
    }

    /// <summary>Creates, provisions and snapshots the guest. Reports human-readable progress.</summary>
    public static async Task Setup(IProgress<(string Step, double? Percent)> progress, CancellationToken cancellation)
    {
        Directory.CreateDirectory(Folder);
        var download = Path.Combine(Folder, "debian.wsl");
        try
        {
            progress.Report(("Checking WSL…", null));
            switch (await CheckWsl(cancellation))
            {
                case Readiness.NeedsRestart: throw new PrivateComputerException("Restart your PC to finish turning on Windows' virtual machine feature, then try again.");
                case Readiness.NeedsWsl: throw new PrivateComputerException("Windows' virtual machine feature isn't turned on yet.");
            }

            // Leftovers from an earlier, unfinished setup are app-owned and safe to replace.
            if (File.Exists(ConfigPath)) File.Delete(ConfigPath);
            await Exec(["--unregister", Distro], TimeSpan.FromMinutes(2), cancellation);

            progress.Report(("Finding the official Debian image…", null));
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
            var (url, sha256) = await DebianImage(http, cancellation);

            await Download(http, url, download, sha256, progress, cancellation);

            progress.Report(("Creating the Private computer…", null));
            if (Directory.Exists(DiskFolder)) Directory.Delete(DiskFolder, recursive: true);
            Directory.CreateDirectory(DiskFolder);
            await Exec(["--import", Distro, DiskFolder, download, "--version", "2"], TimeSpan.FromMinutes(10), cancellation,
                required: true, failure: "WSL could not import the Debian image.");
            File.Delete(download);

            progress.Report(("Installing the desktop and browser (this takes a few minutes)…", null));
            await Exec(["-d", Distro, "-u", "root", "--cd", "/", "--exec", "/bin/bash", "-s"], TimeSpan.FromMinutes(30), cancellation,
                required: true, failure: "Installing the guest desktop failed. Check your internet connection and try again.",
                input: ProvisioningScript());

            progress.Report(("Starting the desktop to check it works…", null));
            await Exec(["--terminate", Distro], TimeSpan.FromSeconds(60), cancellation);
            LaunchDesktop();
            await WaitForScreenshot(TimeSpan.FromSeconds(90), cancellation);

            progress.Report(("Saving a clean copy for Reset computer…", null));
            await Stop(cancellation);
            await Exec(["--export", Distro, CleanImage], TimeSpan.FromMinutes(15), cancellation,
                required: true, failure: "Could not save the clean copy of the Private computer.");

            await File.WriteAllTextAsync(ConfigPath,
                JsonSerializer.Serialize(new { distro = Distro, created = DateTimeOffset.UtcNow }), cancellation);
            progress.Report(("Private computer is ready.", 100));
        }
        catch
        {
            try { if (File.Exists(download)) File.Delete(download); } catch (IOException) { }
            throw;
        }
    }

    private static int dockChecked;

    /// <summary>
    /// The dock (tint2), as in the macOS sandbox. Guests set up before it was added (or restored by Reset
    /// computer) get it installed and the current desktop script; then it starts if it isn't running.
    /// Runs in the background once per app run; a failure (e.g. offline) just leaves the desktop without it.
    /// </summary>
    public static void EnsureDock()
    {
        if (Interlocked.Exchange(ref dockChecked, 1) == 1) return;
        _ = Task.Run(async () =>
        {
            try
            {
                var install = new StringBuilder("set -e\ncommand -v tint2 >/dev/null || { apt-get update -q && apt-get install -y -q --no-install-recommends tint2 && apt-get clean; }\n");
                foreach (var (asset, target) in new[] { ("agent-s-desktop.sh", "agent-s-desktop"), ("agent-s-dock.sh", "agent-s-dock") })
                    install.Append($"echo '{Asset(asset)}' | base64 -d > /usr/local/bin/{target}.new && chmod 0755 /usr/local/bin/{target}.new && mv /usr/local/bin/{target}.new /usr/local/bin/{target}\n");
                var (exit, _, _) = await Exec(["-d", Distro, "-u", "root", "--cd", "/", "--exec", "/bin/bash", "-s"], TimeSpan.FromMinutes(5), CancellationToken.None, input: install.ToString());
                if (exit != 0) { dockChecked = 0; return; }
                await Exec(["-d", Distro, "-u", "agent", "--cd", "~", "--exec", "/usr/local/bin/agent-s-dock"], TimeSpan.FromSeconds(30), CancellationToken.None);
            }
            catch (Exception) { dockChecked = 0; }
        });
    }

    public static async Task WaitForScreenshot(TimeSpan timeout, CancellationToken cancellation)
    {
        var start = Stopwatch.StartNew();
        while (start.Elapsed < timeout)
        {
            cancellation.ThrowIfCancellationRequested();
            try
            {
                var png = await Guest("screenshot", 5 * 1024 * 1024, cancellation);
                if (png.StartsWith("iVBORw0KGgo", StringComparison.Ordinal)) return;
            }
            catch (PrivateComputerException) { }
            if (desktop is { HasExited: true })
                throw new PrivateComputerException("The Private computer's desktop stopped unexpectedly.");
            await Task.Delay(1000, cancellation);
        }
        throw new PrivateComputerException("The Private computer started but its desktop did not respond.");
    }

    private static async Task<(string Url, string Sha256)> DebianImage(HttpClient http, CancellationToken cancellation)
    {
        try
        {
            using var document = JsonDocument.Parse(await http.GetStringAsync(DistributionList, cancellation));
            var arch = RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "Arm64Url" : "Amd64Url";
            foreach (var entry in document.RootElement.GetProperty("ModernDistributions").GetProperty("Debian").EnumerateArray())
            {
                if (!entry.TryGetProperty(arch, out var image)) continue;
                var url = image.GetProperty("Url").GetString();
                var sha = image.GetProperty("Sha256").GetString();
                if (url is not null && url.StartsWith("https://", StringComparison.Ordinal) && sha is { Length: 64 })
                    return (url, sha.ToLowerInvariant());
            }
        }
        catch (Exception error) when (error is HttpRequestException or JsonException or KeyNotFoundException or InvalidOperationException) { }
        throw new PrivateComputerException("Could not find the official Debian WSL image. Check your internet connection.");
    }

    private static async Task Download(HttpClient http, string url, string path, string sha256,
        IProgress<(string, double?)> progress, CancellationToken cancellation)
    {
        try
        {
            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellation);
            response.EnsureSuccessStatusCode();
            var total = response.Content.Headers.ContentLength;
            await using var source = await response.Content.ReadAsStreamAsync(cancellation);
            await using var target = File.Create(path);
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var buffer = new byte[1 << 16];
            long done = 0; var lastReport = Stopwatch.StartNew();
            int read;
            while ((read = await source.ReadAsync(buffer, cancellation)) > 0)
            {
                await target.WriteAsync(buffer.AsMemory(0, read), cancellation);
                hash.AppendData(buffer, 0, read);
                done += read;
                if (lastReport.ElapsedMilliseconds > 250)
                {
                    lastReport.Restart();
                    progress.Report(($"Downloading Debian ({done / 1_048_576} MB{(total is > 0 ? $" of {total / 1_048_576} MB" : "")})…",
                        total is > 0 ? 100.0 * done / total.Value : null));
                }
            }
            if (Convert.ToHexStringLower(hash.GetHashAndReset()) != sha256)
                throw new PrivateComputerException("The downloaded Debian image failed its checksum. Try setup again.");
        }
        catch (HttpRequestException)
        {
            throw new PrivateComputerException("Downloading Debian failed. Check your internet connection and try again.");
        }
    }

    private static string Asset(string name) =>
        Convert.ToBase64String(Encoding.UTF8.GetBytes(
            File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Guest", name)).Replace("\r\n", "\n")));

    private static string ProvisioningScript()
    {
        return $$"""
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            # Skip the interactive first-run user prompt; Agent S creates its own locked user.
            rm -f /etc/wsl-distribution.conf
            cat > /etc/wsl.conf <<'EOF'
            [automount]
            enabled=false
            mountFsTab=false
            [interop]
            enabled=false
            appendWindowsPath=false
            [boot]
            systemd=false
            [user]
            default=agent
            EOF
            id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent
            passwd -l agent >/dev/null
            apt-get update
            apt-get install -y --no-install-recommends ca-certificates xvfb xauth x11-xserver-utils openbox tint2 \
                chromium xdotool xclip scrot dbus-x11 fonts-dejavu-core
            apt-get clean
            mkdir -p /etc/agent-s /usr/local/share/agent-s
            # WSLg owns display :0 and its socket folder is read-only, so use :99 on an abstract socket.
            printf '%s\n' 'AGENT_S_DISPLAY=99' 'AGENT_S_XVFB_EXTRA="-nolisten unix"' > /etc/agent-s/env
            echo '{{Asset("agent-s-guest.sh")}}' | base64 -d > /usr/local/bin/agent-s-guest
            echo '{{Asset("agent-s-desktop.sh")}}' | base64 -d > /usr/local/bin/agent-s-desktop
            echo '{{Asset("agent-s-dock.sh")}}' | base64 -d > /usr/local/bin/agent-s-dock
            echo '{{Asset("agent-s-demo.html")}}' | base64 -d > /usr/local/share/agent-s/demo.html
            chmod 0755 /usr/local/bin/agent-s-guest /usr/local/bin/agent-s-desktop /usr/local/bin/agent-s-dock
            chmod 0644 /usr/local/share/agent-s/demo.html /etc/agent-s/env
            """.Replace("\r\n", "\n");
    }

    /// <summary>Runs wsl.exe; returns exit code and output. With <paramref name="required"/>, a failure throws.</summary>
    private static async Task<(int Exit, string Output, string Errors)> Exec(string[] arguments, TimeSpan timeout,
        CancellationToken cancellation, bool required = false, string? failure = null, string? input = null)
    {
        var start = new ProcessStartInfo(WslExe)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = input is not null,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
            StandardInputEncoding = input is null ? null : new UTF8Encoding(false),
        };
        start.Environment["WSL_UTF8"] = "1";
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        deadline.CancelAfter(timeout);
        using var process = new Process { StartInfo = start };
        try { process.Start(); }
        catch (System.ComponentModel.Win32Exception)
        {
            if (required) throw new PrivateComputerException("WSL isn't available on this PC.");
            return (-1, "", "");
        }
        try
        {
            var output = process.StandardOutput.ReadToEndAsync(deadline.Token);
            var errors = process.StandardError.ReadToEndAsync(deadline.Token);
            if (input is not null)
            {
                await process.StandardInput.WriteAsync(input.AsMemory(), deadline.Token);
                process.StandardInput.Close();
            }
            await process.WaitForExitAsync(deadline.Token);
            var result = (process.ExitCode, (await output).Trim(), (await errors).Trim());
            if (required && result.ExitCode != 0)
            {
                var raw = result.Item3.Length > 0 ? result.Item3 : result.Item2;
                if (raw.Contains("HCS_E_SERVICE_NOT_AVAILABLE", StringComparison.OrdinalIgnoreCase) ||
                    raw.Contains("virtualization is not enabled", StringComparison.OrdinalIgnoreCase))
                    throw new PrivateComputerException(RebootPending
                        ? "Restart your PC to finish turning on Windows' virtual machine feature, then try again."
                        : "Windows' virtual machine feature isn't available. Turn on virtualization in your PC's firmware settings, then try again.");
                var detail = string.Join(' ', raw
                    .Split('\n', StringSplitOptions.RemoveEmptyEntries).TakeLast(2)).Trim();
                throw new PrivateComputerException(detail.Length > 0 ? $"{failure} ({detail})" : failure ?? "WSL failed.");
            }
            return result;
        }
        catch (OperationCanceledException) when (!cancellation.IsCancellationRequested)
        {
            throw new PrivateComputerException(failure ?? "WSL did not respond in time.");
        }
        finally
        {
            if (!process.HasExited) { try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } }
        }
    }
}
