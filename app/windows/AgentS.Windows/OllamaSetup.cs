using System.Diagnostics;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text.Json;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>Gets Ollama and a model onto this PC with as few steps as possible.</summary>
internal static class OllamaSetup
{
    /// <summary>The preselected model for new agents, as on macOS.</summary>
    public const string DefaultModel = OllamaModels.Default;
    public static readonly Uri DownloadPage = new("https://ollama.com/download/windows");
    private static readonly Uri Api = new("http://127.0.0.1:11434/");
    private static string AppPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Ollama", "ollama app.exe");

    public static bool IsInstalled => File.Exists(AppPath);

    private static HttpClient Client(TimeSpan timeout) =>
        new(new SocketsHttpHandler { UseProxy = false, AllowAutoRedirect = false }) { BaseAddress = Api, Timeout = timeout };

    /// <summary>Installed models, or null when Ollama isn't answering.</summary>
    public static async Task<string[]?> Models()
    {
        try { using var agent = new LocalAgent(); return await agent.Models(); }
        catch { return null; }
    }

    /// <summary>The local Ollama's models with sizes and whether each runs remotely, or null when it isn't answering.</summary>
    public static async Task<InstalledModel[]?> Installed(TimeSpan? timeout = null)
    {
        try { using var http = Client(timeout ?? TimeSpan.FromSeconds(10)); return OllamaModels.ParseTags(await http.GetStringAsync("api/tags")); }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException) { return null; }
    }

    /// <summary>Ollama's version, or null when it isn't answering.</summary>
    public static async Task<string?> Version(TimeSpan? timeout = null)
    {
        try
        {
            using var http = Client(timeout ?? TimeSpan.FromSeconds(5));
            using var document = JsonDocument.Parse(await http.GetStringAsync("api/version"));
            return document.RootElement.TryGetProperty("version", out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException or JsonException) { return null; }
    }

    public static bool HasModel(string[] installed, string model) =>
        installed.Any(m => OllamaModels.Same(m, model) || (!model.Contains(':') && m.StartsWith(model + ":", StringComparison.OrdinalIgnoreCase)));

    /// <summary>Installs Ollama with winget (no browser, no wizard). Returns false if winget isn't available.</summary>
    public static async Task<bool> Install(CancellationToken cancellation)
    {
        var start = new ProcessStartInfo("winget")
        {
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (var argument in new[] { "install", "--id", "Ollama.Ollama", "--exact", "--silent",
                     "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity" })
            start.ArgumentList.Add(argument);
        try
        {
            using var process = Process.Start(start);
            if (process is null) return false;
            _ = process.StandardOutput.ReadToEndAsync(cancellation);
            _ = process.StandardError.ReadToEndAsync(cancellation);
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            deadline.CancelAfter(TimeSpan.FromMinutes(15));
            await process.WaitForExitAsync(deadline.Token);
            return IsInstalled;
        }
        catch (System.ComponentModel.Win32Exception) { return false; }
        catch (OperationCanceledException) when (!cancellation.IsCancellationRequested) { return IsInstalled; }
    }

    /// <summary>Upgrades Ollama with winget, for a model that needs a newer version. Returns false if winget isn't available.</summary>
    public static async Task<bool> Upgrade(CancellationToken cancellation)
    {
        var start = new ProcessStartInfo("winget") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var argument in new[] { "upgrade", "--id", "Ollama.Ollama", "--exact", "--silent",
                     "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity" })
            start.ArgumentList.Add(argument);
        try
        {
            using var process = Process.Start(start);
            if (process is null) return false;
            _ = process.StandardOutput.ReadToEndAsync(cancellation);
            _ = process.StandardError.ReadToEndAsync(cancellation);
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            deadline.CancelAfter(TimeSpan.FromMinutes(15));
            await process.WaitForExitAsync(deadline.Token);
            return process.ExitCode == 0;
        }
        catch (System.ComponentModel.Win32Exception) { return false; }
        catch (OperationCanceledException) when (!cancellation.IsCancellationRequested) { return false; }
    }

    /// <summary>Starts the installed Ollama app and waits for it to answer.</summary>
    public static async Task<bool> Start(CancellationToken cancellation)
    {
        if (await Models() is not null) return true;
        if (!IsInstalled) return false;
        try { Process.Start(new ProcessStartInfo(AppPath) { UseShellExecute = true, WindowStyle = ProcessWindowStyle.Hidden }); }
        catch (System.ComponentModel.Win32Exception) { return false; }
        for (var i = 0; i < 60; i++)
        {
            await Task.Delay(1000, cancellation);
            if (await Models() is not null) return true;
        }
        return false;
    }

    /// <summary>Downloads a model through Ollama's local API, reporting progress across all its layers.</summary>
    public static async Task Pull(string model, IProgress<(string Step, double? Percent)> progress, CancellationToken cancellation)
    {
        var name = OllamaModels.DisplayName(model);
        using var http = Client(Timeout.InfiniteTimeSpan);
        using var request = new HttpRequestMessage(HttpMethod.Post, "api/pull")
        { Content = JsonContent.Create(new { model, stream = true }) };
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellation);
        if (!response.IsSuccessStatusCode)
            throw new PrivateComputerException($"Ollama couldn't download {name} (HTTP {(int)response.StatusCode}).");
        using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(cancellation));
        var layers = new OllamaPullProgress();
        while (await reader.ReadLineAsync(cancellation) is { } line)
        {
            if (line.Length == 0) continue;
            using var update = JsonDocument.Parse(line);
            var root = update.RootElement;
            if (root.TryGetProperty("error", out var error))
                throw new PrivateComputerException($"Ollama couldn't download {name}: {error.GetString()}");
            var status = root.TryGetProperty("status", out var s) && s.ValueKind == JsonValueKind.String ? s.GetString()! : "";
            double? total = root.TryGetProperty("total", out var t) && t.TryGetDouble(out var tv) ? tv : null;
            double? done = root.TryGetProperty("completed", out var c) && c.TryGetDouble(out var cv) ? cv : null;
            var digest = root.TryGetProperty("digest", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null;
            var fraction = layers.Update(status, digest, total, done);
            progress.Report(fraction is { } f ? ($"Downloading {name}… {Math.Floor(f * 100)}%", f * 100) : ($"Downloading {name}…", null));
            if (status == "success") break;
        }
        // As on macOS: a finished pull counts only once Ollama lists the model.
        progress.Report(("Checking the model in Ollama…", 100));
        if (await Models() is not { } installed || !HasModel(installed, model))
            throw new PrivateComputerException($"Ollama didn't list {name} after the download. Try downloading it again.");
    }

    /// <summary>Available physical memory, or null if Windows doesn't say.</summary>
    public static ulong? AvailableMemory()
    {
        var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
        return GlobalMemoryStatusEx(ref status) ? status.AvailablePhysical : null;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatus
    {
        public uint Length, MemoryLoad;
        public ulong TotalPhysical, AvailablePhysical, TotalPageFile, AvailablePageFile, TotalVirtual, AvailableVirtual, AvailableExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
}
