using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>Gets Ollama and a vision model onto this PC with as few steps as possible.</summary>
internal static class OllamaSetup
{
    public const string DefaultModel = "gemma4";
    public static readonly Uri DownloadPage = new("https://ollama.com/download/windows");
    private static readonly Uri Api = new("http://127.0.0.1:11434/");
    private static string AppPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Ollama", "ollama app.exe");

    public static bool IsInstalled => File.Exists(AppPath);

    /// <summary>Installed models, or null when Ollama isn't answering.</summary>
    public static async Task<string[]?> Models()
    {
        try { using var agent = new LocalAgent(); return await agent.Models(); }
        catch { return null; }
    }

    public static bool HasModel(string[] installed, string model) =>
        installed.Any(m => m.Equals(model, StringComparison.OrdinalIgnoreCase) || m.StartsWith(model + ":", StringComparison.OrdinalIgnoreCase));

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

    /// <summary>Downloads a model through Ollama's local API, reporting progress.</summary>
    public static async Task Pull(string model, IProgress<(string Step, double? Percent)> progress, CancellationToken cancellation)
    {
        using var http = new HttpClient(new SocketsHttpHandler { UseProxy = false, AllowAutoRedirect = false })
        { BaseAddress = Api, Timeout = Timeout.InfiniteTimeSpan };
        using var request = new HttpRequestMessage(HttpMethod.Post, "api/pull")
        { Content = JsonContent.Create(new { model, stream = true }) };
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellation);
        if (!response.IsSuccessStatusCode)
            throw new PrivateComputerException($"Ollama couldn't download {model}.");
        using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(cancellation));
        while (await reader.ReadLineAsync(cancellation) is { } line)
        {
            if (line.Length == 0) continue;
            using var update = JsonDocument.Parse(line);
            var root = update.RootElement;
            if (root.TryGetProperty("error", out var error))
                throw new PrivateComputerException($"Ollama couldn't download {model}: {error.GetString()}");
            long total = root.TryGetProperty("total", out var t) && t.TryGetInt64(out var tv) ? tv : 0;
            long done = root.TryGetProperty("completed", out var c) && c.TryGetInt64(out var cv) ? cv : 0;
            if (total > 0)
                progress.Report(($"Downloading {model} ({done / 1e9:0.0} of {total / 1e9:0.0} GB)…", 100.0 * done / total));
            else
                progress.Report(($"Downloading {model}…", null));
        }
    }
}
