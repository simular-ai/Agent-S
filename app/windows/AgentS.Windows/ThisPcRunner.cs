using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>
/// Runs "This PC" turns, like the macOS app's "This Mac": the installed Claude Code CLI on the PC itself, with
/// Agent S's own profile, the `pc` tool, and an approval hook that asks the person before every tool use.
/// </summary>
internal static class ThisPcRunner
{
    public static readonly Uri ClaudeSetup = new("https://code.claude.com/docs/en/setup");
    public static readonly Uri OpenRouterKeys = new("https://openrouter.ai/settings/keys");

    private static string LocalAppData => Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    /// <summary>No spaces in this path (usually), so the agent can type the tool's path plainly.</summary>
    private static string ToolsFolder => Path.Combine(LocalAppData, "Simular", "agent-s-tools");
    private static string ProfileFolder => Path.Combine(AppIdentity.LocalFolder, "claude-this-pc");
    /// <summary>The `pc` tool as Claude Code's Bash (Git Bash) runs it.</summary>
    public static string PcToolPath => Path.Combine(ToolsFolder, "pc").Replace('\\', '/');
    public static string ScreenshotPath => Path.Combine(ToolsFolder, ThisPcAgent.ScreenshotName);

    private static ApprovalServer? approvals;
    private static OpenRouterProxy? openRouter;

    /// <summary>The app's OpenRouter proxy (started once), shared by This PC turns and the Private computer's relay.</summary>
    public static OpenRouterProxy OpenRouterProxy()
    {
        openRouter ??= new OpenRouterProxy(Secrets.ReadOpenRouter, requiresZdr: () => AppSettings.Current.OpenRouterRequiresZdr);
        openRouter.Start();
        return openRouter;
    }

    /// <summary>Claude Code installed on this PC (npm, winget or the native installer), or null.</summary>
    public static string? FindClaude()
    {
        var names = new[] { "claude.exe", "claude.cmd" };
        var folders = (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Append(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".local", "bin"))
            .Append(Path.Combine(LocalAppData, "Microsoft", "WinGet", "Links"))
            .Append(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "npm"));
        foreach (var folder in folders)
            foreach (var name in names)
            {
                try { var path = Path.Combine(folder.Trim('"'), name); if (File.Exists(path)) return path; }
                catch (ArgumentException) { }
            }
        return null;
    }

    /// <summary>Git Bash, which Claude Code on Windows uses for its Bash tool.</summary>
    public static string? FindGitBash()
    {
        foreach (var root in new[] { Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                                     Path.Combine(LocalAppData, "Programs") })
        {
            var path = Path.Combine(root, "Git", "bin", "bash.exe");
            if (File.Exists(path)) return path;
        }
        return null;
    }

    /// <summary>Starts the approval listener once; <paramref name="decide"/> shows each request to the person.</summary>
    public static void UseApprovals(Func<ApprovalRequest, CancellationToken, Task<bool>> decide)
    {
        if (approvals is not null) return;
        approvals = new ApprovalServer(decide);
        approvals.Start();
    }

    /// <summary>Runs one turn and yields Claude Code's events. Cancelling ends the CLI and everything it started.</summary>
    public static async IAsyncEnumerable<HarnessEvent> Run(Conversation owner, string prompt, string? resume, string model,
        [EnumeratorCancellation] CancellationToken cancellation)
    {
        var claude = FindClaude() ?? throw new PrivateComputerException("Install Claude Code on this PC first, then send again.");
        var bash = FindGitBash() ?? throw new PrivateComputerException("Claude Code on Windows needs Git for Windows (Git Bash). Install it, then send again.");
        if (approvals is null) throw new PrivateComputerException("Approvals aren't ready. Restart Agent S and try again.");
        var tools = InstallTools();

        string baseUrl, apiKey;
        if (owner.Provider == ModelProvider.OpenRouter)
        {
            if (Secrets.ReadOpenRouter() is null) throw new PrivateComputerException("Add your OpenRouter key in Accounts, then send again.");
            var proxy = OpenRouterProxy();
            (baseUrl, apiKey) = (proxy.BaseUrl, proxy.Token);
        }
        else
        {
            (baseUrl, apiKey) = (HarnessCommand.OllamaUrl, "ollama");
            model = await HarnessRunner.ContextModel(model, cancellation);
        }

        Directory.CreateDirectory(ProfileFolder);
        var settings = Path.Combine(ProfileFolder, "agent-s-settings.json");
        await File.WriteAllTextAsync(settings, ThisPcAgent.Settings(
            $"powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"{tools.Approve.Replace('\\', '/')}\""), cancellation);

        var start = new ProcessStartInfo(claude)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        };
        foreach (var argument in ThisPcAgent.ClaudeArguments(prompt, resume, PcToolPath, settings)) start.ArgumentList.Add(argument);
        // Agent S's own Claude Code profile: the person's ~/.claude settings, hooks and credentials stay out of it.
        start.Environment["CLAUDE_CONFIG_DIR"] = ProfileFolder;
        start.Environment["CLAUDE_CODE_GIT_BASH_PATH"] = bash;
        start.Environment["AGENT_S_APPROVAL_PORT"] = approvals.Port.ToString(System.Globalization.CultureInfo.InvariantCulture);
        start.Environment["AGENT_S_APPROVAL_TOKEN"] = approvals.Token;
        start.Environment["AGENT_S_AGENT"] = owner.Id;
        foreach (var name in new[] { "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX" })
            start.Environment.Remove(name);
        foreach (var (name, value) in ThisPcAgent.ModelEnvironment(owner.Provider, baseUrl, apiKey, model)) start.Environment[name] = value;

        using var process = Process.Start(start) ?? throw new PrivateComputerException("Could not start Claude Code.");
        process.StandardInput.Close();
        var errors = new Queue<string>();
        var stderr = Task.Run(async () =>
        {
            while (await process.StandardError.ReadLineAsync() is { } line)
                lock (errors) { errors.Enqueue(line); if (errors.Count > 8) errors.Dequeue(); }
        });
        using var stop = cancellation.Register(() => { try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } });
        bool finished = false;
        while (true)
        {
            string? line;
            try { line = await process.StandardOutput.ReadLineAsync(cancellation); }
            catch (OperationCanceledException) { break; }
            if (line is null) break;
            foreach (var item in HarnessStream.Parse(PrivateHarness.ClaudeCode, line))
            {
                if (item is HarnessFinished) finished = true;
                yield return item;
            }
        }
        cancellation.ThrowIfCancellationRequested();
        await process.WaitForExitAsync(CancellationToken.None);
        await stderr;
        if (!finished)
        {
            string detail;
            lock (errors) detail = string.Join(' ', errors).Trim();
            throw new PrivateComputerException("Claude Code ended without finishing the turn"
                + (detail.Length > 0 ? $": {(detail.Length > 400 ? detail[^400..] : detail)}" : $" (exit code {process.ExitCode})."));
        }
    }

    /// <summary>Writes the `pc` tool and approval hook (refreshed every turn, so updates arrive with the app).</summary>
    private static (string Pc, string Approve) InstallTools()
    {
        Directory.CreateDirectory(ToolsFolder);
        foreach (var name in new[] { "pc", "pc.ps1", "approve.ps1" })
        {
            var text = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "PcTools", name));
            // Git Bash needs LF in the wrapper; PowerShell is fine either way.
            File.WriteAllText(Path.Combine(ToolsFolder, name), name == "pc" ? text.Replace("\r\n", "\n") : text, new UTF8Encoding(name != "pc"));
        }
        return (Path.Combine(ToolsFolder, "pc"), Path.Combine(ToolsFolder, "approve.ps1"));
    }
}
