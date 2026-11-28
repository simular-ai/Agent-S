using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>
/// Runs Simular Box turns, as on macOS: the installed Claude Code CLI in the app, with no local tools, and
/// Box's MCP server as its only tool source. The Box is the sandbox, so tool uses don't ask on this PC.
/// </summary>
internal static class BoxRunner
{
    private static string ProfileFolder => Path.Combine(AppIdentity.LocalFolder, "claude-box");

    private static BoxClient? client;
    private static string? clientKey;

    /// <summary>The app's own Box connection (for the screen, live view and wake), renewed when the key changes.</summary>
    public static BoxClient? Client()
    {
        var key = Secrets.ReadBox();
        if (key is null) { client?.Dispose(); client = null; clientKey = null; return null; }
        if (client is null || clientKey != key) { client?.Dispose(); client = new BoxClient(key); clientKey = key; }
        return client;
    }

    public static async IAsyncEnumerable<HarnessEvent> Run(Conversation owner, string prompt, string? resume, string model,
        [EnumeratorCancellation] CancellationToken cancellation)
    {
        var claude = ThisPcRunner.FindClaude() ?? throw new PrivateComputerException("Install Claude Code on this PC first, then send again.");
        var boxKey = Secrets.ReadBox() ?? throw new PrivateComputerException("Add your Simular Box key in Accounts, then send again.");

        string baseUrl, apiKey;
        if (owner.Provider == ModelProvider.OpenRouter)
        {
            if (Secrets.ReadOpenRouter() is null) throw new PrivateComputerException("Add your OpenRouter key in Accounts, then send again.");
            var proxy = ThisPcRunner.OpenRouterProxy();
            (baseUrl, apiKey) = (proxy.BaseUrl, proxy.Token);
        }
        else
        {
            (baseUrl, apiKey) = (HarnessCommand.OllamaUrl, "ollama");
            model = await HarnessRunner.ContextModel(model, cancellation);
        }

        Directory.CreateDirectory(ProfileFolder);
        // The config names the key's variable only; the key itself stays in the environment.
        var mcpConfig = Path.Combine(ProfileFolder, "box-mcp.json");
        await File.WriteAllTextAsync(mcpConfig, Box.McpConfig, cancellation);

        var start = new ProcessStartInfo(claude)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = ProfileFolder,
        };
        foreach (var argument in Box.ClaudeArguments(prompt, resume, mcpConfig, owner.BoxName)) start.ArgumentList.Add(argument);
        start.Environment["CLAUDE_CONFIG_DIR"] = ProfileFolder;
        start.Environment["BOX_TOKEN"] = boxKey;
        if (ThisPcRunner.FindGitBash() is { } bash) start.Environment["CLAUDE_CODE_GIT_BASH_PATH"] = bash;
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
            lock (errors) detail = string.Join(' ', errors).Trim().Replace(boxKey, "•••");
            throw new PrivateComputerException("Claude Code ended without finishing the turn"
                + (detail.Length > 0 ? $": {(detail.Length > 400 ? detail[^400..] : detail)}" : $" (exit code {process.ExitCode})."));
        }
    }
}
