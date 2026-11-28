using System.Diagnostics;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Text;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>
/// Runs Claude Code, Codex and DeepSeek inside the WSL Private computer, as on macOS: one process per
/// turn with line-delimited JSON on stdout, the model reached through the Ollama relay.
/// </summary>
internal static class HarnessRunner
{
    private static string WslExe => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "wsl.exe");
    private static string GuestFolder => Path.Combine(AppContext.BaseDirectory, "Guest");
    private static readonly SemaphoreSlim installGate = new(1, 1);
    private static bool installed;
    private static Process? relayProcess;

    /// <summary>Only the WSL computer gives harnesses a shell; the Hyper-V prototype allows desktop actions only.</summary>
    public static bool Supported => PrivateComputer.UsesWsl;

    /// <summary>Forget the install check after Reset computer restores the clean image.</summary>
    public static void Invalidate() { installed = false; StopRelay(); }

    // ---- Install -------------------------------------------------------------------------------

    /// <summary>Installs the harnesses and screen tools once per app run (quick when already installed).</summary>
    public static async Task EnsureInstalled(IProgress<string> progress, CancellationToken cancellation)
    {
        if (installed) return;
        await installGate.WaitAsync(cancellation);
        try
        {
            if (installed) return;
            var script = new StringBuilder("set -e\nmkdir -p /usr/local/lib/sai/new\n");
            foreach (var name in new[] { "agent-s-env.sh", "web", "web.mjs", "ui", "ui.ts", "agent-s-x11-socket.py", "agent-s-ollama-relay.py" })
                script.Append($"echo '{Asset(name)}' | base64 -d > /usr/local/lib/sai/new/{name}\n");
            script.Append($"echo '{Asset("install-harnesses.sh")}' | base64 -d > /tmp/agent-s-install.sh\n");
            script.Append($"bash /tmp/agent-s-install.sh {HarnessCommand.DeepSeekVersion}\n");
            var errors = new Queue<string>();
            var exit = await Stream(["-d", WslComputer.Distro, "-u", "root", "--cd", "/", "--exec", "/bin/bash", "-s"], script.ToString(),
                line => progress.Report(line), line => { errors.Enqueue(line); if (errors.Count > 6) errors.Dequeue(); },
                TimeSpan.FromMinutes(25), cancellation);
            if (exit != 0)
                throw new PrivateComputerException("Installing the coding agents in the Private computer failed. Check your internet connection and try again."
                    + (errors.Count > 0 ? $" ({string.Join(' ', errors).Trim()})" : ""));
            installed = true;
        }
        finally { installGate.Release(); }
    }

    private static string Asset(string name) => Convert.ToBase64String(Encoding.UTF8.GetBytes(
        File.ReadAllText(Path.Combine(GuestFolder, name)).Replace("\r\n", "\n")));

    // ---- Model ---------------------------------------------------------------------------------

    /// <summary>
    /// Returns the larger-context variant of <paramref name="model"/> (see <see cref="HarnessCommand.ContextModel"/>),
    /// creating it in the host's Ollama if needed. Falls back to the model itself if Ollama refuses.
    /// </summary>
    public static async Task<string> ContextModel(string model, CancellationToken cancellation)
    {
        var variant = HarnessCommand.ContextModel(model);
        using var http = new HttpClient(new SocketsHttpHandler { UseProxy = false, AllowAutoRedirect = false })
        { BaseAddress = new Uri("http://127.0.0.1:11434/"), Timeout = TimeSpan.FromMinutes(2) };
        try
        {
            using (var show = await http.PostAsync("api/show", Json(new { model = variant }), cancellation))
                if (show.IsSuccessStatusCode) return variant;
            using var create = await http.PostAsync("api/create",
                Json(new { model = variant, from = model, parameters = new { num_ctx = HarnessCommand.ContextTokens }, stream = false }), cancellation);
            return create.IsSuccessStatusCode ? variant : model;
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException && !cancellation.IsCancellationRequested)
        {
            return model;
        }
    }

    private static StringContent Json(object value) =>
        new(System.Text.Json.JsonSerializer.Serialize(value), Encoding.UTF8, "application/json");

    // ---- Ollama relay --------------------------------------------------------------------------

    /// <summary>Starts the guest's 127.0.0.1:11434 relay to the host's Ollama, if it isn't running.</summary>
    public static async Task EnsureRelay(CancellationToken cancellation)
    {
        if (relayProcess is { HasExited: false }) return;
        // A relay left from an earlier app run would still hold the port.
        await Stream(["-d", WslComputer.Distro, "-u", "agent", "--exec", "pkill", "-f", "agent-s-ollama-relay"], null,
            _ => { }, _ => { }, TimeSpan.FromSeconds(20), cancellation);
        var start = new ProcessStartInfo(WslExe)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (var argument in WslComputer.AsAgent("/usr/local/bin/agent-s-ollama-relay"))
            start.ArgumentList.Add(argument);
        var process = Process.Start(start) ?? throw new PrivateComputerException("Could not start the model relay in WSL.");
        relayProcess = process;
        _ = process.StandardError.BaseStream.CopyToAsync(System.IO.Stream.Null);
        var relay = new OllamaRelay(process.StandardOutput.BaseStream, process.StandardInput.BaseStream, async (guestPort, token) =>
        {
            // Guest 11434 is Ollama; guest 11435 is the app's OpenRouter proxy (which adds the key).
            int hostPort = guestPort == HarnessCommand.OpenRouterGuestPort
                ? ThisPcRunner.OpenRouterProxy().Port
                : 11434;
            var tcp = new TcpClient { NoDelay = true };
            try { await tcp.ConnectAsync("127.0.0.1", hostPort, token); }
            catch { tcp.Dispose(); throw; }
            return new OwnedStream(tcp);
        });
        _ = Task.Run(async () =>
        {
            try { await relay.Run(CancellationToken.None); }
            catch (Exception) { }
            finally { if (relayProcess == process) StopRelay(); }
        });
        // Give the listener a moment to bind before the harness connects.
        await Task.Delay(400, cancellation);
        if (process.HasExited) throw new PrivateComputerException("The model relay in the Private computer stopped. Try Stop computer, then send again.");
    }

    public static void StopRelay()
    {
        var process = relayProcess;
        relayProcess = null;
        if (process is { HasExited: false }) { try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } }
    }

    // ---- Turns ---------------------------------------------------------------------------------

    /// <summary>Runs one turn and yields its events. Cancelling stops the harness's whole process group.</summary>
    public static async IAsyncEnumerable<HarnessEvent> Run(PrivateHarness harness, string prompt, string? resume, string model,
        ModelProvider provider, [EnumeratorCancellation] CancellationToken cancellation)
    {
        // For OpenRouter the guest gets the proxy's token; the key itself never enters the guest.
        string? apiKey = null;
        if (provider == ModelProvider.OpenRouter)
        {
            if (Secrets.ReadOpenRouter() is null) throw new PrivateComputerException("Add your OpenRouter key in Accounts, then send again.");
            apiKey = ThisPcRunner.OpenRouterProxy().Token;
        }
        // setsid makes this bash the leader of a new process group, so Stop can end everything it started.
        var script = ". /usr/local/lib/sai/agent-s-env.sh\n"
            + "mkdir -p ~/work ~/.local/state/agent-s\n"
            + "echo $$ > ~/.local/state/agent-s/turn.pid\n"
            + HarnessCommand.Build(harness, prompt, resume, model, provider, apiKey) + "\n";
        var start = new ProcessStartInfo(WslExe)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8, StandardInputEncoding = new UTF8Encoding(false),
        };
        foreach (var argument in WslComputer.AsAgent("setsid", "--wait", "/bin/bash", "-s"))
            start.ArgumentList.Add(argument);
        using var process = Process.Start(start) ?? throw new PrivateComputerException("Could not start WSL.");
        var errors = new Queue<string>();
        var stderr = Task.Run(async () =>
        {
            while (await process.StandardError.ReadLineAsync() is { } line)
                lock (errors) { errors.Enqueue(line); if (errors.Count > 8) errors.Dequeue(); }
        });
        await process.StandardInput.WriteAsync(script.Replace("\r\n", "\n"));
        process.StandardInput.Close();

        using var stop = cancellation.Register(() => _ = Kill(process));
        bool finished = false;
        while (true)
        {
            string? line;
            try { line = await process.StandardOutput.ReadLineAsync(cancellation); }
            catch (OperationCanceledException) { break; }
            if (line is null) break;
            foreach (var item in HarnessStream.Parse(harness, line))
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
            if (detail.Contains("command not found", StringComparison.Ordinal)) installed = false;
            throw new PrivateComputerException($"{harness.DisplayName()} ended without finishing the turn"
                + (detail.Length > 0 ? $": {(detail.Length > 400 ? detail[^400..] : detail)}" : $" (exit code {process.ExitCode})."));
        }
    }

    private static async Task Kill(Process process)
    {
        try
        {
            await Stream(["-d", WslComputer.Distro, "-u", "agent", "--exec", "/bin/bash", "-c",
                "p=$(cat ~/.local/state/agent-s/turn.pid 2>/dev/null) && kill -TERM -- -\"$p\" 2>/dev/null; true"], null,
                _ => { }, _ => { }, TimeSpan.FromSeconds(15), CancellationToken.None);
        }
        catch (Exception) { }
        if (!process.HasExited) { try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } }
    }

    /// <summary>Runs wsl.exe, streaming output lines; returns the exit code.</summary>
    private static async Task<int> Stream(string[] arguments, string? input, Action<string> output, Action<string> error,
        TimeSpan timeout, CancellationToken cancellation)
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
        using var process = Process.Start(start) ?? throw new PrivateComputerException("WSL isn't available on this PC.");
        try
        {
            var reading = Task.WhenAll(
                Task.Run(async () => { while (await process.StandardOutput.ReadLineAsync(deadline.Token) is { } line) output(line); }),
                Task.Run(async () => { while (await process.StandardError.ReadLineAsync(deadline.Token) is { } line) error(line); }));
            if (input is not null)
            {
                await process.StandardInput.WriteAsync(input.AsMemory(), deadline.Token);
                process.StandardInput.Close();
            }
            await process.WaitForExitAsync(deadline.Token);
            await reading;
            return process.ExitCode;
        }
        catch (OperationCanceledException) when (!cancellation.IsCancellationRequested)
        {
            throw new PrivateComputerException("The Private computer did not respond in time.");
        }
        finally
        {
            if (!process.HasExited) { try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } }
        }
    }

    /// <summary>A TCP connection's stream that also closes its client.</summary>
    private sealed class OwnedStream(TcpClient client) : System.IO.Stream
    {
        private readonly NetworkStream inner = client.GetStream();
        public override bool CanRead => true; public override bool CanSeek => false; public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override Task FlushAsync(CancellationToken cancellationToken) => inner.FlushAsync(cancellationToken);
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => inner.ReadAsync(buffer, cancellationToken);
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default) => inner.WriteAsync(buffer, cancellationToken);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) { inner.Dispose(); client.Dispose(); } base.Dispose(disposing); }
    }
}
