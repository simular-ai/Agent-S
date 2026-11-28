using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace AgentS.Windows;

/// <summary>One isolated harness process and its line-delimited JSON connection.</summary>
internal sealed class ThisPcRpc : IAsyncDisposable
{
    private readonly Process process;
    private readonly Task stderr;
    private readonly Queue<JsonElement> buffered = new();
    private readonly string harness;
    private int nextId;

    public ThisPcRpc(ProcessStartInfo start, string harness)
    {
        this.harness = harness;
        process = Process.Start(start) ?? throw new PrivateComputerException($"Could not start {harness}.");
        stderr = process.StandardError.BaseStream.CopyToAsync(Stream.Null);
    }

    public static ProcessStartInfo StartInfo(string executable, string workingDirectory, string home)
    {
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false, CreateNoWindow = true, RedirectStandardInput = true,
            RedirectStandardOutput = true, RedirectStandardError = true, WorkingDirectory = workingDirectory,
            StandardInputEncoding = new UTF8Encoding(false), StandardOutputEncoding = Encoding.UTF8,
        };
        start.Environment.Clear();
        foreach (var key in new[] { "SystemRoot", "WINDIR", "ComSpec", "PATHEXT" })
            if (Environment.GetEnvironmentVariable(key) is { } value) start.Environment[key] = value;
        foreach (var key in new[] { "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP" })
            start.Environment[key] = home;
        start.Environment["PATH"] = Path.GetDirectoryName(executable) + ";" + Environment.SystemDirectory;
        return start;
    }

    public async Task<JsonElement> Request(string method, object parameters, CancellationToken token)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
        timeout.CancelAfter(TimeSpan.FromSeconds(45));
        var id = ++nextId;
        try
        {
            await Write(new { jsonrpc = "2.0", id, method, @params = parameters }, timeout.Token);
            while (true)
            {
                var message = await Read(timeout.Token);
                if (message.Get("method").ValueKind != JsonValueKind.Undefined ||
                    !message.Get("id").TryInt(out int response) || response != id)
                {
                    buffered.Enqueue(message);
                    continue;
                }
                if (message.Get("result") is { ValueKind: not JsonValueKind.Undefined } result) return result;
                throw new PrivateComputerException($"{harness} rejected {method}.");
            }
        }
        catch (OperationCanceledException) when (!token.IsCancellationRequested)
        {
            throw new PrivateComputerException($"{harness} did not respond to {method}.");
        }
    }

    public Task<JsonElement> Next(CancellationToken token) =>
        buffered.TryDequeue(out var message) ? Task.FromResult(message) : Read(token);

    private async Task<JsonElement> Read(CancellationToken token)
    {
        while (await process.StandardOutput.ReadLineAsync(token) is { } line)
        {
            try
            {
                using var json = JsonDocument.Parse(line);
                if (json.RootElement.ValueKind == JsonValueKind.Object) return json.RootElement.Clone();
            }
            catch (JsonException) { }
        }
        throw new PrivateComputerException($"{harness} exited before the message finished.");
    }

    public async Task Write(object value, CancellationToken token)
    {
        await process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(value).AsMemory(), token);
        await process.StandardInput.FlushAsync(token);
    }

    public async ValueTask DisposeAsync()
    {
        try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
        try { await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5)); }
        catch (TimeoutException) { }
        try { await stderr.WaitAsync(TimeSpan.FromSeconds(2)); }
        catch (Exception error) when (error is IOException or TimeoutException) { }
        process.Dispose();
    }
}

internal static class ThisPcJson
{
    public static JsonElement Get(this JsonElement value, string key) =>
        value.ValueKind == JsonValueKind.Object && value.TryGetProperty(key, out var child) ? child : default;

    public static string? String(this JsonElement value, string key) =>
        value.Get(key) is { ValueKind: JsonValueKind.String } item ? item.GetString() : null;

    public static bool TryInt(this JsonElement value, out int number)
    {
        number = 0;
        return value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out number);
    }
}
