using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace AgentS.Core;

/// <summary>One HTTP/1.1 request read from a loopback connection (headers and a Content-Length body).</summary>
public sealed record LoopbackRequest(string Method, string Path, IReadOnlyList<(string Name, string Value)> Headers, byte[] Body)
{
    public string? Header(string name) => Headers.FirstOrDefault(h => h.Name.Equals(name, StringComparison.OrdinalIgnoreCase)).Value;

    /// <summary>The bearer token or x-api-key the client presented.</summary>
    public string? Token => Header("x-api-key") ?? (Header("authorization") is { } auth && auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? auth[7..].Trim() : null);

    public static async Task<LoopbackRequest?> Read(Stream stream, int maxBytes, CancellationToken cancellation)
    {
        var buffer = new MemoryStream();
        var chunk = new byte[16 * 1024];
        int headerEnd = -1;
        while (headerEnd < 0)
        {
            int read = await stream.ReadAsync(chunk, cancellation);
            if (read == 0) return null;
            buffer.Write(chunk, 0, read);
            if (buffer.Length > 64 * 1024 && IndexOf(buffer, "\r\n\r\n"u8) < 0) return null;
            headerEnd = IndexOf(buffer, "\r\n\r\n"u8);
        }
        var bytes = buffer.ToArray();
        var head = Encoding.ASCII.GetString(bytes, 0, headerEnd).Split("\r\n");
        var start = head[0].Split(' ');
        if (start.Length < 2) return null;
        var headers = head.Skip(1).Select(line => line.Split(':', 2)).Where(p => p.Length == 2)
            .Select(p => (p[0].Trim(), p[1].Trim())).ToList();
        int length = int.TryParse(headers.FirstOrDefault(h => h.Item1.Equals("content-length", StringComparison.OrdinalIgnoreCase)).Item2, out var n) ? n : 0;
        if (length < 0 || length > maxBytes) return null;
        var body = new MemoryStream();
        body.Write(bytes, headerEnd + 4, bytes.Length - headerEnd - 4);
        while (body.Length < length)
        {
            int read = await stream.ReadAsync(chunk, cancellation);
            if (read == 0) return null;
            body.Write(chunk, 0, read);
        }
        return new LoopbackRequest(start[0], start[1], headers, body.ToArray()[..length]);
    }

    private static int IndexOf(MemoryStream stream, ReadOnlySpan<byte> pattern) =>
        stream.GetBuffer().AsSpan(0, (int)stream.Length).IndexOf(pattern);

    public static async Task Respond(Stream stream, int status, string body, string type = "text/plain", CancellationToken cancellation = default)
    {
        var bytes = Encoding.UTF8.GetBytes(body);
        var head = $"HTTP/1.1 {status} {(HttpStatusCode)status}\r\nContent-Type: {type}\r\nContent-Length: {bytes.Length}\r\nConnection: close\r\n\r\n";
        await stream.WriteAsync(Encoding.ASCII.GetBytes(head), cancellation);
        await stream.WriteAsync(bytes, cancellation);
        await stream.FlushAsync(cancellation);
    }
}

/// <summary>A loopback TCP listener on a free port that serves each connection with <see cref="Serve"/>.</summary>
public abstract class LoopbackServer : IDisposable
{
    private readonly TcpListener listener = new(IPAddress.Loopback, 0);
    private readonly CancellationTokenSource stop = new();

    /// <summary>Required on every request, so other programs on the PC (or web pages calling localhost) can't use it.</summary>
    public string Token { get; } = "agent-s-" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(24));
    public int Port { get; private set; }

    public void Start()
    {
        if (Port != 0) return;
        listener.Start();
        Port = ((IPEndPoint)listener.LocalEndpoint).Port;
        _ = Task.Run(async () =>
        {
            while (!stop.IsCancellationRequested)
            {
                TcpClient client;
                try { client = await listener.AcceptTcpClientAsync(stop.Token); }
                catch (Exception) { return; }
                _ = Task.Run(async () =>
                {
                    using (client)
                    {
                        try { await Serve(client.GetStream(), stop.Token); }
                        catch (Exception) { /* One failed request never stops the server. */ }
                    }
                });
            }
        });
    }

    protected abstract Task Serve(NetworkStream stream, CancellationToken cancellation);

    public void Dispose()
    {
        stop.Cancel();
        listener.Stop();
        GC.SuppressFinalize(this);
    }
}

/// <summary>One tool use a "This PC" agent wants to make, waiting for the person.</summary>
public sealed record ApprovalRequest(string Agent, string Tool, JsonElement Input);

/// <summary>
/// Listens on the PC's loopback for Claude Code's PreToolUse hook. Each request is one tool use; the
/// answer ("allow" or "deny") comes from <c>decide</c>, which shows it to the person.
/// </summary>
public sealed class ApprovalServer(Func<ApprovalRequest, CancellationToken, Task<bool>> decide) : LoopbackServer
{
    protected override async Task Serve(NetworkStream stream, CancellationToken cancellation)
    {
        var request = await LoopbackRequest.Read(stream, 4 << 20, cancellation);
        if (request is null) return;
        if (request.Method != "POST" || request.Path != "/approve" || request.Token != Token)
        {
            await LoopbackRequest.Respond(stream, 403, "deny", cancellation: cancellation);
            return;
        }
        bool allowed = false;
        try
        {
            using var document = JsonDocument.Parse(request.Body);
            var root = document.RootElement;
            var tool = root.TryGetProperty("tool_name", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : "";
            var input = root.TryGetProperty("tool_input", out var i) ? i.Clone() : default;
            allowed = tool.Length > 0 && await decide(new ApprovalRequest(request.Header("x-agent-s-agent") ?? "", tool, input), cancellation);
        }
        catch (JsonException) { }
        await LoopbackRequest.Respond(stream, 200, allowed ? "allow" : "deny", cancellation: cancellation);
    }
}

/// <summary>
/// OpenRouter, bring your own key, as on macOS: the key stays in Windows Credential Locker and this loopback
/// proxy adds it to each request, so the agent (or a page that tricks it) never holds it. Only /v1 API paths
/// are forwarded, and responses stream back as they arrive.
/// </summary>
public sealed class OpenRouterProxy(Func<string?> key, HttpMessageHandler? handler = null) : LoopbackServer
{
    public static readonly Uri Api = new("https://openrouter.ai/api");
    private readonly HttpClient http = new(handler ?? new SocketsHttpHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromMinutes(10) };

    public string BaseUrl => $"http://127.0.0.1:{Port}";

    private static readonly HashSet<string> Dropped = new(StringComparer.OrdinalIgnoreCase)
    { "host", "authorization", "x-api-key", "content-length", "connection", "accept-encoding", "transfer-encoding", "keep-alive" };

    protected override async Task Serve(NetworkStream stream, CancellationToken cancellation)
    {
        var request = await LoopbackRequest.Read(stream, 32 << 20, cancellation);
        if (request is null) return;
        const string json = "application/json";
        if (request.Token != Token) { await LoopbackRequest.Respond(stream, 403, """{"error":{"message":"Forbidden"}}""", json, cancellation); return; }
        if (key() is not { Length: > 0 } secret)
        { await LoopbackRequest.Respond(stream, 401, """{"error":{"message":"Add your OpenRouter key in Agent S's Accounts."}}""", json, cancellation); return; }
        var path = request.Path;
        if (!path.StartsWith("/v1/", StringComparison.Ordinal) || path.Contains("..") || path.Contains("%2e", StringComparison.OrdinalIgnoreCase))
        { await LoopbackRequest.Respond(stream, 404, """{"error":{"message":"Not found"}}""", json, cancellation); return; }

        using var upstream = new HttpRequestMessage(new HttpMethod(request.Method), Api.AbsoluteUri + path);
        if (request.Body.Length > 0) upstream.Content = new ByteArrayContent(request.Body);
        foreach (var (name, value) in request.Headers.Where(h => !Dropped.Contains(h.Name)))
        {
            if (!upstream.Headers.TryAddWithoutValidation(name, value)) upstream.Content?.Headers.TryAddWithoutValidation(name, value);
        }
        upstream.Headers.TryAddWithoutValidation("Authorization", "Bearer " + secret);
        upstream.Headers.TryAddWithoutValidation("X-Title", "Agent S");
        HttpResponseMessage response;
        try { response = await http.SendAsync(upstream, HttpCompletionOption.ResponseHeadersRead, cancellation); }
        catch (HttpRequestException) { await LoopbackRequest.Respond(stream, 502, """{"error":{"message":"Could not reach OpenRouter."}}""", json, cancellation); return; }
        using (response)
        {
            var head = new StringBuilder($"HTTP/1.1 {(int)response.StatusCode} {response.ReasonPhrase}\r\n");
            foreach (var header in response.Headers.Concat(response.Content.Headers))
            {
                if (header.Key.ToLowerInvariant() is "content-length" or "transfer-encoding" or "connection" or "content-encoding") continue;
                head.Append(header.Key).Append(": ").Append(string.Join(", ", header.Value)).Append("\r\n");
            }
            head.Append("Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
            await stream.WriteAsync(Encoding.ASCII.GetBytes(head.ToString()), cancellation);
            await using var body = await response.Content.ReadAsStreamAsync(cancellation);
            var buffer = new byte[16 * 1024];
            int read;
            // Each read is forwarded at once, so streamed events reach the agent as they arrive.
            while ((read = await body.ReadAsync(buffer, cancellation)) > 0)
            {
                await stream.WriteAsync(Encoding.ASCII.GetBytes($"{read:x}\r\n"), cancellation);
                await stream.WriteAsync(buffer.AsMemory(0, read), cancellation);
                await stream.WriteAsync("\r\n"u8.ToArray(), cancellation);
                await stream.FlushAsync(cancellation);
            }
            await stream.WriteAsync("0\r\n\r\n"u8.ToArray(), cancellation);
        }
    }

    /// <summary>The public model list (no key needed).</summary>
    public static async Task<string[]> Models(CancellationToken cancellation = default)
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
            using var document = JsonDocument.Parse(await http.GetStringAsync(Api.AbsoluteUri + "/v1/models", cancellation));
            return document.RootElement.GetProperty("data").EnumerateArray()
                .Select(m => m.TryGetProperty("id", out var id) ? id.GetString() : null).OfType<string>().Order(StringComparer.Ordinal).ToArray();
        }
        catch (Exception error) when (error is HttpRequestException or JsonException or KeyNotFoundException or InvalidOperationException or TaskCanceledException)
        { return []; }
    }
}
