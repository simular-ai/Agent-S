using System.Net;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace AgentS.Core;

/// Uses the same public /v1/agents contract as Models/SaiAPI.swift. Never retries writes.
public sealed class SaiClient : IDisposable
{
    public static readonly Uri Endpoint = new("https://api.simular.ai/");
    private readonly HttpClient http;
    private readonly string key;
    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

    public SaiClient(string key) : this(key, new HttpClientHandler { AllowAutoRedirect = false }) { }
    public SaiClient(string key, HttpMessageHandler handler)
    {
        if (string.IsNullOrWhiteSpace(key)) throw new ArgumentException("Add your Sai API key.");
        this.key = key.Trim();
        http = new HttpClient(handler) { BaseAddress = Endpoint, Timeout = Timeout.InfiniteTimeSpan };
    }

    private HttpRequestMessage Request(string path, object? body = null)
    {
        var request = new HttpRequestMessage(body is null ? HttpMethod.Get : HttpMethod.Post, path);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        request.Headers.UserAgent.ParseAdd("AgentS-Windows/0.1");
        if (body is not null)
            request.Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json");
        return request;
    }

    private static void Check(HttpResponseMessage response, CancellationToken cancellation)
    {
        if (response.IsSuccessStatusCode) return;
        // Do not put response bodies, live URLs, credentials or proxy pages in UI errors/logs.
        if (response.StatusCode == HttpStatusCode.Unauthorized)
            throw new SaiException("Sai rejected the API key. Check it in Accounts.");
        cancellation.ThrowIfCancellationRequested();
        throw new SaiException($"Sai returned HTTP {(int)response.StatusCode}. The request was not confirmed; do not assume it succeeded.");
    }

    private async Task<JsonElement> Call(string path, object? body, CancellationToken cancellation)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        deadline.CancelAfter(TimeSpan.FromSeconds(30));
        using var request = Request(path, body);
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token);
        Check(response, deadline.Token);
        using var stream = await response.Content.ReadAsStreamAsync(deadline.Token);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: deadline.Token);
        return document.RootElement.Clone();
    }

    public async Task<Computer[]> Computers(CancellationToken cancellation = default)
    {
        var data = await Call("v1/agents/machines", null, cancellation);
        return data.GetProperty("machines").Deserialize<Computer[]>(Json) ?? [];
    }

    public async Task<SaiModel[]> Models(CancellationToken cancellation = default)
    {
        var data = await Call("v1/agents/models", null, cancellation);
        return data.GetProperty("models").Deserialize<SaiModel[]>(Json) ?? [];
    }

    public Task Approve(string id, string answer, CancellationToken cancellation = default) =>
        Call("v1/agents/approve", new { approvalId = id, response = answer }, cancellation);
    public Task Abort(string session, CancellationToken cancellation = default) =>
        Call("v1/agents/abort", new { sessionId = session }, cancellation);
    public async Task Wake(string computer, CancellationToken cancellation = default)
    {
        var data = await Call("v1/agents/wake", new { machineId = computer }, cancellation);
        var status = data.TryGetProperty("status", out var s) ? s.GetString() : null;
        if ((data.TryGetProperty("startingUp", out var starting) && starting.ValueKind == JsonValueKind.True) ||
            (data.TryGetProperty("waking", out var waking) && waking.ValueKind == JsonValueKind.True) ||
            status is "active" or "wakingup") return;
        throw new SaiException("Sai did not confirm that this computer is starting.");
    }

    public async Task<Uri> LiveLink(string computer, CancellationToken cancellation = default)
    {
        var data = await Call($"v1/agents/machines/{Uri.EscapeDataString(computer)}/live", new { }, cancellation);
        return SafeWebLink(data.GetProperty("liveUrl").GetString()) ?? throw new SaiException("Sai did not return a valid screen link.");
    }

    public static Uri? SafeWebLink(string? value) =>
        Uri.TryCreate(value, UriKind.Absolute, out var uri) && uri.Scheme == "https" &&
        string.IsNullOrEmpty(uri.UserInfo) ? uri : null;

    public async IAsyncEnumerable<JsonElement> Run(string message, string computer, string? session,
        string? model, [EnumeratorCancellation] CancellationToken cancellation = default)
    {
        var body = new Dictionary<string, object> { ["message"] = message, ["machineId"] = computer };
        if (!string.IsNullOrEmpty(session)) body["sessionId"] = session;
        if (!string.IsNullOrEmpty(model) && model != "auto") body["model"] = model;
        using var request = Request("v1/agents/message", body);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
        using var headersDeadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        headersDeadline.CancelAfter(TimeSpan.FromSeconds(30));
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, headersDeadline.Token);
        Check(response, cancellation);
        var contentType = response.Content.Headers.ContentType?.MediaType;
        if (contentType != "text/event-stream") throw new SaiException("Sai did not return an event stream.");
        using var turnDeadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        turnDeadline.CancelAfter(TimeSpan.FromHours(1));
        using var stream = await response.Content.ReadAsStreamAsync(turnDeadline.Token);
        using var reader = new StreamReader(stream);
        var finished = false;
        await foreach (var entry in EventStream.Read(reader, turnDeadline.Token))
        {
            if (entry.TryGetProperty("type", out var type) && type.GetString() == "finish") finished = true;
            yield return entry;
        }
        if (!finished) throw new SaiException("The stream disconnected before confirming completion. Check the task in Sai before retrying.");
    }

    public void Dispose() => http.Dispose();
}

public sealed class SaiException(string message) : Exception(message);

public static class EventStream
{
    public static async IAsyncEnumerable<JsonElement> Read(TextReader reader,
        [EnumeratorCancellation] CancellationToken cancellation = default)
    {
        var payload = new StringBuilder();
        var buffered = 0;
        while (true)
        {
            var line = await reader.ReadLineAsync(cancellation);
            if (line is null || line.Length == 0)
            {
                if (payload.Length > 0)
                {
                    var text = payload.ToString().TrimEnd('\n');
                    if (text == "[DONE]") yield break;
                    using var document = JsonDocument.Parse(text);
                    yield return document.RootElement.Clone();
                    payload.Clear();
                }
                buffered = 0;
                if (line is null) yield break;
            }
            else if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                var value = line[5..];
                if (value.StartsWith(' ')) value = value[1..];
                buffered += value.Length;
                if (buffered > 1_048_576) throw new SaiException("Sai returned an oversized stream event.");
                payload.Append(value).Append('\n');
            }
        }
    }
}
