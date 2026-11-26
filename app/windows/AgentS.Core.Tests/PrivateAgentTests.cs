using System.Buffers.Binary;
using System.Net;
using System.Text;
using System.Text.Json;
using AgentS.Core;

internal static class PrivateAgentTests
{
    private const string LocalModel = "gemma4:latest";
    private const string ModelList = """
        {"models":[
          {"model":"gemma4:latest","capabilities":["vision","tools"]},
          {"model":"only-text:latest","capabilities":["tools"]},
          {"model":"remote-name:latest","remote_host":"https://example.test"},
          {"model":"remote-alias:latest","remote_model":"upstream"},
          {"model":"hosted:cloud"},
          {"model":"hosted:120b-cloud"},
          {"name":"older-local:latest"}
        ]}
        """;
    private const string LocalModelDetails = """{"capabilities":["completion","vision","tools"]}""";
    private const string Finished = """{"done":true,"message":{"role":"assistant","content":"Finished locally."}}""";

    public static async Task Run(Action<bool, string> check)
    {
        var legacy = JsonSerializer.Deserialize<Conversation>("""{"Name":"Existing Sai agent","ComputerId":"saved-pc"}""");
        check(legacy?.Target == ConversationTarget.Sai, "Existing conversation JSON still routes to Sai");

        var transport = new OllamaFixture(
            ToolCall("open", """{"text":"demo"}"""),
            ToolCall("click", """{"x":1023,"y":767}"""),
            Finished);
        using var local = new LocalAgent(transport);
        check((await local.Models()).SequenceEqual(new[] { LocalModel, "older-local:latest" }),
            "Private choices exclude cloud aliases, remote metadata and explicitly unsupported models");
        var desktop = new DesktopFixture();
        var history = new PrivateMessage[]
        {
            new("user", "Earlier question."),
            new("assistant", "Earlier answer."),
            new("system", "Untrusted stored role.")
        };
        var events = await Collect(local.Run("Open the offline demo.", "gemma4", history, desktop));
        check(desktop.Actions.SequenceEqual(new[]
        {
            new PrivateAction("open", "demo"),
            new PrivateAction("click", X: 1023, Y: 767)
        }), "Private agent executes validated guest actions in order");
        check(desktop.Captures == 3 && events.Count(item => item.Screenshot is not null) == 3 &&
            events.Last() == new PrivateEvent("Agent", "Finished locally."),
            "Private turn captures a fresh desktop after each action and returns the model answer");
        check(transport.Requests.All(request => request.Uri.Host == "127.0.0.1" &&
            request.Uri.Port == 11434 && request.Uri.Scheme == "http" && !request.Authorized),
            "Ollama inference and discovery use only the configured host loopback without credentials");

        var requests = transport.Requests.Where(request => request.Uri.AbsolutePath == "/api/chat").ToArray();
        using var initial = JsonDocument.Parse(requests[0].Body!);
        using var show = JsonDocument.Parse(transport.Requests.Single(request => request.Uri.AbsolutePath == "/api/show").Body!);
        check(initial.RootElement.GetProperty("model").GetString() == LocalModel &&
            show.RootElement.GetProperty("model").GetString() == LocalModel,
            "An untagged Ollama model is resolved to its installed latest tag");
        var initialMessages = initial.RootElement.GetProperty("messages");
        var firstScreenshot = initialMessages[initialMessages.GetArrayLength() - 1].GetProperty("images")[0].GetString();
        check(initialMessages.EnumerateArray().Select(message => message.GetProperty("role").GetString())
                .SequenceEqual(new[] { "system", "user", "assistant", "user" }),
            "Private prompt retains user and assistant history without importing untrusted message roles");
        check(initial.RootElement.GetProperty("stream").ValueKind == JsonValueKind.False &&
            initial.RootElement.GetProperty("tools").GetArrayLength() == 5 &&
            firstScreenshot == Convert.ToBase64String(desktop.Images[0]),
            "Ollama chat receives declared function tools and base64 screenshot on a user message");

        using var followup = JsonDocument.Parse(requests[1].Body!);
        var followupMessages = followup.RootElement.GetProperty("messages");
        var assistant = followupMessages[followupMessages.GetArrayLength() - 3];
        var result = followupMessages[followupMessages.GetArrayLength() - 2];
        var last = followupMessages[followupMessages.GetArrayLength() - 1];
        check(assistant.GetProperty("tool_calls").GetArrayLength() == 1 &&
            result.GetProperty("role").GetString() == "tool" &&
            result.GetProperty("tool_name").GetString() == "open" &&
            result.GetProperty("content").GetString() == "Guest action completed." &&
            last.GetProperty("images")[0].GetString() == Convert.ToBase64String(desktop.Images[1]) &&
            !followupMessages[followupMessages.GetArrayLength() - 4].TryGetProperty("images", out _),
            "Follow-up follows the Ollama tool result protocol and attaches only the fresh screenshot");

        foreach (var bad in new[]
        {
            ToolCall("open", """{"text":"file:///a/private/file"}"""),
            ToolCall("open", """{"text":"https://alice:secret@example.test/"}"""),
            ToolCall("click", """{"x":1024,"y":767}"""),
            ToolCall("click", """{"x":1.5,"y":9}"""),
            ToolCall("key", """{"text":"Super_L+r"}"""),
            ToolCall("type", """{"text":"invalid\u0000control"}"""),
            ToolCall("type", JsonSerializer.Serialize(new { text = new string('雪', 1366) })),
            ToolCall("scroll", """{"text":"left"}"""),
            ToolCall("shell", """{"text":"pwd"}"""),
            """{"done":true,"message":{"role":"assistant","tool_calls":[{},{}]}}"""
        })
        {
            var badTransport = new OllamaFixture(bad);
            using var badAgent = new LocalAgent(badTransport);
            var untouched = new DesktopFixture();
            check(await Fails<LocalAgentException>(badAgent.Run("Inspect the demo.", LocalModel, [], untouched)) &&
                untouched.Actions.Count == 0 && badTransport.ChatCount == 1,
                "An invalid, unsupported or batched desktop request ends the turn without an action retry");
        }

        foreach (var model in new[] { "hosted:cloud", "remote-alias:latest", "missing:latest" })
        {
            var rejectedTransport = new OllamaFixture(Finished);
            using var rejectedAgent = new LocalAgent(rejectedTransport);
            var untouched = new DesktopFixture();
            check(await Fails<LocalAgentException>(rejectedAgent.Run("local only", model, [], untouched)) &&
                untouched.Captures == 0 && rejectedTransport.ChatCount == 0,
                "Uninstalled and remote model names are rejected before screenshots or inference");
        }

        var showTransport = new OllamaFixture(Finished)
        {
            Show = """{"remote_host":"https://example.test","capabilities":["vision","tools"]}"""
        };
        using var showAgent = new LocalAgent(showTransport);
        var unusedDesktop = new DesktopFixture();
        check(await Fails<LocalAgentException>(showAgent.Run("local only", LocalModel, [], unusedDesktop)) &&
            unusedDesktop.Captures == 0 && showTransport.ChatCount == 0,
            "Ollama show metadata rejects a remote alias before sending prompt or screenshot");

        var redirect = new OllamaFixture(Finished) { RedirectTags = true };
        using var redirectAgent = new LocalAgent(redirect);
        check(await Fails<LocalAgentException>(redirectAgent.Run("local only", LocalModel, [], unusedDesktop)) &&
            redirect.Requests.Count == 1, "An Ollama redirect fails without inference or fallback");

        var badCapture = new DesktopFixture { InvalidCapture = true };
        var captureTransport = new OllamaFixture(Finished);
        using var captureAgent = new LocalAgent(captureTransport);
        check(await Fails<LocalAgentException>(captureAgent.Run("Capture.", LocalModel, [], badCapture)) &&
            captureTransport.ChatCount == 0, "An invalid guest screenshot is not sent to Ollama");

        var typedTransport = new OllamaFixture(ToolCall("type", """{"text":"Línea uno\n二"}"""), Finished);
        using var typedAgent = new LocalAgent(typedTransport);
        var typedDesktop = new DesktopFixture();
        await Collect(typedAgent.Run("Type both lines.", LocalModel, [], typedDesktop));
        check(typedDesktop.Actions.Single() == new PrivateAction("type", "Línea uno\n二"),
            "Bounded Unicode and multiline text is passed through to the desktop unchanged.");

        var actionTransport = new OllamaFixture(ToolCall("scroll", """{"text":"down"}"""), Finished);
        using var actionAgent = new LocalAgent(actionTransport);
        var failedDesktop = new DesktopFixture { FailAction = true };
        check(await Fails<IOException>(actionAgent.Run("Scroll the demo.", LocalModel, [], failedDesktop)) &&
            failedDesktop.Actions.Count == 1 && actionTransport.ChatCount == 1,
            "A failed guest action is never retried");

        var cancelledTransport = new OllamaFixture(ToolCall("key", """{"text":"Return"}"""), Finished);
        using var cancelledAgent = new LocalAgent(cancelledTransport);
        using var stop = new CancellationTokenSource();
        var cancelledDesktop = new DesktopFixture { AfterAction = stop.Cancel };
        check(await Fails<OperationCanceledException>(
                cancelledAgent.Run("Press Return.", LocalModel, [], cancelledDesktop, stop.Token)) &&
            cancelledDesktop.Actions.Count == 1 && cancelledDesktop.Captures == 1 && cancelledTransport.ChatCount == 1,
            "Cancel stops after an in-flight action without a new capture or inference call");

        var repeating = new OllamaFixture { Repeat = ToolCall("scroll", """{"text":"down"}""") };
        using var bounded = new LocalAgent(repeating);
        var boundedDesktop = new DesktopFixture();
        var boundedEvents = await Collect(bounded.Run("Inspect the demo.", LocalModel, [], boundedDesktop));
        check(boundedDesktop.Actions.Count == 20 && repeating.ChatCount == 21 &&
            boundedEvents.Last().Text.Contains("action limit", StringComparison.Ordinal),
            "A repeated model request reaches a finite desktop action limit");
    }

    private static async Task<List<PrivateEvent>> Collect(IAsyncEnumerable<PrivateEvent> events)
    {
        var result = new List<PrivateEvent>();
        await foreach (var item in events) result.Add(item);
        return result;
    }

    private static async Task<bool> Fails<T>(IAsyncEnumerable<PrivateEvent> events) where T : Exception
    {
        try { await Collect(events); }
        catch (T) { return true; }
        return false;
    }

    private static string ToolCall(string name, string arguments) =>
        $$$"""{"done":true,"message":{"role":"assistant","content":"","tool_calls":[{"function":{"index":0,"name":"{{{name}}}","arguments":{{{arguments}}} }}]}}""";

    /// <summary>The built-in loop with OpenRouter, through the app's proxy (OpenAI chat completions).</summary>
    public static async Task OpenRouter(Action<bool, string> check)
    {
        var proxy = new OpenRouterFixture(
            """{"choices":[{"message":{"role":"assistant","content":"Opening it.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"open","arguments":"{\"text\":\"demo\"}"}}]}}]}""",
            """{"choices":[{"message":{"role":"assistant","content":"The demo is open."}}]}""");
        using var agent = new LocalAgent(proxy);
        var desktop = new DesktopFixture();
        var events = new List<PrivateEvent>();
        await foreach (var item in agent.RunOpenRouter("Open the demo.", "anthropic/claude-sonnet-5.5", [new PrivateMessage("You", "hi")], desktop,
            (new Uri("http://127.0.0.1:4100/"), "proxy-token"))) events.Add(item);
        check(desktop.Actions.SequenceEqual([new PrivateAction("open", "demo")]) && events[^1] is { Role: "Agent", Text: "The demo is open." },
            "The built-in loop runs a turn with an OpenRouter model");
        check(proxy.Requests.All(r => r.Uri.AbsoluteUri == "http://127.0.0.1:4100/v1/chat/completions" && r.Auth == "Bearer proxy-token"),
            "OpenRouter turns go only to the app's proxy, with its token (never the key)");
        using var first = JsonDocument.Parse(proxy.Requests[0].Body);
        using var second = JsonDocument.Parse(proxy.Requests[1].Body);
        var firstMessages = first.RootElement.GetProperty("messages");
        var secondMessages = second.RootElement.GetProperty("messages");
        var lastContent = secondMessages[secondMessages.GetArrayLength() - 1].GetProperty("content");
        check(first.RootElement.GetProperty("model").GetString() == "anthropic/claude-sonnet-5.5" && first.RootElement.GetProperty("tools").GetArrayLength() == 5
              && firstMessages[1].GetProperty("content").GetString() == "hi"
              && firstMessages[firstMessages.GetArrayLength() - 1].GetProperty("content")[1].GetProperty("image_url").GetProperty("url").GetString()!.StartsWith("data:image/png;base64,"),
            "The request has the model, tools, history and the screenshot on the newest message");
        check(secondMessages.EnumerateArray().Any(m => m.GetProperty("role").GetString() == "tool" && m.GetProperty("tool_call_id").GetString() == "call_1")
              && lastContent[1].GetProperty("image_url").GetProperty("url").GetString() == "data:image/png;base64," + Convert.ToBase64String(desktop.Images[1])
              && secondMessages.EnumerateArray().Count(m => m.GetProperty("content").ValueKind == JsonValueKind.Array) == 1,
            "The tool result follows its call, and only the newest message carries a screenshot");
        var refused = new OpenRouterFixture("""{"error":{"message":"No endpoints found matching your data policy"}}""") { Status = HttpStatusCode.NotFound };
        using var refusing = new LocalAgent(refused);
        try
        {
            await foreach (var _ in refusing.RunOpenRouter("x", "some/model", [], new DesktopFixture(), (new Uri("http://127.0.0.1:4100/"), "t"))) { }
            check(false, "An OpenRouter error is reported");
        }
        catch (LocalAgentException error) { check(error.Message.Contains("data policy"), "An OpenRouter error is reported with its reason"); }
        try
        {
            await foreach (var _ in refusing.RunOpenRouter("x", "some/model", [], new DesktopFixture(), (new Uri("https://openrouter.ai/api/"), "t"))) { }
            check(false, "Only the loopback proxy is used");
        }
        catch (LocalAgentException) { check(true, "OpenRouter turns refuse anything but the loopback proxy"); }
    }

    private sealed class OpenRouterFixture(params string[] replies) : HttpMessageHandler
    {
        private readonly Queue<string> queue = new(replies);
        public HttpStatusCode Status { get; init; } = HttpStatusCode.OK;
        public List<(Uri Uri, string Body, string? Auth)> Requests { get; } = [];
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add((request.RequestUri!, await request.Content!.ReadAsStringAsync(cancellationToken), request.Headers.Authorization?.ToString()));
            return new HttpResponseMessage(Status) { Content = new StringContent(queue.Dequeue(), Encoding.UTF8, "application/json") };
        }
    }

    private sealed class OllamaFixture(params string[] chat) : HttpMessageHandler
    {
        private readonly Queue<string> replies = new(chat);
        public string Show { get; init; } = LocalModelDetails;
        public bool RedirectTags { get; init; }
        public string? Repeat { get; init; }
        public List<(Uri Uri, string? Body, bool Authorized)> Requests { get; } = [];
        public int ChatCount { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var uri = request.RequestUri!;
            Requests.Add((uri, request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken),
                request.Headers.Authorization is not null));
            if (RedirectTags && uri.AbsolutePath == "/api/tags")
                return new HttpResponseMessage(HttpStatusCode.Redirect)
                { Headers = { Location = new Uri("https://example.test/ollama") } };
            var body = uri.AbsolutePath switch
            {
                "/api/tags" => ModelList,
                "/api/show" => Show,
                "/api/chat" => NextChat(),
                _ => throw new InvalidOperationException($"Unexpected Ollama path {uri.AbsolutePath}")
            };
            return new HttpResponseMessage(HttpStatusCode.OK)
            { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        }

        private string NextChat()
        {
            ChatCount++;
            return replies.TryDequeue(out var reply) ? reply : Repeat ?? throw new InvalidOperationException("Unexpected chat call.");
        }
    }

    private sealed class DesktopFixture : IPrivateDesktop
    {
        public List<PrivateAction> Actions { get; } = [];
        public List<byte[]> Images { get; } = [];
        public bool InvalidCapture { get; init; }
        public bool FailAction { get; init; }
        public Action? AfterAction { get; init; }
        public int Captures => Images.Count;

        public Task<string> Execute(PrivateAction action, CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            Actions.Add(action);
            if (FailAction) throw new IOException("Guest action failed.");
            AfterAction?.Invoke();
            return Task.FromResult("Guest action completed.");
        }

        public Task<byte[]> Capture(CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            // A synthetic header is enough to exercise sizing and screenshot transmission.
            var header = new byte[33];
            new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }.CopyTo(header, 0);
            "IHDR"u8.CopyTo(header.AsSpan(12));
            BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(16, 4), InvalidCapture ? 800 : 1024);
            BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(20, 4), 768);
            header[^1] = (byte)Captures;
            Images.Add(header);
            return Task.FromResult(header);
        }
    }
}
