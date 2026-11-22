using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using AgentS.Core;

/// <summary>This PC agents (approvals, Claude Code arguments) and the OpenRouter proxy, against loopback fixtures.</summary>
internal static class ThisPcTests
{
    private const string Pc = "C:/Users/me/AppData/Local/Simular/agent-s-tools/pc";

    public static async Task Run(Action<bool, string> check)
    {
        // ---- Saved agents ----
        var older = JsonSerializer.Deserialize<Conversation>("""{"Name":"Private agent","Target":"Private"}""")!;
        check(older.Provider == ModelProvider.Ollama, "Existing agents default to Ollama");
        var saved = JsonSerializer.Serialize(new Conversation { Target = ConversationTarget.ThisPc, Provider = ModelProvider.OpenRouter, ModelId = "anthropic/claude-sonnet-5.5" });
        var restored = JsonSerializer.Deserialize<Conversation>(saved)!;
        check(saved.Contains("\"ThisPc\"") && saved.Contains("\"OpenRouter\"") && restored.Target == ConversationTarget.ThisPc && restored.Provider == ModelProvider.OpenRouter,
            "This PC target and provider are saved by name");
        check(!JsonSerializer.Serialize(new ChatEntry { Text = "x", LocalAnswer = _ => { } }).Contains("LocalAnswer"), "Pending approvals are never saved to history");

        // ---- What runs without asking ----
        JsonElement Input(string json) { using var d = JsonDocument.Parse(json); return d.RootElement.Clone(); }
        JsonElement Bash(string command) => Input(JsonSerializer.Serialize(new { command }));
        foreach (var look in new[] { $"{Pc} apps", $"{Pc} tree", $"{Pc} tree --all", $"{Pc} screenshot" })
            check(ThisPcAgent.IsLookOnly("Bash", Bash(look), Pc), "Looking runs without asking: " + look[(Pc.Length + 1)..]);
        foreach (var act in new[] { $"{Pc} click 3", $"{Pc} type 2 hi", $"{Pc} open notepad", $"{Pc} key ctrl+l", $"{Pc} tree; rm -rf ~",
                                    $"{Pc} tree && curl x", $"{Pc} tree $(whoami)", $"{Pc} tree > out.txt", "ls", $"/other/pc tree", $"{Pc} apps extra" })
            check(!ThisPcAgent.IsLookOnly("Bash", Bash(act), Pc), "Asks first: " + act.Replace(Pc, "pc"));
        check(!ThisPcAgent.IsLookOnly("Write", Input("""{"file_path":"C:/x.txt","content":"x"}"""), Pc), "Writing a file asks first");
        check(ThisPcAgent.IsLookOnly("Read", Input("""{"file_path":"C:\\Users\\me\\AppData\\Local\\Simular\\agent-s-tools\\agent-s-screen.png"}"""), Pc),
            "Reading the screenshot pc just took runs without asking");
        check(!ThisPcAgent.IsLookOnly("Read", Input("""{"file_path":"C:/Users/me/Documents/taxes.pdf"}"""), Pc), "Reading any other file asks first");
        const string spaced = "C:/Users/My Name/AppData/Local/Simular/agent-s-tools/pc";
        check(ThisPcAgent.IsLookOnly("Bash", Bash($"\"{spaced}\" tree"), spaced) && ThisPcAgent.Instructions(spaced).Contains($"\\\"{spaced}\\\" tree"),
            "A tool path with spaces is quoted in the instructions and still recognized");
        check(ThisPcAgent.Describe("Bash", Bash("del C:\\x")) == "Run: del C:\\x" &&
              ThisPcAgent.Describe("Edit", Input("""{"file_path":"a.txt","old_string":"x","new_string":"y"}""")).Contains("replace “x” with “y”"),
            "Approvals show the exact command or change");

        // ---- Claude Code arguments ----
        var args = ThisPcAgent.ClaudeArguments("hello", null, Pc, "C:/settings.json");
        check(args[0] == "-p" && args[1] == "hello" && args.Contains("bypassPermissions") && args.Contains("--settings") && args.Contains("C:/settings.json") &&
              args[args.IndexOf("--setting-sources") + 1] == "" && args[args.IndexOf("--tools") + 1] == "Bash,Read,Write,Edit" && !args.Contains("--resume"),
            "Claude Code runs headless with Agent S's settings only and the approval hook");
        check(ThisPcAgent.ClaudeArguments("x", "s-1", Pc, "s.json") is var resumed && resumed[resumed.IndexOf("--resume") + 1] == "s-1", "This PC sessions resume");
        using (var settings = JsonDocument.Parse(ThisPcAgent.Settings("powershell.exe -File \"C:/t/approve.ps1\"")))
        {
            var hook = settings.RootElement.GetProperty("hooks").GetProperty("PreToolUse")[0];
            check(hook.GetProperty("matcher").GetString() == "*" && hook.GetProperty("hooks")[0].GetProperty("command").GetString()!.Contains("approve.ps1"),
                "Every tool use goes through the approval hook");
        }
        var env = ThisPcAgent.ModelEnvironment(ModelProvider.OpenRouter, "http://127.0.0.1:5", "agent-s-token", "anthropic/claude-sonnet-5.5");
        check(env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5" && env["ANTHROPIC_AUTH_TOKEN"] == "agent-s-token" && env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "200000",
            "OpenRouter turns use the proxy and its token, never the key");
        check(ThisPcAgent.ModelEnvironment(ModelProvider.Ollama, HarnessCommand.OllamaUrl, "ollama", "m")["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "32768",
            "Ollama turns keep a local model's window");

        // ---- Approval server ----
        ApprovalRequest? asked = null;
        using var approvals = new ApprovalServer((request, _) => { asked = request; return Task.FromResult(request.Tool == "Bash"); });
        approvals.Start();
        async Task<(HttpStatusCode, string)> Post(int port, string path, string? token, string body, string? agent = null)
        {
            using var http = new HttpClient(new SocketsHttpHandler { UseProxy = false });
            using var request = new HttpRequestMessage(HttpMethod.Post, $"http://127.0.0.1:{port}{path}") { Content = new StringContent(body, Encoding.UTF8, "application/json") };
            if (token is not null) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
            if (agent is not null) request.Headers.Add("X-Agent-S-Agent", agent);
            using var response = await http.SendAsync(request);
            return (response.StatusCode, await response.Content.ReadAsStringAsync());
        }
        var hookBody = """{"session_id":"s","tool_name":"Bash","tool_input":{"command":"pc click 3"}}""";
        check(await Post(approvals.Port, "/approve", approvals.Token, hookBody, "agent-1") is (HttpStatusCode.OK, "allow") &&
              asked is { Agent: "agent-1", Tool: "Bash" } && asked.Input.GetProperty("command").GetString() == "pc click 3",
            "The hook's tool use reaches the person and their answer comes back");
        check((await Post(approvals.Port, "/approve", approvals.Token, """{"tool_name":"Write","tool_input":{}}""")).Item2 == "deny", "A denial is returned to the hook");
        asked = null;
        check(await Post(approvals.Port, "/approve", "wrong-token", hookBody) is (HttpStatusCode.Forbidden, "deny") && asked is null,
            "Without the turn's token nothing is asked and the answer is deny");
        check((await Post(approvals.Port, "/approve", approvals.Token, "not json")).Item2 == "deny", "A malformed request is denied");

        // ---- OpenRouter proxy ----
        var upstream = new FakeOpenRouter();
        string? key = "sk-or-secret";
        using var proxy = new OpenRouterProxy(() => key, upstream);
        proxy.Start();
        using var client = new HttpClient(new SocketsHttpHandler { UseProxy = false });
        async Task<HttpResponseMessage> Call(string path, string? token, bool apiKeyHeader = false)
        {
            var request = new HttpRequestMessage(HttpMethod.Post, proxy.BaseUrl + path) { Content = new StringContent("""{"model":"m","stream":true}""", Encoding.UTF8, "application/json") };
            if (token is not null)
            {
                if (apiKeyHeader) request.Headers.Add("x-api-key", token);
                else request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
            }
            return await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        }
        using (var response = await Call("/v1/messages", proxy.Token, apiKeyHeader: true))
        {
            var text = await response.Content.ReadAsStringAsync();
            check(response.StatusCode == HttpStatusCode.OK && text == "event: one\n\nevent: two\n\n" && upstream.Url == "https://openrouter.ai/api/v1/messages",
                "The proxy forwards /v1 requests to OpenRouter and streams the answer back");
            check(upstream.Auth == "Bearer sk-or-secret" && upstream.ApiKey is null && upstream.Body == """{"model":"m","stream":true}""",
                "The proxy swaps the agent's token for the saved key");
        }
        upstream.Url = null;
        using (var response = await Call("/v1/messages", "guess")) check(response.StatusCode == HttpStatusCode.Forbidden && upstream.Url is null, "Without the proxy token nothing reaches OpenRouter");
        // Raw requests: HttpClient would tidy these paths before sending them.
        async Task<string> Raw(string path)
        {
            using var tcp = new TcpClient();
            await tcp.ConnectAsync(IPAddress.Loopback, proxy.Port);
            var stream = tcp.GetStream();
            await stream.WriteAsync(Encoding.ASCII.GetBytes($"GET {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {proxy.Token}\r\nContent-Length: 0\r\n\r\n"));
            return (await new StreamReader(stream).ReadLineAsync()) ?? "";
        }
        check((await Raw("/v1/../keys")).Contains(" 404 ") && upstream.Url is null, "Only OpenRouter's /v1 API is reachable");
        check((await Raw("/v1/%2e%2e/keys")).Contains(" 404 ") && (await Raw("/api/keys")).Contains(" 404 ") && upstream.Url is null, "Encoded and other paths are refused");
        key = null;
        using (var response = await Call("/v1/messages", proxy.Token))
            check(response.StatusCode == HttpStatusCode.Unauthorized && (await response.Content.ReadAsStringAsync()).Contains("Accounts"), "A missing key is reported, not sent");
    }

    private sealed class FakeOpenRouter : HttpMessageHandler
    {
        public string? Url, Auth, ApiKey, Body;
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Url = request.RequestUri!.AbsoluteUri;
            Auth = request.Headers.Authorization?.ToString();
            ApiKey = request.Headers.TryGetValues("x-api-key", out var values) ? values.First() : null;
            Body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("event: one\n\nevent: two\n\n", Encoding.UTF8, "text/event-stream") };
        }
    }
}
