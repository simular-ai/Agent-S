using System.Net;
using System.Text;
using System.Text.Json;
using AgentS.Core;

/// <summary>Simular Box: Claude Code's arguments and the app's own MCP calls, against a fake Box server.</summary>
internal static class BoxTests
{
    public static async Task Run(Action<bool, string> check)
    {
        var saved = JsonSerializer.Deserialize<Conversation>(JsonSerializer.Serialize(new Conversation { Target = ConversationTarget.Box, BoxName = "work" }))!;
        check(saved.Target == ConversationTarget.Box && saved.BoxName == "work", "Box agents save their target and box name");

        var args = Box.ClaudeArguments("open example.com", null, "C:/p/box-mcp.json", null);
        check(args[args.IndexOf("--tools") + 1] == "" && args.Contains("--strict-mcp-config") && args[args.IndexOf("--mcp-config") + 1] == "C:/p/box-mcp.json",
            "Box turns give Claude Code no local tools, only Box's MCP server");
        var blocked = args[args.IndexOf("--disallowedTools") + 1];
        check(new[] { "box_delete", "box_upgrade", "billing", "box_create", "box_restore_paths" }.All(t => blocked.Contains("mcp__box__" + t)),
            "Tools that change the account or boxes are blocked");
        check(Box.Instructions("work").Contains("the box named \"work\"") && Box.Instructions(null).Contains("default Simular Box"), "Instructions name the chosen box");
        check(Box.McpConfig.Contains("${BOX_TOKEN}") && Box.McpConfig.Contains(Box.McpUrl), "The MCP config names the key's variable, never the key");
        check(Box.ToolLabel("mcp__box__browser_navigate") == "browser_navigate" && Box.IsAsleep("Hibernated") && !Box.IsAsleep("running") && !Box.IsAsleep(null),
            "Tool labels and sleep states");
        check(Box.SafeViewUrl("https://box.simular.ai/view/1") is not null && Box.SafeViewUrl("https://evil.example/view") is null &&
              Box.SafeViewUrl("http://box.simular.ai/x") is null, "Only Simular's own https live-view links open");

        var server = new FakeBox();
        using var client = new BoxClient("box-key", server);
        var (state, view) = await client.Status("work");
        check(server.Methods.Take(3).SequenceEqual(["initialize", "notifications/initialized", "tools/call"]) && server.Auth.All(a => a == "Bearer box-key"),
            "The app opens an MCP session with the key before calling a tool");
        check(state == "running" && view == new Uri("https://box.simular.ai/view/work"), "box_status gives the state and live-view link");
        check(server.LastArguments!.Value.GetProperty("box").GetString() == "work" && server.Sessions.Last() == "s-1", "Calls name the box and reuse the session");
        var png = await client.Screenshot(null);
        check(png is not null && png.SequenceEqual(new byte[] { 1, 2, 3 }) && !server.LastArguments!.Value.TryGetProperty("box", out _) &&
              server.LastArguments.Value.GetProperty("max_edge").GetInt32() == 1280, "Screenshots come back as image bytes; the default box sends no name");
        server.ExpireSession = true;
        var beforeRetry = server.Methods.Count;
        var (reconnected, _) = await client.Status("work");
        check(reconnected == "running" &&
              server.Methods.Skip(beforeRetry).SequenceEqual(["tools/call", "initialize", "notifications/initialized", "tools/call"]) &&
              server.Sessions[beforeRetry + 1] == "" && server.Sessions.Last() == "s-2",
            "An expired MCP session is renewed before retrying the request");
        var beforeWake = server.Methods.Count;
        server.ToolError = true;
        try { await client.Wake(null); check(false, "A Box tool error is reported"); }
        catch (BoxException error) { check(error.Message.Contains("no such box"), "A Box tool error is reported"); }
        check(server.Methods.Skip(beforeWake).SequenceEqual(["tools/call"]), "An ordinary wake error does not replay the tool call");
        server.ToolError = false;
        server.Status = HttpStatusCode.Unauthorized;
        try { await new BoxClient("revoked", server).Status(null); check(false, "A rejected key is reported"); }
        catch (BoxException error) { check(error.Message.Contains("rejected"), "A rejected key is reported"); }
    }

    /// <summary>Answers like Box's MCP server: JSON for initialize, server-sent events for tool calls.</summary>
    private sealed class FakeBox : HttpMessageHandler
    {
        public List<string> Methods = [], Auth = [], Sessions = [];
        public JsonElement? LastArguments;
        public HttpStatusCode Status = HttpStatusCode.OK;
        public bool ToolError, ExpireSession;
        private int sessionNumber;

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (Status != HttpStatusCode.OK) return new HttpResponseMessage(Status);
            using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
            var method = body.RootElement.GetProperty("method").GetString()!;
            Methods.Add(method); Auth.Add(request.Headers.Authorization?.ToString() ?? "");
            Sessions.Add(request.Headers.TryGetValues("Mcp-Session-Id", out var ids) ? ids.First() : "");
            if (ExpireSession && method == "tools/call")
            {
                ExpireSession = false;
                return new HttpResponseMessage(HttpStatusCode.NotFound);
            }
            var id = body.RootElement.TryGetProperty("id", out var i) ? i.GetInt32() : 0;
            string reply;
            if (method == "initialize") reply = $$$"""{"jsonrpc":"2.0","id":{{{id}}},"result":{"protocolVersion":"2025-06-18"}}""";
            else if (method == "tools/call")
            {
                var p = body.RootElement.GetProperty("params");
                LastArguments = p.GetProperty("arguments").Clone();
                var result = ToolError ? """{"isError":true,"content":[{"type":"text","text":"no such box"}]}"""
                    : p.GetProperty("name").GetString() switch
                    {
                        "box_status" => """{"content":[{"type":"text","text":"Status: {\"state\":\"running\",\"view_url\":\"https://box.simular.ai/view/work\"}"}]}""",
                        "screenshot" => """{"content":[{"type":"image","data":"AQID","mimeType":"image/png"}]}""",
                        _ => """{"content":[]}""",
                    };
                var sse = $"event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{result}}}\n\n";
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(sse, Encoding.UTF8, "text/event-stream") };
            }
            else return new HttpResponseMessage(HttpStatusCode.Accepted) { Content = new StringContent("") };
            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(reply, Encoding.UTF8, "application/json") };
            response.Headers.Add("Mcp-Session-Id", $"s-{++sessionNumber}");
            return response;
        }
    }
}
