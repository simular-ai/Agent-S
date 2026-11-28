using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>Runs DeepSeek's restricted SDK profile with the Windows tool exposed through a per-turn MCP bridge.</summary>
internal static class ThisPcDeepSeek
{
    public static string RuntimeFolder => Path.Combine(AppIdentity.LocalFolder, "DeepSeekRuntime");

    public static (string Node, string Entry)? Find()
    {
        var root = AppIdentity.IsRelease ? RuntimeFolder
            : Environment.GetEnvironmentVariable("AGENT_S_DSH_RUNTIME") is { Length: > 0 } dev ? dev : RuntimeFolder;
        var portable = Path.Combine(root, "bin", "node.exe");
        var node = File.Exists(portable) ? portable : ThisPcRunner.FindExecutable("node.exe", Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "nodejs"));
        var package = Path.Combine(root, "node_modules", "@deepseek-ai", "dsh");
        if (node is null || !File.Exists(Path.Combine(package, "package.json"))) return null;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(package, "package.json")));
            if (doc.RootElement.String("version") != HarnessCommand.DeepSeekVersion ||
                doc.RootElement.Get("bin").String("dsh") is not { } bin) return null;
            var entry = Path.GetFullPath(Path.Combine(package, bin));
            return entry.StartsWith(Path.GetFullPath(package) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) &&
                File.Exists(entry) ? (node, entry) : null;
        }
        catch (Exception error) when (error is IOException or ArgumentException or JsonException) { return null; }
    }

    public static async IAsyncEnumerable<HarnessEvent> Run((string Node, string Entry) runtime, Conversation owner,
        string prompt, string? resume, string model, string modelUrl, string key, ThisPcCommand command,
        [EnumeratorCancellation] CancellationToken token)
    {
        if (!Guid.TryParse(owner.Id, out var agentId)) throw new PrivateComputerException("The agent ID is invalid.");
        var root = Path.Combine(AppIdentity.LocalFolder, "deepseek-this-pc", agentId.ToString("N"), owner.Provider.ToString());
        var work = Directory.CreateDirectory(Path.Combine(root, "work")).FullName;
        var home = Directory.CreateDirectory(Path.Combine(root, "profile")).FullName;
        var sessions = Directory.CreateDirectory(Path.Combine(root, "sessions")).FullName;
        var patch = Path.Combine(home, "tools.patch.yml");
        File.Copy(Path.Combine(AppContext.BaseDirectory, "DeepSeekRuntime", "this-pc.patch.yml"), patch, overwrite: true);

        using var tool = new ThisPcMcpBridge(command, token);
        tool.Start();
        var start = ThisPcRpc.StartInfo(runtime.Node, work, home);
        foreach (var argument in new[] { runtime.Entry, "--profile", "sdk-minimal", "--patch", patch })
            start.ArgumentList.Add(argument);
        start.Environment["DSH_HOME"] = home;
        start.Environment["DSH_AGENTS_HOME"] = home;
        start.Environment["DSH_TELEMETRY_MODE"] = "DISABLED";
        start.Environment["AGENT_S_SDK"] = "1";
        start.Environment["AGENT_S_DSH_SESSIONS"] = sessions;
        start.Environment["AGENT_S_MODEL"] = model;
        start.Environment["AGENT_S_MODEL_URL"] = modelUrl + "/v1";
        start.Environment["AGENT_S_MODEL_TOKEN"] = key;
        start.Environment["AGENT_S_MODEL_CONTEXT"] = owner.Provider.ContextTokens().ToString(System.Globalization.CultureInfo.InvariantCulture);
        start.Environment["AGENT_S_MODEL_OUTPUT"] = owner.Provider.MaxOutputTokens().ToString(System.Globalization.CultureInfo.InvariantCulture);
        start.Environment["AGENT_S_PC_URL"] = $"http://127.0.0.1:{tool.Port}/mcp";
        start.Environment["AGENT_S_PC_AUTHORIZATION"] = "Bearer " + tool.Token;
        start.Environment["DSH_SYSTEM_PROMPT"] = ThisPcCommand.Instructions;

        await using var rpc = new ThisPcRpc(start, "DeepSeek");
        await rpc.Request("initialize", new { cwd = work, provider = "agent-s", model, maxTokens = owner.Provider.MaxOutputTokens() }, token);
        var sessionId = "session-" + Guid.NewGuid().ToString("D");
        if (resume?.StartsWith("session-", StringComparison.Ordinal) == true && Guid.TryParse(resume[8..], out _))
        {
            await rpc.Request("session/resume", new { sessionId = resume }, token);
            sessionId = resume;
        }
        tool.CanRun = true;
        var sent = await rpc.Request("session/prompt", new { sessionId, contentBlocks = new[] { new { type = "text", text = prompt } } }, token);
        var messageId = sent.String("messageId") ?? throw new PrivateComputerException("DeepSeek did not accept the message.");
        yield return new HarnessStarted(sessionId);

        bool committed = false;
        string? completion = null;
        while (true)
        {
            var message = await rpc.Next(token);
            var p = message.Get("params");
            if (p.String("sessionId") != sessionId) continue;
            if (message.String("method") == "session.status" && p.String("status") == "idle" && committed && completion is not null)
            {
                tool.CanRun = false;
                bool success = completion == "completed";
                yield return new HarnessFinished(success ? null : "The DeepSeek turn did not complete. Check the selected model and provider.", !success);
                yield break;
            }
            if (message.String("method") != "session.event") continue;
            var item = p.Get("event");
            var data = item.Get("data");
            switch (item.String("type"))
            {
                case "user/message" when data.String("id") == messageId:
                    committed = true;
                    break;
                case "turn/end" when committed:
                    completion = data.Get("reason").String("kind") ?? "unknown";
                    break;
                case "assistant/message" when committed && data.Get("message").Get("content") is { ValueKind: JsonValueKind.Array } blocks:
                    foreach (var block in blocks.EnumerateArray())
                        if (block.String("type") == "text" && block.String("text") is { } text) yield return new HarnessText(text);
                    var usage = data.Get("usage");
                    if (usage.Get("inputTokens").TryInt(out int input) && usage.Get("outputTokens").TryInt(out int output))
                        yield return new HarnessUsage(input, output);
                    break;
                case "tool/call" when committed && data.String("callId") is { } call && data.String("arguments") is { } args:
                    if (Command(args) is { } shell)
                        yield return new HarnessToolCall(call, "Bash", shell, shell);
                    break;
                case "tool/result" when committed && item.String("surfaceOp") == "append":
                    var result = data.Get("message");
                    if (result.String("toolCallId") is { } id && result.Get("content") is { ValueKind: JsonValueKind.Array } parts)
                    {
                        var outputText = string.Join("\n", parts.EnumerateArray().Select(part => part.String("text")).OfType<string>());
                        yield return new HarnessToolResult(id, outputText[..Math.Min(outputText.Length, HarnessStream.MaxToolOutput)],
                            result.Get("isError").ValueKind == JsonValueKind.True);
                    }
                    break;
            }
        }
    }

    private static string? Command(string args)
    {
        try { using var json = JsonDocument.Parse(args); return json.RootElement.String("command"); }
        catch (JsonException) { return null; }
    }
}

internal sealed class ThisPcMcpBridge(ThisPcCommand command, CancellationToken turn) : LoopbackServer
{
    private int inUse;
    private volatile bool canRun;
    public bool CanRun { get => canRun; set => canRun = value; }

    protected override async Task Serve(NetworkStream stream, CancellationToken cancellation)
    {
        const string json = "application/json";
        var request = await LoopbackRequest.Read(stream, 2 * 1024 * 1024, cancellation);
        if (request is null) return;
        if (request.Path != "/mcp" || request.Header("authorization") != "Bearer " + Token)
        {
            await LoopbackRequest.Respond(stream, 403, "", json, cancellation);
            return;
        }
        if (request.Method != "POST")
        {
            await LoopbackRequest.Respond(stream, 405, "", json, cancellation);
            return;
        }
        using var document = JsonDocument.Parse(request.Body);
        var message = document.RootElement;
        if (message.String("jsonrpc") != "2.0" || message.String("method") is not { } method)
        {
            await LoopbackRequest.Respond(stream, 400, "", json, cancellation);
            return;
        }
        var id = message.Get("id");
        if (id.ValueKind == JsonValueKind.Undefined)
        {
            await LoopbackRequest.Respond(stream, 202, "", json, cancellation);
            return;
        }
        var parameters = message.Get("params");
        object? result = method switch
        {
            "initialize" => new
            {
                protocolVersion = parameters.String("protocolVersion") is "2025-03-26" or "2025-06-18" or "2025-11-25"
                    ? parameters.String("protocolVersion") : "2025-11-25",
                capabilities = new { tools = new { } }, serverInfo = new { name = "Agent S", version = "1" },
            },
            "tools/list" => new { tools = new[] { new { name = ThisPcCommand.Name, description = ThisPcCommand.Description, inputSchema = ThisPcCommand.Schema } } },
            "tools/call" => await Call(parameters, cancellation),
            "ping" => new { },
            _ => null,
        };
        string packet = result is null
            ? JsonSerializer.Serialize(new { jsonrpc = "2.0", id, error = new { code = -32601, message = "Unknown method." } })
            : JsonSerializer.Serialize(new { jsonrpc = "2.0", id, result });
        await LoopbackRequest.Respond(stream, 200, packet, json, cancellation);
    }

    private async Task<object> Call(JsonElement parameters, CancellationToken server)
    {
        var args = parameters.Get("arguments");
        if (parameters.String("name") != ThisPcCommand.Name || args.String("command") is not { } shell)
            return ToolResponse(new("That tool or input is unavailable.", false));
        if (!canRun || turn.IsCancellationRequested || Interlocked.Exchange(ref inUse, 1) != 0)
            return ToolResponse(new("Wait for an active turn with no other PC action.", false));
        try
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(server, turn);
            return ToolResponse(await command.Run(shell, linked.Token));
        }
        catch (Exception error) when (error is PrivateComputerException or System.ComponentModel.Win32Exception or IOException)
        {
            return ToolResponse(new("The approved PC action could not run.", false));
        }
        finally { Volatile.Write(ref inUse, 0); }
    }

    private static object ToolResponse(ThisPcCommandResult result)
    {
        var content = new List<object> { new { type = "text", text = result.Text } };
        if (result.Image is { } image) content.Add(new { type = "image", mimeType = "image/png", data = Convert.ToBase64String(image) });
        return new { isError = !result.Success, content };
    }
}
