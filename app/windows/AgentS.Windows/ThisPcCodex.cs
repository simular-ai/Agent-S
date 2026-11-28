using System.Runtime.CompilerServices;
using System.Text.Json;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>Codex App Server runs without execution environments. Agent S handles its one dynamic tool.</summary>
internal static class ThisPcCodex
{
    public static async IAsyncEnumerable<HarnessEvent> Run(string executable, Conversation owner, string prompt,
        string? resume, string model, string modelUrl, string key, ThisPcCommand command,
        [EnumeratorCancellation] CancellationToken token)
    {
        var root = Path.Combine(AppIdentity.LocalFolder, "codex-this-pc");
        var profile = Path.Combine(root, owner.Provider.ToString());
        if (!Guid.TryParse(owner.Id, out var agentId)) throw new PrivateComputerException("The agent ID is invalid.");
        var work = Path.Combine(root, "work", agentId.ToString("N"));
        Directory.CreateDirectory(profile);
        Directory.CreateDirectory(work);
        var start = ThisPcRpc.StartInfo(executable, work, profile);
        const string provider = "agent_s_this_pc";
        start.Environment["CODEX_HOME"] = profile;
        start.Environment["AGENT_S_PROXY_KEY"] = key;
        foreach (var arg in new[]
        {
            "app-server", "--listen", "stdio://",
            "-c", "web_search=\"disabled\"", "-c", "mcp_servers={}",
            "--disable", "shell_tool", "--disable", "hooks", "--disable", "apps", "--disable", "plugins",
            "--disable", "remote_plugin", "--disable", "multi_agent_v2", "--disable", "code_mode",
            "--disable", "request_permissions_tool",
            "-c", $"model_provider=\"{provider}\"",
            "-c", $"model_providers.{provider}={{name=\"Agent S\",base_url=\"{modelUrl}/v1\",env_key=\"AGENT_S_PROXY_KEY\",wire_api=\"responses\",requires_openai_auth=false,supports_websockets=false}}",
        }) start.ArgumentList.Add(arg);
        await using var rpc = new ThisPcRpc(start, "Codex");
        await rpc.Request("initialize", new
        {
            clientInfo = new { name = "agent_s", title = "Agent S", version = "1.0" },
            capabilities = new { experimentalApi = true, requestAttestation = false },
        }, token);
        await rpc.Write(new { jsonrpc = "2.0", method = "initialized" }, token);

        var options = new Dictionary<string, object>
        {
            ["model"] = model, ["modelProvider"] = provider, ["approvalPolicy"] = "never", ["sandbox"] = "read-only",
        };
        JsonElement opened, thread;
        if (resume is not null)
        {
            if (!Guid.TryParse(resume, out _)) throw new PrivateComputerException("The saved Codex session is invalid.");
            options["threadId"] = resume;
            options["excludeTurns"] = true;
            opened = await rpc.Request("thread/resume", options, token);
            await rpc.Request("thread/settings/update", new { threadId = resume, environments = Array.Empty<object>() }, token);
            thread = (await rpc.Request("thread/read", new { threadId = resume }, token)).Get("thread");
        }
        else
        {
            options["cwd"] = work;
            options["environments"] = Array.Empty<object>();
            options["dynamicTools"] = new[]
            {
                new { type = "function", name = ThisPcCommand.Name, description = ThisPcCommand.Description, inputSchema = ThisPcCommand.Schema },
            };
            options["developerInstructions"] = ThisPcCommand.Instructions;
            opened = await rpc.Request("thread/start", options, token);
            thread = opened.Get("thread");
        }

        var threadId = thread.String("id");
        if (!Guid.TryParse(threadId, out _) || (resume is not null && resume != threadId)
            || !StringComparer.OrdinalIgnoreCase.Equals(thread.String("cwd"), work)
            || thread.Get("environments") is not { ValueKind: JsonValueKind.Array } environments || environments.GetArrayLength() != 0
            || opened.String("modelProvider") != provider || opened.String("approvalPolicy") != "never"
            || opened.Get("sandbox").String("type") != "readOnly" || opened.Get("sandbox").Get("networkAccess").ValueKind != JsonValueKind.False)
            throw new PrivateComputerException("Codex could not start an isolated session. Update Codex and try again.");

        var turn = await rpc.Request("turn/start", new
        {
            threadId, model, input = new[] { new { type = "text", text = prompt, text_elements = Array.Empty<object>() } },
            environments = Array.Empty<object>(), approvalPolicy = "never",
            sandboxPolicy = new { type = "readOnly", networkAccess = false },
        }, token);
        var turnId = turn.Get("turn").String("id") ?? throw new PrivateComputerException("Codex did not start a turn.");
        yield return new HarnessStarted(threadId!);

        while (true)
        {
            var message = await rpc.Next(token);
            var method = message.String("method");
            var p = message.Get("params");
            bool matching = p.String("threadId") == threadId && p.String("turnId") == turnId;
            if (method is not null && message.Get("id") is { ValueKind: not JsonValueKind.Undefined } id)
            {
                if (!matching || method != "item/tool/call" || p.String("tool") != ThisPcCommand.Name
                    || p.Get("namespace").ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null)
                    || p.Get("arguments").String("command") is not { } shell || p.String("callId") is not { } call)
                {
                    await rpc.Write(new { jsonrpc = "2.0", id, error = new { code = -32601, message = "Agent S does not allow this request." } }, token);
                    continue;
                }

                yield return new HarnessToolCall(call, "Bash", shell, shell);
                var result = await command.Run(shell, token);
                var items = new List<object> { new { type = "inputText", text = result.Text } };
                if (result.Image is { } image) items.Add(new { type = "inputImage", imageUrl = "data:image/png;base64," + Convert.ToBase64String(image) });
                await rpc.Write(new { jsonrpc = "2.0", id, result = new { success = result.Success, contentItems = items } }, token);
                yield return new HarnessToolResult(call, result.Text[..Math.Min(result.Text.Length, HarnessStream.MaxToolOutput)], !result.Success);
                continue;
            }

            if (p.String("threadId") != threadId) continue;
            switch (method)
            {
                case "item/agentMessage/delta" when matching && p.String("itemId") is { } itemId && p.String("delta") is { } delta:
                    yield return new HarnessTextDelta(itemId, delta);
                    break;
                case "item/completed" when matching && p.Get("item") is var item && item.String("type") == "agentMessage"
                    && item.String("id") is { } responseId && item.String("text") is { } answer:
                    yield return new HarnessTextSnapshot(responseId, answer);
                    break;
                case "thread/tokenUsage/updated" when matching && p.Get("tokenUsage").Get("last") is var last
                    && last.Get("inputTokens").TryInt(out var input) && last.Get("outputTokens").TryInt(out var output):
                    yield return new HarnessUsage(input, output);
                    break;
                case "turn/completed" when p.Get("turn") is var completed && completed.String("id") == turnId:
                    bool success = completed.String("status") == "completed";
                    yield return new HarnessFinished(success ? null : "The Codex turn did not complete. Check the selected model and provider.", !success);
                    yield break;
            }
        }
    }
}
