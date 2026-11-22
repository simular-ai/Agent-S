using System.Buffers.Binary;
using System.IO.Pipelines;
using System.Text;
using System.Text.Json;
using AgentS.Core;

/// <summary>The harness commands, stream parsers and Ollama relay, against fixtures (no guest, no model).</summary>
internal static class HarnessTests
{
    public static async Task Run(Action<bool, string> check)
    {
        // ---- Saved agents ----
        var older = JsonSerializer.Deserialize<Conversation>("""{"Name":"Private agent","Target":"Private","ModelId":"gemma4"}""")!;
        check(older.Harness is null && older.HarnessSessions.Count == 0, "Existing Private agents keep the built-in loop");
        var saved = new Conversation { Target = ConversationTarget.Private, Harness = PrivateHarness.Codex };
        saved.HarnessSessions[PrivateHarness.Codex] = "thread-1";
        var json = JsonSerializer.Serialize(saved);
        var restored = JsonSerializer.Deserialize<Conversation>(json)!;
        check(json.Contains("\"Codex\"") && restored.Harness == PrivateHarness.Codex && restored.HarnessSessions[PrivateHarness.Codex] == "thread-1",
            "Harness and its session are saved by name and restored");

        // ---- Commands ----
        const string tricky = "Say 'hi' && rm -rf ~ $(whoami) `id`";
        var claude = HarnessCommand.Build(PrivateHarness.ClaudeCode, tricky, null, "gemma4");
        check(claude.Contains(HarnessCommand.Quote(tricky)) && HarnessCommand.Quote("it's") == "'it'\\''s'", "Prompts are single-quoted for the guest shell");
        check(claude.Contains("ANTHROPIC_BASE_URL='http://127.0.0.1:11434'") && claude.Contains("--output-format stream-json") &&
              claude.Contains("--permission-mode bypassPermissions") && claude.Contains("--tools Bash,Read,Write,Edit") && !claude.Contains("--resume"),
            "Claude Code runs headless on the relayed Ollama with the small tool set");
        check(claude.TrimEnd().EndsWith("</dev/null", StringComparison.Ordinal), "Claude Code can't read the rest of the turn script from stdin");
        check(HarnessCommand.Build(PrivateHarness.ClaudeCode, "go", "abc", "gemma4").Contains("--resume 'abc'"), "Claude Code resumes its session");

        var codex = HarnessCommand.Build(PrivateHarness.Codex, "go", "t-1", "gemma4");
        check(codex.Contains("codex exec --json --oss --local-provider ollama") && codex.Contains("--dangerously-bypass-approvals-and-sandbox") &&
              codex.Contains("'web_search=\"disabled\"'") && codex.Contains("> AGENTS.md"), "Codex runs --oss on Ollama with web search off");
        check(codex.IndexOf("--skip-git-repo-check", StringComparison.Ordinal) < codex.IndexOf("resume 't-1'", StringComparison.Ordinal),
            "Codex options come before resume");

        var dsh = HarnessCommand.Build(PrivateHarness.DeepSeek, "go", null, "gemma4");
        check(dsh.Contains("dsh --profile headless") && dsh.Contains("DSH_TELEMETRY_MODE=DISABLED"), "DeepSeek runs headless without telemetry");
        var patch = HarnessCommand.DeepSeekPatch("gemma4\"\n- id: evil !!js process.exit()");
        check(!patch.Contains("!!js") && !patch.Contains("\n- id: evil") && patch.Contains("model: \"gemma4-id:eviljsprocess.exit\""),
            "DeepSeek model id can't break out of its YAML scalar");
        check(patch.Contains("session-log-deepseek\n  disabled: true") && patch.Contains("web-search-deepseek\n  disabled: true"),
            "DeepSeek session upload and web search stay off");
        check(HarnessCommand.ContextModel("gemma4:latest") == "agent-s/gemma4-latest:32k" &&
              HarnessCommand.ContextModel("Some/Model:7B q4") == "agent-s/some-model-7b-q4:32k",
            "Harnesses use an app-owned 32K-context variant of the chosen model");
        // OpenRouter in the Private computer: the guest reaches the app's proxy on 11435 with its token, never the key.
        var orClaude = HarnessCommand.Build(PrivateHarness.ClaudeCode, "go", null, "anthropic/claude-sonnet-5.5", ModelProvider.OpenRouter, "agent-s-token");
        check(orClaude.Contains("ANTHROPIC_BASE_URL='http://127.0.0.1:11435'") && orClaude.Contains("ANTHROPIC_AUTH_TOKEN='agent-s-token'") &&
              orClaude.Contains("CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000"), "Claude Code on OpenRouter uses the relayed proxy and a hosted window");
        var orCodex = HarnessCommand.Build(PrivateHarness.Codex, "go", null, "openai/gpt-oss-120b", ModelProvider.OpenRouter, "agent-s-token");
        check(!orCodex.Contains("--oss") && orCodex.Contains("model_provider=\"saige\"") && orCodex.Contains("base_url=\"http://127.0.0.1:11435/v1\"") &&
              orCodex.Contains("wire_api=\"responses\"") && orCodex.Contains("AGENT_S_PROXY_KEY='agent-s-token'"), "Codex on OpenRouter uses a Responses provider at the proxy");
        var orDsh = HarnessCommand.Build(PrivateHarness.DeepSeek, "go", null, "qwen/qwen3-coder", ModelProvider.OpenRouter, "agent-s-token");
        check(orDsh.Contains("baseURL: http://127.0.0.1:11435/v1") && orDsh.Contains("AGENT_S_PROXY_KEY='agent-s-token'") && orDsh.Contains("id: \"qwen/qwen3-coder\""),
            "DeepSeek on OpenRouter points its provider at the proxy");
        try { HarnessCommand.Build(PrivateHarness.Codex, "go", null, "m", ModelProvider.OpenRouter); check(false, "OpenRouter without the proxy token is refused"); }
        catch (ArgumentException) { check(true, "OpenRouter without the proxy token is refused"); }
        try { HarnessCommand.Build(PrivateHarness.AgentS, "go", null, "gemma4"); check(false, "Built-in harness has no guest command"); }
        catch (ArgumentException) { check(true, "Built-in harness has no guest command"); }
        foreach (var harness in new[] { PrivateHarness.ClaudeCode, PrivateHarness.Codex, PrivateHarness.DeepSeek })
            check(AgentInstructions.Text(harness).Contains("1024x768") && !AgentInstructions.Text(harness).Contains("/mnt/sai"),
                $"{harness} instructions describe the WSL screen");

        // ---- Claude Code stream ----
        List<HarnessEvent> Parse(PrivateHarness harness, params string[] lines) => lines.SelectMany(l => HarnessStream.Parse(harness, l)).ToList();
        var events = Parse(PrivateHarness.ClaudeCode,
            """{"type":"system","subtype":"init","session_id":"s-1","model":"gemma4"}""",
            """{"type":"assistant","message":{"content":[{"type":"text","text":"Opening it."},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"web open https://example.com\nmore"}}]}}""",
            """{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"FAILED"}],"is_error":true}]}}""",
            """{"type":"result","result":"Done","is_error":false,"usage":{"input_tokens":10,"cache_read_input_tokens":5,"output_tokens":3}}""",
            "not json", """{"type":"rate_limit_event"}""");
        check(events.Count == 6 && events[0] == new HarnessStarted("s-1") && events[1] == new HarnessText("Opening it.") &&
              events[2] == new HarnessToolCall("t1", "Bash", "web open https://example.com") &&
              events[3] == new HarnessToolResult("t1", "FAILED", true) && events[4] == new HarnessUsage(15, 3) &&
              events[5] == new HarnessFinished("Done", false), "Claude Code stream-json becomes chat events; unknown lines are skipped");

        // ---- Codex stream ----
        events = Parse(PrivateHarness.Codex,
            """{"type":"thread.started","thread_id":"th-9"}""",
            """{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"/bin/bash -lc 'web snapshot'"}}""",
            """{"type":"item.completed","item":{"id":"i1","type":"command_execution","aggregated_output":"ok","exit_code":0}}""",
            """{"type":"item.completed","item":{"id":"i2","type":"reasoning","text":"thinking"}}""",
            """{"type":"item.completed","item":{"id":"i3","type":"agent_message","text":"All done."}}""",
            """{"type":"turn.completed","usage":{"input_tokens":20,"cached_input_tokens":8,"output_tokens":4}}""");
        check(events.Count == 6 && events[0] == new HarnessStarted("th-9") && events[1] == new HarnessToolCall("i1", "Shell", "web snapshot") &&
              events[2] == new HarnessToolResult("i1", "ok", false) && events[3] == new HarnessText("All done.") &&
              events[4] == new HarnessUsage(20, 4) && events[5] == new HarnessFinished(null, false), "Codex --json becomes chat events");
        check(Parse(PrivateHarness.Codex, """{"type":"turn.failed","error":{"message":"model not found"}}""").Single() == new HarnessFinished("model not found", true),
            "A failed Codex turn is reported as an error");

        // ---- DeepSeek stream ----
        events = Parse(PrivateHarness.DeepSeek,
            """{"type":"session","sessionId":"session-1"}""",
            """{"type":"tool_call","callId":"c1","tool":"bash","input":{"command":"web read","description":"web read"}}""",
            """{"type":"tool_result","callId":"c1","status":"failed","result":{"code":1}}""",
            """{"type":"text","text":"Here it is."}""",
            """{"type":"status","phase":"step_end","usage":{"inputTokens":7,"cacheReadTokens":1,"outputTokens":2}}""",
            """{"type":"final","reason":{"kind":"max_steps"}}""");
        check(events.Count == 6 && events[0] == new HarnessStarted("session-1") && events[1] == new HarnessToolCall("c1", "bash", "web read") &&
              events[2] is HarnessToolResult { IsError: true } && events[3] == new HarnessText("Here it is.") &&
              events[4] == new HarnessUsage(8, 2) && events[5] == new HarnessFinished("DeepSeek Harness stopped: max_steps", true),
            "DeepSeek --json becomes chat events and an unfinished turn is an error");
        var long_ = new string('x', HarnessStream.MaxToolOutput + 50);
        var clipped = (HarnessToolResult)Parse(PrivateHarness.DeepSeek, $$"""{"type":"tool_result","callId":"c","status":"completed","result":"{{long_}}"}""").Single();
        check(clipped.Output.Length < long_.Length && !clipped.IsError, "Long tool output is clipped for the chat");

        // ---- Ollama relay ----
        await RelayRoundTrip(check);
    }

    private static async Task RelayRoundTrip(Action<bool, string> check)
    {
        var toHost = new Pipe(); var toGuest = new Pipe();
        var ollama = new Dictionary<int, (Pipe In, Pipe Out)>();
        int opened = 0;
        var ports = new List<int>();
        var relay = new OllamaRelay(toHost.Reader.AsStream(), toGuest.Writer.AsStream(), (port, _) =>
        {
            ports.Add(port);
            var pair = (In: new Pipe(), Out: new Pipe());
            ollama[++opened] = pair;
            return Task.FromResult<Stream>(new Duplex(pair.Out.Reader.AsStream(), pair.In.Writer.AsStream()));
        });
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var running = relay.Run(cancel.Token);
        var guest = toHost.Writer.AsStream();
        async Task Frame(byte type, uint id, string payload = "")
        {
            var bytes = Encoding.UTF8.GetBytes(payload);
            var header = new byte[9]; header[0] = type;
            BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(1), id);
            BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(5), (uint)bytes.Length);
            await guest.WriteAsync(header); await guest.WriteAsync(bytes); await guest.FlushAsync();
        }
        await Frame(OllamaRelay.Open, 7);
        await Frame(OllamaRelay.Data, 7, "GET /api/version");
        // Ollama receives what the guest sent on connection 7 (once the relay has connected it).
        while (!ollama.ContainsKey(1)) await Task.Delay(10, cancel.Token);
        var request = new byte[16];
        await ollama[1].In.Reader.AsStream().ReadExactlyAsync(request, cancel.Token);
        check(Encoding.UTF8.GetString(request) == "GET /api/version", "Relay forwards a guest connection to Ollama");
        // Ollama answers and closes; the guest gets the data then a close for the same connection.
        var answer = ollama[1].Out.Writer.AsStream();
        await answer.WriteAsync(Encoding.UTF8.GetBytes("{\"version\":\"1\"}"), cancel.Token); await answer.FlushAsync(cancel.Token);
        await ollama[1].Out.Writer.CompleteAsync();
        var back = toGuest.Reader.AsStream();
        var header = new byte[9];
        await back.ReadExactlyAsync(header, cancel.Token);
        var body = new byte[BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(5))];
        await back.ReadExactlyAsync(body, cancel.Token);
        check(header[0] == OllamaRelay.Data && BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(1)) == 7 &&
              Encoding.UTF8.GetString(body) == "{\"version\":\"1\"}", "Relay returns Ollama's answer on the same connection");
        await back.ReadExactlyAsync(header, cancel.Token);
        check(header[0] == OllamaRelay.Close && BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(1)) == 7, "Relay closes the guest connection when Ollama does");
        // A connection to the guest's 11435 carries that port, so the host connects it to the OpenRouter proxy.
        var open = new byte[11]; open[0] = OllamaRelay.Open;
        BinaryPrimitives.WriteUInt32BigEndian(open.AsSpan(1), 8);
        BinaryPrimitives.WriteUInt32BigEndian(open.AsSpan(5), 2);
        BinaryPrimitives.WriteUInt16BigEndian(open.AsSpan(9), HarnessCommand.OpenRouterGuestPort);
        await guest.WriteAsync(open); await guest.FlushAsync();
        while (ports.Count < 2) await Task.Delay(10, cancel.Token);
        check(ports[0] == HarnessCommand.OllamaGuestPort && ports[1] == HarnessCommand.OpenRouterGuestPort, "Relay tells Ollama and OpenRouter connections apart by guest port");
        await toHost.Writer.CompleteAsync();
        await running;
        check(true, "Relay ends when the guest closes its side");
    }

    private sealed class Duplex(Stream read, Stream write) : Stream
    {
        public override bool CanRead => true; public override bool CanSeek => false; public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => write.Flush();
        public override Task FlushAsync(CancellationToken token) => write.FlushAsync(token);
        public override int Read(byte[] buffer, int offset, int count) => read.Read(buffer, offset, count);
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken token = default) => read.ReadAsync(buffer, token);
        public override void Write(byte[] buffer, int offset, int count) => write.Write(buffer, offset, count);
        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken token = default) => write.WriteAsync(buffer, token);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }
}
