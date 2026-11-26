using System.Text;
using System.Text.Json;
using AgentS.Core;

/// <summary>Behavior ported from the macOS app: OpenRouter ZDR, the Ollama library, model defaults and memory estimates.</summary>
internal static class ParityTests
{
    public static void Run(Action<bool, string> check)
    {
        // ---- OpenRouter request policy (OpenRouterRequestPolicyTests.swift) ----
        JsonElement Body(string payload, string path, string method = "POST", bool zdr = true)
        {
            using var document = JsonDocument.Parse(OpenRouterRequestPolicy.Body(Encoding.UTF8.GetBytes(payload), method, path, zdr));
            return document.RootElement.Clone();
        }
        foreach (var path in new[] { "/api/v1/messages", "/api/v1/responses", "/api/v1/chat/completions" })
        {
            var sent = Body("""{"model":"example/model","stream":true,"messages":[{"role":"user","content":"hello"}]}""", path);
            check(sent.GetProperty("provider").GetProperty("zdr").GetBoolean() && sent.GetProperty("model").GetString() == "example/model"
                  && sent.GetProperty("messages").GetArrayLength() == 1, "ZDR is added to " + path + " and the rest is kept");
        }
        var preferences = Body("""{"provider":{"zdr":false,"order":["provider"],"allow_fallbacks":false}}""", "/api/v1/messages").GetProperty("provider");
        check(preferences.GetProperty("zdr").GetBoolean() && preferences.GetProperty("order")[0].GetString() == "provider"
              && !preferences.GetProperty("allow_fallbacks").GetBoolean(), "Only zdr changes in existing provider preferences");
        check(Body("""{"provider":null}""", "/api/v1/responses").GetProperty("provider").GetProperty("zdr").GetBoolean(), "Null provider preferences become ZDR");
        bool Refused(string payload, string method, string path)
        {
            try { OpenRouterRequestPolicy.Body(Encoding.UTF8.GetBytes(payload), method, path, true); return false; }
            catch (OpenRouterPolicyException) { return true; }
        }
        check(new[] { "", "{", "[]" }.All(p => Refused(p, "POST", "/api/v1/responses")), "Malformed or non-object model requests are refused");
        check(Refused("""{"provider":[]}""", "POST", "/api/v1/chat/completions") && Refused("{}", "GET", "/api/v1/future") && Refused("{}", "POST", "/api/v1/future"),
            "Bad provider preferences and other routes are refused");
        var original = Encoding.UTF8.GetBytes("""{ "provider" : { "zdr" : false } }""");
        check(OpenRouterRequestPolicy.Body([], "GET", "/api/v1/models", true).Length == 0
              && OpenRouterRequestPolicy.Body(original, "POST", "/api/v1/future", false) == original, "Body-less discovery passes; with ZDR off nothing changes");

        // ---- ZDR catalog (OpenRouterZDRCatalogTests.swift) ----
        var catalog = new OpenRouterZdrCatalog("""
            {"data":[
              {"model_id":"example/mixed","safety_retention":true},{"model_id":"example/mixed","safety_retention":null},
              {"model_id":"example/mixed","safety_retention":false},{"model_id":"example/mixed"},
              {"model_id":"example/safety","safety_retention":true},{"model_id":"example/safety","safety_retention":true},
              {"model_id":"example/ambiguous","safety_retention":true},{"model_id":"example/ambiguous"}
            ]}
            """);
        check(catalog.Availability("example/mixed", true) == OpenRouterZdrAvailability.Available
              && catalog.Availability("example/safety", true) == OpenRouterZdrAvailability.SafetyRetentionOnly
              && catalog.Availability("example/ambiguous", true) == OpenRouterZdrAvailability.Unknown, "ZDR availability per model");
        check(catalog.Availability("example/other", true) == OpenRouterZdrAvailability.Unavailable
              && catalog.Availability("example/other", false) == OpenRouterZdrAvailability.Unknown, "A public model without a ZDR endpoint is unavailable");
        check(OpenRouterZdrCatalog.Label(OpenRouterZdrAvailability.Available) == "ZDR available"
              && OpenRouterZdrCatalog.Warning(OpenRouterZdrAvailability.Available) is null, "ZDR labels");
        var models = OpenRouterModel.Parse("""{"data":[{"id":"b/two","name":"Two","context_length":200000,"architecture":{"input_modalities":["text","image"]}},{"id":"a/one","name":"One"},{"name":"no id"}]}""");
        check(models.Select(m => m.Id).SequenceEqual(["a/one", "b/two"]) && models[1].ContextText == "200K context"
              && models[1].Matches("two B/") && !models[1].Matches("one"), "OpenRouter models are listed and searchable");

        var priced = OpenRouterModel.Parse("""{"data":[{"id":"a/paid","name":"Paid","pricing":{"prompt":"0.000003","completion":"0.000015"},"architecture":{"input_modalities":["text","image"]}},{"id":"b/free","pricing":{"prompt":"0","completion":"0"}},{"id":"c/odd","pricing":{"prompt":"x"}}]}""");
        check(priced[0].PriceText == "$3 / $15 per M tokens" && priced[1].PriceText == "Free" && priced[2].PriceText == "" && priced[0].Input == ModelInput.Multimodal,
            "OpenRouter models show their price per million tokens and inputs");

        // ---- Ollama models (Ollama.swift, AppModel.swift) ----
        check(OllamaModels.Default == "nemotron-3-nano:4b" && OllamaModels.Suggested.SequenceEqual(["nemotron-3-nano:4b", "nemotron-3.5-bolt:e4b", "gemma4"]),
            "Nemotron 3 Nano 4B is the default model, as on macOS");
        check(OllamaModels.DisplayName("gemma4:latest") == "Gemma 4" && OllamaModels.DisplayName("qwen3") == "qwen3", "Model display names");
        check(OllamaModels.MinimumVersionFor(OllamaModels.NemotronBoltE4B) == "0.40.10" && OllamaModels.MinimumVersionFor("gemma4") == "0.14.0", "Minimum Ollama versions");
        check(OllamaModels.IsAtLeast("0.14.0", "0.14.0") && OllamaModels.IsAtLeast("0.40.12-rc1", "0.40.10") && !OllamaModels.IsAtLeast("0.13.9", "0.14.0"), "Ollama version comparison");
        check(OllamaModels.IsCloudReference("gemma4:cloud") && OllamaModels.IsCloudReference("x:31b-cloud") && !OllamaModels.IsCloudReference("gemma4"), "Cloud model references");
        var tags = OllamaModels.ParseTags("""{"models":[{"name":"gemma4:latest","size":9600000000},{"name":"glm:cloud","remote_host":"https://ollama.com"},{"name":"x","remote_model":""}]}""");
        check(tags.Length == 3 && !tags[0].Remote && tags[0].Size == 9_600_000_000 && tags[1].Remote && !tags[2].Remote, "Installed models and remote inference");

        // ---- Memory estimate (OllamaMemoryEstimateTests.swift) ----
        const ulong GiB = 1UL << 30;
        var range = OllamaMemoryEstimate.FromCatalogSize("2.8 GB – 2.9 GB")!;
        var required = 2 * GiB + GiB + 2_900_000_000;
        check(!range.ExceedsBudget(required, 2 * GiB) && range.ExceedsBudget(required - 1, 2 * GiB), "The upper end of a size range counts");
        check(!OllamaMemoryEstimate.FromCatalogSize("1 GB")!.ExceedsBudget(GiB + 1_000_000_000, 0)
              && OllamaMemoryEstimate.FromCatalogSize("1 GiB")!.ExceedsBudget(GiB + 1_000_000_000, 0), "Decimal and binary sizes differ");
        var three = OllamaMemoryEstimate.FromBytes(3 * GiB)!;
        check(!three.ExceedsBudget(4 * GiB, 0) && three.ExceedsBudget(4 * GiB, 2 * GiB) && !three.ExceedsBudget(6 * GiB, 2 * GiB), "Only a sandbox that still has to start counts");
        var large = OllamaMemoryEstimate.FromBytes(8 * GiB + 1)!;
        check(!large.ExceedsBudget(10 * GiB + 2, 0) && large.ExceedsBudget(10 * GiB + 1, 0), "Headroom is a quarter of a large model, rounded up");
        check(new[] { "", "Unknown", "Medium Usage", "0 GB", "-2 GB", "32b parameters", "18 GB – Unknown", "18446744073709551616 B" }
              .All(s => OllamaMemoryEstimate.FromCatalogSize(s) is null) && OllamaMemoryEstimate.FromBytes(0) is null, "Unknown sizes have no estimate");
        check(OllamaMemoryEstimate.FromBytes(ulong.MaxValue)!.ExceedsBudget(ulong.MaxValue, 0) && three.ExceedsBudget(8 * GiB, ulong.MaxValue), "Out-of-range budgets warn");
        check(OllamaMemoryEstimate.FormatGiB(6 * GiB) == "6 GB" && OllamaMemoryEstimate.FormatGiB((ulong)(9.6 * GiB)) == "9.6 GB", "Memory is shown like macOS");

        // ---- Pull progress ----
        var pull = new OllamaPullProgress();
        check(pull.Update("pulling manifest", null, null, null) is null, "No fraction before a layer size is known");
        pull.Update("pulling a", "a", 300, null);
        check(pull.Update("pulling b", "b", 100, 100) is 0.25 && pull.Update("pulling a", "a", 300, 300) is 1.0 && pull.Update("success", null, null, null) is 1.0,
            "Progress counts every announced layer");

        // ---- Ollama library (OllamaCatalogTests.swift) ----
        var families = new[] { "nemotron-3-nano", "gemma4", "qwen3.5" }.Concat(Enumerable.Range(1, 25).Select(i => $"model-{i}")).ToArray();
        var html = "<nav><a href=\"/library/featured\">Featured</a></nav><div id=\"repo\">\n<ul role=\"list\">"
            + string.Concat(families.Select(f => $"<li><a href=\"/library/{f}\">{f}</a></li>\n"))
            + "<li><a href=\"/library/gemma4\">Duplicate</a></li><li><a href=\"/library/gemma4:latest\">Tag</a></li><li><a href=\"https://example.com/library/model\">X</a></li></ul></div>";
        check(OllamaCatalog.ParseFamilies(html).SequenceEqual(families), "The library keeps its order and only its own list");
        check(OllamaCatalog.TagsUrl("qwen3.5").AbsoluteUri == "https://ollama.com/library/qwen3.5/tags", "Tags page address");
        check(new[] { "", "../gemma4", "owner/model", "gemma4:4b", " qwen3.5", "gemma4%2Ftags", "gemma4?x=1" }.All(bad =>
        { try { OllamaCatalog.TagsUrl(bad); return false; } catch (LocalAgentException) { return true; } }), "Only one exact family name is accepted");
        var variants = OllamaCatalog.ParseVariants("""
            <p class="block col-span-6 md:hidden">4 models</p>
            <a href="/library/gemma4" class="md:hidden flex group"><svg title="Cloud"></svg>Medium Usage</a>
            <a href="/library/gemma4:4b" class="md:hidden flex group">
              <span>28e2e9e36b77</span> • 2.8GB - 2.9GB • 256K context window
            </a>
            <a href="/library/gemma4:4b" class="group-hover:underline">gemma4:4b</a>
            <a href="/library/gemma4:4b-q4_K_M" class="md:hidden flex group"><span>28e2e9e36b77</span> • 420MB • 256K context window</a>
            <a href="/library/gemma4:large-cloud" class="md:hidden flex group">Medium Usage</a>
            """, "gemma4");
        check(variants.Select(v => v.Name).SequenceEqual(["gemma4:4b", "gemma4:4b-q4_K_M"]) && variants.Select(v => v.Size).SequenceEqual(["2.8 GB – 2.9 GB", "420 MB"]),
            "Only local variants are listed, with download sizes");
        var inputs = OllamaCatalog.ParseVariants("""
            <p>3 models</p><span>vision</span>
            <a href="/library/model:one" class="md:hidden flex group"> • 2.8GB • 128K context window • <span class="hidden sm:inline">
                Text, Image input • 1 month ago</span></a>
            <a href="/library/model:two" class="md:hidden flex group"> • 2.9GB • Text input • 2 months ago</a>
            <a href="/library/model:three" class="md:hidden flex group">Missing input information</a>
            """, "model");
        check(inputs.Select(v => v.Input).SequenceEqual(new ModelInput?[] { ModelInput.Multimodal, ModelInput.TextOnly, null }), "Each variant's inputs");
        var hosted = """
            <p class="block col-span-6 md:hidden">3 models</p>
            <a href="/library/deepseek-v4.1-flash:cloud" class="md:hidden flex group">Medium Usage</a>
            <a href="/library/deepseek-v4.1-flash:1017-cloud" class="md:hidden flex group">Medium Usage</a>
            <a href="/library/deepseek-v4.1-flash:latest" class="md:hidden flex group">567966df89fe • Medium Usage • 1M context window</a>
            """;
        check(OllamaCatalog.ParseVariants(hosted, "deepseek-v4.1-flash").Length == 0, "A hosted-only family has no downloads");
        bool Unreadable(Action parse) { try { parse(); return false; } catch (LocalAgentException) { return true; } }
        check(Unreadable(() => OllamaCatalog.ParseVariants(hosted, "gemma4")) && Unreadable(() => OllamaCatalog.ParseFamilies("<h1>unavailable</h1>")),
            "An unreadable page is an error, not an empty list");

        // ---- Tool step titles (ToolActivityTests.swift) ----
        check(ToolActivity.Title("Bash", "web open https://x.test", "web open https://x.test") == "Open a webpage"
              && ToolActivity.Title("Bash", "", "cd ~/work && web snapshot | head -40") == "Inspect the webpage"
              && ToolActivity.Title("Shell", "", "/bin/bash -lc 'web open https://x.test && sleep 2 && web read'") == "Open and read a webpage",
            "Browser commands read as webpage actions");
        check(ToolActivity.Title("Bash", "", "ui screenshot") == "Capture the computer screen" && ToolActivity.Title("Bash", "", "C:/Users/a/agent-s-tools/pc click 4") == "Click a control"
              && ToolActivity.Title("Bash", "", "cat > notes.txt") == "Write a file" && ToolActivity.Title("Bash", "", "ls -la") == "List files",
            "ui, pc and file commands get plain titles");
        check(ToolActivity.Title("Bash", "", "curl $(whoami)") == "Run a terminal command" && ToolActivity.Title("Read", "x") == "Read a file"
              && ToolActivity.Title("mcp__box__browser_navigate", "https://x.test") == "Open a webpage" && ToolActivity.Title("mystery", "") == "Use a tool",
            "Anything else gets a generic title");
        check(ToolActivity.Title("mcp__box__computer_act", "Check Saturday opening hours") == "Check Saturday opening hours"
              && ToolActivity.Title("mcp__box__computer_act", "rm -rf /tmp/x") == "Use the computer", "A provider's prose title is kept, a raw command isn't");
        using (var input = JsonDocument.Parse("""{"description":"x","command":"web read"}"""))
            check(ToolActivity.Detail(input.RootElement) == "web read", "Step details show the command");

        // ---- Take over ----
        check(TakeOverInput.ToScreen(0, 0, 1024, 768) == (0, 0) && TakeOverInput.ToScreen(512, 384, 512, 384) is null
              && TakeOverInput.ToScreen(256, 192, 512, 384) == (512, 384), "Screen points scale to guest pixels");
        // 800×300 view: the 4:3 picture is 400×300, centered with 200px bars.
        check(TakeOverInput.ToScreen(100, 150, 800, 300) is null && TakeOverInput.ToScreen(400, 150, 800, 300) == (512, 384)
              && TakeOverInput.ToScreen(5, 5, 0, 0) is null, "Letterbox bars and empty views map to nothing");
        check(TakeOverInput.Move(-5, 9000) == "move 0 767" && TakeOverInput.Button(3, true) == "down 3" && TakeOverInput.Button(1, false) == "up 1",
            "Moves are clamped to the screen and buttons are numbered");
        check(TakeOverInput.Wheel(120, false) == "wheel 4" && TakeOverInput.Wheel(-120, false) == "wheel 5" && TakeOverInput.Wheel(1, true) == "wheel 7",
            "Wheel notches become X buttons 4–7");
        check(TakeOverInput.Key(["ctrl", "shift"], "Tab") == "key ctrl+shift+Tab" && TakeOverInput.Key(["ctrl"], "l") == "key ctrl+l"
              && TakeOverInput.Key([], "F5") == "key F5", "Key chords");
        check(TakeOverInput.Key([], "a; rm -rf ~") is null && TakeOverInput.Key(["hyper"], "a") is null && TakeOverInput.Key([], "L") is null,
            "Anything outside the key list is refused");
        check(TakeOverInput.Text("héllo\n") == "text " + Convert.ToBase64String(Encoding.UTF8.GetBytes("héllo\n"))
              && TakeOverInput.Text("") is null && TakeOverInput.Text(new string('a', 4097)) is null, "Text is base64 and bounded");
    }
}
