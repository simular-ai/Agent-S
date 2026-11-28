using System.Text.Json;
using System.Text.Json.Nodes;

namespace AgentS.Core;

/// <summary>Why the OpenRouter proxy refused a request while Zero Data Retention is required.</summary>
public sealed class OpenRouterPolicyException(string message) : Exception(message);

/// <summary>
/// The macOS app's OpenRouter request policy: with ZDR required, only inference POSTs pass, each with
/// <c>"provider": {"zdr": true}</c> merged into its body, so OpenRouter routes only to ZDR endpoints.
/// </summary>
public static class OpenRouterRequestPolicy
{
    private static readonly HashSet<string> InferencePaths = ["/api/v1/messages", "/api/v1/responses", "/api/v1/chat/completions"];

    /// <param name="path">The upstream URL's path (e.g. <c>/api/v1/messages</c>), without its query.</param>
    public static byte[] Body(byte[] body, string method, string path, bool requiresZdr)
    {
        if (!requiresZdr) return body;
        if (method is "GET" or "HEAD" && body.Length == 0) return body;
        if (method != "POST" || !InferencePaths.Contains(path))
            throw new OpenRouterPolicyException("Zero Data Retention is required, so Agent S only forwards model requests to OpenRouter.");
        JsonObject payload;
        try { payload = JsonNode.Parse(body) as JsonObject ?? throw new OpenRouterPolicyException("The model request isn't a JSON object."); }
        catch (JsonException) { throw new OpenRouterPolicyException("The model request isn't valid JSON."); }
        var provider = payload["provider"] switch
        {
            null => new JsonObject(),
            JsonObject preferences => preferences,
            _ => throw new OpenRouterPolicyException("The model request's provider preferences aren't an object."),
        };
        provider["zdr"] = true;
        payload["provider"] = provider;
        return JsonSerializer.SerializeToUtf8Bytes(payload);
    }
}

public enum OpenRouterZdrAvailability { Available, SafetyRetentionOnly, Unavailable, Unknown }

/// <summary>OpenRouter's public ZDR endpoint catalog (<c>GET /api/v1/endpoints/zdr</c>), per model.</summary>
public sealed class OpenRouterZdrCatalog
{
    public static readonly Uri Url = new("https://openrouter.ai/api/v1/endpoints/zdr");
    private readonly Dictionary<string, OpenRouterZdrAvailability> byModel;

    public OpenRouterZdrCatalog(string json)
    {
        using var document = JsonDocument.Parse(json);
        var endpoints = document.RootElement.GetProperty("data").EnumerateArray().Select(e =>
        {
            var id = e.TryGetProperty("model_id", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString()! : "";
            bool? retention = e.TryGetProperty("safety_retention", out var r) && r.ValueKind is JsonValueKind.True or JsonValueKind.False ? r.GetBoolean() : null;
            return (Id: id, Retention: retention);
        }).ToArray();
        if (endpoints.Any(e => e.Id.Length == 0 || e.Id.Any(char.IsWhiteSpace))) throw new JsonException("Invalid OpenRouter model ID");
        byModel = endpoints.GroupBy(e => e.Id).ToDictionary(g => g.Key, g =>
            g.Any(e => e.Retention == false) ? OpenRouterZdrAvailability.Available
            : g.All(e => e.Retention == true) ? OpenRouterZdrAvailability.SafetyRetentionOnly
            : OpenRouterZdrAvailability.Unknown);
    }

    public OpenRouterZdrAvailability Availability(string id, bool isPublicModel) =>
        byModel.TryGetValue(id, out var found) ? found : isPublicModel ? OpenRouterZdrAvailability.Unavailable : OpenRouterZdrAvailability.Unknown;

    public static string Label(OpenRouterZdrAvailability availability) => availability switch
    {
        OpenRouterZdrAvailability.Available => "ZDR available",
        OpenRouterZdrAvailability.SafetyRetentionOnly => "Safety Retention only",
        OpenRouterZdrAvailability.Unavailable => "No public ZDR endpoint",
        _ => "ZDR unverified",
    };

    /// <summary>The warning shown for a model under required ZDR, or null when it should work.</summary>
    public static string? Warning(OpenRouterZdrAvailability availability) => availability switch
    {
        OpenRouterZdrAvailability.SafetyRetentionOnly => "This model's ZDR endpoints require allowing Safety Retention in OpenRouter.",
        OpenRouterZdrAvailability.Unavailable or OpenRouterZdrAvailability.Unknown => "OpenRouter may reject this model when ZDR is requested.",
        _ => null,
    };
}

/// <summary>A model in OpenRouter's public list: what the model browser shows.</summary>
/// <param name="PromptPrice">US dollars per million input tokens (null when OpenRouter doesn't say).</param>
public sealed record OpenRouterModel(string Id, string Name, int? ContextLength, string[] InputModalities,
    decimal? PromptPrice = null, decimal? CompletionPrice = null, bool VariablePrice = false)
{
    public static readonly Uri ListUrl = new("https://openrouter.ai/api/v1/models");

    /// <summary>The macOS label: "$3.00 in · $15.00 out / 1M tokens", "Free text tokens", "Varies by routed model", "Price unavailable".</summary>
    public string PriceText => PriceLabel(Id, PromptPrice, CompletionPrice, VariablePrice);

    /// <summary>The price label for a model id, also before the list has loaded (openrouter/auto always varies).</summary>
    public static string PriceLabel(string id, decimal? input = null, decimal? output = null, bool variable = false) =>
        variable || id == "openrouter/auto" ? "Varies by routed model"
        : (input, output) switch
        {
            (0m, 0m) => "Free text tokens",
            ({ } i, { } o) => $"{Money(i)} in · {Money(o)} out / 1M tokens",
            _ => "Price unavailable",
        };

    /// <summary>US dollars with 2 to 8 decimals, as macOS formats them.</summary>
    private static string Money(decimal value) => "$" + value.ToString("#,0.00######", System.Globalization.CultureInfo.InvariantCulture);

    public ModelInput? Input => OllamaCatalog.FromModalities(InputModalities);

    /// <summary>OpenRouter's per-token price (a decimal string); -1 means it varies by the routed model.</summary>
    private static decimal? PerToken(JsonElement model, string field) =>
        model.TryGetProperty("pricing", out var pricing) && pricing.ValueKind == JsonValueKind.Object
        && pricing.TryGetProperty(field, out var value) && value.ValueKind == JsonValueKind.String
        && decimal.TryParse(value.GetString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var perToken)
        ? perToken : null;
    private static decimal? PerMillion(decimal? perToken) => perToken is >= 0 ? perToken * 1_000_000m : null;
    public static OpenRouterModel[] Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        return document.RootElement.GetProperty("data").EnumerateArray().Select(m =>
        {
            string? Str(JsonElement e, string key) => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
            var id = Str(m, "id");
            if (string.IsNullOrWhiteSpace(id)) return null;
            int? context = m.TryGetProperty("context_length", out var c) && c.ValueKind == JsonValueKind.Number && c.TryGetInt32(out var n) ? n : null;
            var inputs = m.TryGetProperty("architecture", out var a) && a.ValueKind == JsonValueKind.Object && a.TryGetProperty("input_modalities", out var i) && i.ValueKind == JsonValueKind.Array
                ? i.EnumerateArray().Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!).ToArray() : [];
            decimal? prompt = PerToken(m, "prompt"), completion = PerToken(m, "completion");
            return new OpenRouterModel(id, Str(m, "name") ?? id, context, inputs, PerMillion(prompt), PerMillion(completion),
                VariablePrice: prompt == -1 || completion == -1);
        }).OfType<OpenRouterModel>().OrderBy(m => m.Id, StringComparer.Ordinal).ToArray();
    }

    /// <summary>Matches a search by name, provider or id, ignoring case.</summary>
    public bool Matches(string query) => string.IsNullOrWhiteSpace(query)
        || query.Split(' ', StringSplitOptions.RemoveEmptyEntries).All(word =>
            Id.Contains(word, StringComparison.OrdinalIgnoreCase) || Name.Contains(word, StringComparison.OrdinalIgnoreCase));

    /// <summary>"131K context", "1M context", as macOS's compact numbers.</summary>
    public string ContextText => ContextLength is { } tokens ? $"{Compact(tokens)} context" : "";

    private static string Compact(int value) => value switch
    {
        >= 1_000_000 => (value / 1_000_000d).ToString(value >= 10_000_000 ? "0" : "0.#", System.Globalization.CultureInfo.InvariantCulture) + "M",
        >= 1_000 => (value / 1_000d).ToString(value >= 10_000 ? "0" : "0.#", System.Globalization.CultureInfo.InvariantCulture) + "K",
        _ => value.ToString(System.Globalization.CultureInfo.InvariantCulture),
    };
}

/// <summary>
/// OpenRouter's public model list with each model's ZDR availability, loaded together and cached for
/// 10 minutes. Neither request needs the key.
/// </summary>
public sealed class OpenRouterCatalog(HttpMessageHandler? handler = null)
{
    private readonly HttpClient http = new(handler ?? new SocketsHttpHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(30) };
    private (DateTime Until, OpenRouterModel[] Models, OpenRouterZdrCatalog? Zdr)? cached;

    /// <returns>The models (empty if the list couldn't be read) and the ZDR catalog (null if it couldn't be read).</returns>
    public async Task<(OpenRouterModel[] Models, OpenRouterZdrCatalog? Zdr)> Load(CancellationToken cancellation = default, bool force = false)
    {
        if (!force && cached is { } fresh && fresh.Until > DateTime.UtcNow) return (fresh.Models, fresh.Zdr);
        var models = Task.Run(async () =>
        {
            try { return OpenRouterModel.Parse(await http.GetStringAsync(OpenRouterModel.ListUrl, cancellation)); }
            catch (Exception error) when (error is HttpRequestException or JsonException or KeyNotFoundException or InvalidOperationException or TaskCanceledException) { return []; }
        }, cancellation);
        var zdr = Task.Run(async () =>
        {
            try { return new OpenRouterZdrCatalog(await http.GetStringAsync(OpenRouterZdrCatalog.Url, cancellation)); }
            catch (Exception error) when (error is HttpRequestException or JsonException or KeyNotFoundException or InvalidOperationException or TaskCanceledException) { return null; }
        }, cancellation);
        var result = (await models, await zdr);
        if (result.Item1.Length > 0) cached = (DateTime.UtcNow.AddMinutes(10), result.Item1, result.Item2);
        return result;
    }
}
