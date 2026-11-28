using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AgentS.Core;

public sealed record OpenRouterKeyUsage(
    [property: JsonPropertyName("usage_daily"), JsonRequired] decimal Daily,
    [property: JsonPropertyName("usage_weekly"), JsonRequired] decimal Weekly,
    [property: JsonPropertyName("usage_monthly"), JsonRequired] decimal Monthly,
    [property: JsonPropertyName("usage"), JsonRequired] decimal Total)
{
    private sealed record Envelope([property: JsonPropertyName("data")] OpenRouterKeyUsage? Data);
    private static readonly HttpClient Http = new(new SocketsHttpHandler { AllowAutoRedirect = false })
    {
        Timeout = TimeSpan.FromSeconds(15),
    };

    public static string FormatSpend(decimal amount) => amount.ToString("$0.00######", CultureInfo.InvariantCulture);

    public static async Task<OpenRouterKeyUsage> Read(string key)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, OpenRouterProxy.Api.AbsoluteUri + "/v1/key");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        request.Headers.CacheControl = new CacheControlHeaderValue { NoCache = true, NoStore = true };
        using var response = await Http.SendAsync(request);
        response.EnsureSuccessStatusCode();
        using var body = await response.Content.ReadAsStreamAsync();
        return (await JsonSerializer.DeserializeAsync<Envelope>(body))?.Data
            ?? throw new JsonException("OpenRouter returned no key usage.");
    }
}
