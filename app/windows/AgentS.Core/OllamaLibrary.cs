using System.Globalization;
using System.Net;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AgentS.Core;

/// <summary>What a model accepts, as the macOS model browser shows it.</summary>
public enum ModelInput { TextOnly, Multimodal }

/// <summary>
/// Ollama model names, defaults and requirements, matching the macOS app: Nemotron 3 Nano 4B is the
/// default; Nemotron 3.5 Bolt E4B and Gemma 4 are the other suggestions.
/// </summary>
public static class OllamaModels
{
    public const string NemotronNano4B = "nemotron-3-nano:4b";
    public const string NemotronBoltE4B = "nemotron-3.5-bolt:e4b";
    public const string Gemma4 = "gemma4";
    public const string Default = NemotronNano4B;
    /// <summary>Ollama 0.14 serves the Anthropic Messages API Claude Code speaks.</summary>
    public const string MinimumVersion = "0.14.0";
    public static readonly string[] Suggested = [NemotronNano4B, NemotronBoltE4B, Gemma4];

    public static string NormalizedName(string model) => model.EndsWith(":latest", StringComparison.Ordinal) ? model[..^7] : model;
    public static bool Same(string a, string b) => NormalizedName(a) == NormalizedName(b);

    public static string DisplayName(string name) => NormalizedName(name) switch
    {
        Gemma4 => "Gemma 4",
        NemotronBoltE4B => "Nemotron 3.5 Bolt E4B",
        NemotronNano4B => "Nemotron 3 Nano 4B",
        var other => other,
    };

    public static ModelInput? SuggestedInput(string name) => NormalizedName(name) switch
    {
        NemotronNano4B => ModelInput.TextOnly,
        NemotronBoltE4B or Gemma4 => ModelInput.Multimodal,
        _ => null,
    };

    public static string InputLabel(ModelInput? input) => input switch
    {
        ModelInput.TextOnly => "Text only",
        ModelInput.Multimodal => "Multimodal",
        _ => "",
    };

    /// <summary>The download size the macOS app shows before the catalog answers.</summary>
    public static string? PresetSize(string name) => NormalizedName(name) switch
    {
        NemotronNano4B => "2.8 GB – 2.9 GB",
        NemotronBoltE4B => "7.2 GB – 7.9 GB",
        _ => null,
    };

    public static string MinimumVersionFor(string name) => NormalizedName(name) == NemotronBoltE4B ? "0.40.10" : MinimumVersion;

    /// <summary>"gemma4:cloud", "x:31b-cloud": names Ollama runs on its servers, not this PC.</summary>
    public static bool IsCloudReference(string model)
    {
        var colon = model.LastIndexOf(':');
        if (colon < 0) return false;
        var tag = model[(colon + 1)..].Trim().ToLowerInvariant();
        return !tag.Contains('/') && (tag == "cloud" || tag.EndsWith("-cloud", StringComparison.Ordinal));
    }

    /// <summary>Dotted version comparison; "0.14.1-rc0" counts as 0.14.1.</summary>
    public static bool IsAtLeast(string version, string minimum)
    {
        static int Part(string s) => int.TryParse(new string(s.TakeWhile(char.IsAsciiDigit).ToArray()), out var n) ? n : 0;
        var a = version.Split('.').Select(Part).ToArray();
        var b = minimum.Split('.').Select(Part).ToArray();
        for (var i = 0; i < Math.Max(a.Length, b.Length); i++)
        {
            int x = i < a.Length ? a[i] : 0, y = i < b.Length ? b[i] : 0;
            if (x != y) return x > y;
        }
        return true;
    }

    /// <summary>
    /// The models in Ollama's /api/tags answer: name, size and whether Ollama runs it remotely.
    /// Leaves out the app's own larger-context variants (agent-s/…:32k, see HarnessCommand.ContextModel):
    /// they share an installed model's weights and are never a choice.
    /// </summary>
    public static InstalledModel[] ParseTags(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            if (!document.RootElement.TryGetProperty("models", out var list) || list.ValueKind != JsonValueKind.Array) return [];
            return list.EnumerateArray().Select(item =>
            {
                string? Str(string key) => item.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
                if (Str("name") is not { } name || name.StartsWith(HarnessCommand.ContextModelPrefix, StringComparison.Ordinal)) return null;
                long size = item.TryGetProperty("size", out var s) && s.TryGetInt64(out var n) ? n : 0;
                bool remote = Str("remote_model") is { Length: > 0 } || Str("remote_host") is { Length: > 0 };
                return new InstalledModel(name, size, remote || IsCloudReference(name));
            }).OfType<InstalledModel>().ToArray();
        }
        catch (JsonException) { return []; }
    }

    /// <summary>"7.2 GB", "980 MB".</summary>
    public static string FormatBytes(long bytes) => bytes >= 1_000_000_000
        ? (bytes / 1e9).ToString("0.0", CultureInfo.InvariantCulture) + " GB"
        : Math.Max(1, bytes / 1_000_000).ToString(CultureInfo.InvariantCulture) + " MB";
}

/// <summary>A model in the local Ollama. Remote ones run on Ollama's servers (hosted inference).</summary>
public sealed record InstalledModel(string Name, long Size, bool Remote);

/// <summary>
/// The macOS app's advisory memory check: warns when available RAM, minus a sandbox that still has to
/// start, can't hold the model plus headroom (the larger of 1 GiB and a quarter of the model).
/// </summary>
public sealed class OllamaMemoryEstimate
{
    private const ulong GiB = 1UL << 30;
    private static readonly Regex Size = new(@"^([0-9]+(?:\.[0-9]+)?)\s*([kKMGTP]?i?B)(?:\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*([kKMGTP]?i?B))?$");
    private static readonly Dictionary<string, double> Units = new()
    {
        ["B"] = 1, ["KB"] = 1e3, ["MB"] = 1e6, ["GB"] = 1e9, ["TB"] = 1e12, ["PB"] = 1e15,
        ["KIB"] = 1 << 10, ["MIB"] = 1 << 20, ["GIB"] = 1 << 30, ["TIB"] = Math.Pow(2, 40), ["PIB"] = Math.Pow(2, 50),
    };

    public ulong ModelBytes { get; }
    private OllamaMemoryEstimate(ulong bytes) => ModelBytes = bytes;

    public static OllamaMemoryEstimate? FromBytes(ulong bytes) => bytes > 0 ? new(bytes) : null;

    /// <summary>"2.8 GB – 2.9 GB" (the larger end counts), "980 MiB".</summary>
    public static OllamaMemoryEstimate? FromCatalogSize(string size)
    {
        var match = Size.Match(size.Trim());
        if (!match.Success || Bytes(match.Groups[1].Value, match.Groups[2].Value) is not { } first) return null;
        if (!match.Groups[3].Success) return FromBytes(first);
        return Bytes(match.Groups[3].Value, match.Groups[4].Value) is { } last ? FromBytes(Math.Max(first, last)) : null;
    }

    /// <param name="sandboxMemory">Only the memory a sandbox that is still off will take when it starts.</param>
    public bool ExceedsBudget(ulong availableMemory, ulong sandboxMemory)
    {
        if (availableMemory < sandboxMemory) return true;
        var budget = availableMemory - sandboxMemory;
        if (budget < ModelBytes) return true;
        var quarter = ModelBytes / 4 + (ModelBytes % 4 == 0 ? 0UL : 1UL);
        return budget - ModelBytes < Math.Max(GiB, quarter);
    }

    /// <summary>"6 GB", "9.6 GB".</summary>
    public static string FormatGiB(ulong bytes)
    {
        var gb = bytes / (double)GiB;
        return gb >= 10 || Math.Round(gb) == gb ? $"{Math.Round(gb):0} GB" : gb.ToString("0.0", CultureInfo.InvariantCulture) + " GB";
    }

    public static string Warning(ulong availableMemory) =>
        $"This model may need more than the {FormatGiB(availableMemory)} of RAM available. Try a smaller local model or a cloud model.";

    private static ulong? Bytes(string amount, string unit)
    {
        if (!double.TryParse(amount, NumberStyles.Float, CultureInfo.InvariantCulture, out var n) || !Units.TryGetValue(unit.ToUpperInvariant(), out var m)) return null;
        var bytes = Math.Ceiling(n * m);
        // 2^64 itself doesn't fit; doubles can't say exactly where below it, so stay clear of the edge.
        return bytes < 18_446_744_073_709_549_568d ? (ulong)bytes : null;
    }
}

/// <summary>
/// Download progress across a pull's layers, as on macOS: Ollama announces each layer separately and
/// omits <c>completed</c> while it is zero, so the fraction is completed bytes over all announced bytes.
/// </summary>
public sealed class OllamaPullProgress
{
    private readonly Dictionary<string, (double Total, double Completed)> layers = [];

    /// <returns>0…1, or null before any layer's size is known.</returns>
    public double? Update(string status, string? digest, double? total, double? completed)
    {
        if (status is "verifying sha256 digest" or "writing manifest" or "removing any unused layers" or "success") return 1;
        if (total is { } t && double.IsFinite(t) && t > 0)
        {
            var done = completed ?? 0;
            if (double.IsFinite(done)) layers[digest ?? ""] = (t, Math.Clamp(done, 0, t));
        }
        var all = layers.Values.Sum(l => l.Total);
        return all > 0 ? layers.Values.Sum(l => l.Completed) / all : null;
    }
}

/// <summary>
/// Ollama's public model library (ollama.com/library), read like the macOS app reads it: the family list,
/// then each family's tags page for local variants with download sizes and inputs. Hosted (cloud) variants
/// are left out. Redirects aren't followed and results are cached for 10 minutes.
/// </summary>
public sealed class OllamaCatalog(HttpMessageHandler? handler = null)
{
    public static readonly Uri LibraryUrl = new("https://ollama.com/library");
    public sealed record Variant(string Name, string? Size, ModelInput? Input);

    private static readonly Regex Identifier = new("^[A-Za-z0-9][A-Za-z0-9._-]*$");
    private readonly HttpClient http = new(handler ?? new SocketsHttpHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromSeconds(30) };
    private (DateTime Until, string[] Value)? families;
    private readonly Dictionary<string, (DateTime Until, Variant[] Value)> tags = [];

    public async Task<string[]> Families(CancellationToken cancellation = default)
    {
        if (families is { } cached && cached.Until > DateTime.UtcNow) return cached.Value;
        var value = ParseFamilies(await Html(LibraryUrl, cancellation));
        families = (DateTime.UtcNow.AddMinutes(10), value);
        return value;
    }

    public async Task<Variant[]> Variants(string family, CancellationToken cancellation = default)
    {
        var url = TagsUrl(family);
        lock (tags) { if (tags.TryGetValue(family, out var cached) && cached.Until > DateTime.UtcNow) return cached.Value; }
        var value = ParseVariants(await Html(url, cancellation), family);
        lock (tags) tags[family] = (DateTime.UtcNow.AddMinutes(10), value);
        return value;
    }

    /// <summary>
    /// The whole library: every family's local variants with sizes, reported family by family as the pages
    /// arrive (a few at a time, in library order). A family whose page can't be read is reported empty.
    /// </summary>
    public async Task LoadAll(Action<string, Variant[]?> family, int concurrency = 6, CancellationToken cancellation = default)
    {
        var names = await Families(cancellation);
        using var gate = new SemaphoreSlim(concurrency);
        await Task.WhenAll(names.Select(async name =>
        {
            await gate.WaitAsync(cancellation);
            try
            {
                Variant[]? variants;
                try { variants = await Variants(name, cancellation); }
                catch (Exception error) when (error is LocalAgentException or HttpRequestException or TaskCanceledException && !cancellation.IsCancellationRequested) { variants = null; }
                family(name, variants);
            }
            finally { gate.Release(); }
        }));
    }

    private async Task<string> Html(Uri url, CancellationToken cancellation)
    {
        using var response = await http.GetAsync(url, cancellation);
        if (response.StatusCode != HttpStatusCode.OK) throw new LocalAgentException("The Ollama model library is unavailable.");
        if (response.Content.Headers.ContentType?.MediaType != "text/html") throw new LocalAgentException("Couldn't read the Ollama model library.");
        return await response.Content.ReadAsStringAsync(cancellation);
    }

    public static Uri TagsUrl(string family) => Identifier.IsMatch(family)
        ? new Uri($"{LibraryUrl.AbsoluteUri}/{family}/tags")
        : throw new LocalAgentException("That isn't a valid Ollama library model name.");

    public static string[] ParseFamilies(string html)
    {
        // The library is one result list; links elsewhere on the page (navigation) don't count.
        var listing = Regex.Match(html, @"<div\b[^>]*\sid=[""']repo[""'][^>]*>\s*<ul\b[^>]*>([\s\S]*?)</ul>");
        if (!listing.Success) throw new LocalAgentException("Couldn't read the Ollama model library.");
        var names = Regex.Matches(listing.Groups[1].Value, @"<a\b[^>]*\shref=[""']/library/([A-Za-z0-9][A-Za-z0-9._-]*)[""']")
            .Select(m => m.Groups[1].Value).Distinct().ToArray();
        return names.Length > 0 ? names : throw new LocalAgentException("Couldn't read the Ollama model library.");
    }

    public static Variant[] ParseVariants(string html, string family)
    {
        var count = Regex.Match(html, @"<p\b[^>]*>\s*(\d+)\s+models?\s*</p>");
        if (!count.Success || !int.TryParse(count.Groups[1].Value, out var expected) || expected <= 0)
            throw new LocalAgentException("Couldn't read the Ollama model library.");
        var libraryPath = "/library/" + family;
        var seen = new HashSet<string>();
        var variants = new List<Variant>();
        // Each model has mobile and desktop markup; the mobile link (md:hidden) has the name, size and inputs once.
        foreach (Match link in Regex.Matches(html, @"<a\b([^>]*)>([\s\S]*?)</a\s*>"))
        {
            var attributes = link.Groups[1].Value;
            var classes = Regex.Match(attributes, @"\sclass=[""']([^""']+)[""']");
            var href = Regex.Match(attributes, @"\shref=[""']([^""']+)[""']");
            if (!classes.Success || !href.Success || !classes.Groups[1].Value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Contains("md:hidden")) continue;
            var path = href.Groups[1].Value;
            string name, tag;
            if (path == libraryPath) { name = family; tag = ""; }
            else
            {
                if (!path.StartsWith(libraryPath + ":", StringComparison.Ordinal)) continue;
                tag = path[(libraryPath.Length + 1)..];
                if (!Identifier.IsMatch(tag)) continue;
                name = $"{family}:{tag}";
            }
            if (!seen.Add(name)) continue;
            var body = link.Groups[2].Value;
            if (tag.ToLowerInvariant().Split('-').Contains("cloud") || IsHosted(body)) continue;
            variants.Add(new Variant(name, ModelSize(body), Inputs(body)));
        }
        // A page with only hosted entries legitimately has no downloadable variants.
        if (seen.Count != expected) throw new LocalAgentException("Couldn't read the Ollama model library.");
        return [.. variants];
    }

    private static bool IsHosted(string body) =>
        Regex.IsMatch(body, @"•\s*[A-Za-z ]+\s+Usage\s*•")
        || Regex.IsMatch(body, @"\s(?:title|aria-label)=[""'][^""']*[Cc]loud[^""']*[""']")
        || Regex.IsMatch(body, @"<title>\s*[Cc]loud\s*</title>");

    private static string? ModelSize(string body)
    {
        var match = Regex.Match(body.Replace("&nbsp;", " "), @"•\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)(?:\s*[-–]\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B))?\s*•");
        if (!match.Success) return null;
        var first = $"{match.Groups[1].Value} {match.Groups[2].Value}";
        return match.Groups[3].Success ? $"{first} – {match.Groups[3].Value} {match.Groups[4].Value}" : first;
    }

    private static ModelInput? Inputs(string body)
    {
        var match = Regex.Match(body, @"•\s*(?:<span\b[^>]*>\s*)?([A-Za-z]+(?:\s*,\s*[A-Za-z]+)*)\s+input\s*•");
        if (!match.Success) return null;
        return FromModalities(match.Groups[1].Value.Split(','));
    }

    /// <summary>Only "text" is text-only; more than one input kind is multimodal; anything else is unknown.</summary>
    public static ModelInput? FromModalities(IEnumerable<string> modalities)
    {
        var kinds = modalities.Select(k => k.Trim().ToLowerInvariant()).Where(k => k.Length > 0).ToHashSet();
        if (kinds.SetEquals(["text"])) return ModelInput.TextOnly;
        return kinds.Count > 1 ? ModelInput.Multimodal : null;
    }
}
