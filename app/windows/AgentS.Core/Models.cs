using System.Text.Json.Serialization;

namespace AgentS.Core;

public sealed record Computer(
    [property: JsonPropertyName("machineId")] string Id,
    string Name, string Kind, string? Os, bool Online, string? Status, bool CanWake)
{
    public string Availability => Kind == "own" ? (Online ? "Online" : "Offline") :
        Status?.ToLowerInvariant() switch
        {
            "hibernated" or "hibernating" => "Asleep",
            "provisioning" or "wakingup" => "Starting",
            "active" => "Running",
            _ => Online ? "Online" : "Offline"
        };
    public override string ToString() => $"{Name} · {Availability}";
}

public sealed record SaiModel(string Id, string Name, bool Allowed = true)
{
    public override string ToString() => Name;
}

[JsonConverter(typeof(JsonStringEnumConverter<ConversationTarget>))]
public enum ConversationTarget { Sai, Private }

public sealed class Conversation
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");
    public string Name { get; set; } = "New agent";
    public ConversationTarget Target { get; set; } = ConversationTarget.Sai;
    public string ComputerId { get; set; } = "";
    public string? ModelId { get; set; }
    public string? SessionId { get; set; }
    /// <summary>Private agents only: the program that runs turns. Null (older agents) is the built-in loop.</summary>
    public PrivateHarness? Harness { get; set; }
    /// <summary>The harness's own session to resume, per harness, so switching back continues where it was.</summary>
    public Dictionary<PrivateHarness, string> HarnessSessions { get; set; } = [];
    public List<ChatEntry> Entries { get; set; } = [];
    public override string ToString() => Name;
}

public sealed class ChatEntry
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");
    public string Role { get; set; } = "Sai";
    public string Text { get; set; } = "";
    // Approval URLs and controls are deliberately not persisted or replayed after restarting.
    [JsonIgnore] public string? ApprovalId { get; set; }
    [JsonIgnore] public Uri? ApprovalUrl { get; set; }
    [JsonIgnore] public bool LinkOnly { get; set; }
    [JsonIgnore] public bool Answered { get; set; }
}
