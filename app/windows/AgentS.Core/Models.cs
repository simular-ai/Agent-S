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
public enum ConversationTarget { Sai, Private, ThisPc, Box }

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
    /// <summary>How the model is reached: Ollama on this PC, or OpenRouter through the app's proxy.</summary>
    public ModelProvider Provider { get; set; } = ModelProvider.Ollama;
    /// <summary>Simular Box agents: the box to work on, by name; null is the account's default box.</summary>
    public string? BoxName { get; set; }
    /// <summary>The avatar color the person chose (Black, Blue, Green, Orange, Purple, Red, Teal), or null for the automatic one.</summary>
    public string? Color { get; set; }
    public List<ChatEntry> Entries { get; set; } = [];
    public override string ToString() => Name;

    /// <summary>"Claude Code · gemma4", "Sai · auto": the sidebar preview before there's a conversation, as on macOS.</summary>
    public string Summary() => Target switch
    {
        ConversationTarget.Sai => $"Sai · {(ModelId is null or "auto" ? "auto" : ModelId)}",
        ConversationTarget.ThisPc or ConversationTarget.Box => $"Claude Code · {ModelId ?? OllamaModels.Default}",
        _ => $"{(Harness is { } h && h.RunsInComputer() ? h.DisplayName() : "Agent S")} · {ModelId ?? OllamaModels.Default}",
    };
}

public sealed class ChatEntry
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");
    public string Role { get; set; } = "Sai";
    public string Text { get; set; } = "";
    /// <summary>Tool steps ("Activity"): the tool's name, the input shown in its details, its output, and how it ended.</summary>
    public string? Tool { get; set; }
    public string? Input { get; set; }
    public string? Output { get; set; }
    public bool Done { get; set; }
    public bool Failed { get; set; }
    // Approval URLs and controls are deliberately not persisted or replayed after restarting.
    [JsonIgnore] public string? ApprovalId { get; set; }
    [JsonIgnore] public Uri? ApprovalUrl { get; set; }
    [JsonIgnore] public bool LinkOnly { get; set; }
    [JsonIgnore] public bool Answered { get; set; }
    /// <summary>A This PC tool use waiting for the person: called with true to allow, false to deny.</summary>
    [JsonIgnore] public Action<bool>? LocalAnswer { get; set; }
}
