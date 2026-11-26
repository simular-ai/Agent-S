using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AgentS.Core;

[JsonConverter(typeof(JsonStringEnumConverter<AgentMemoryKind>))]
public enum AgentMemoryKind
{
    [JsonStringEnumMemberName("activeWork")] ActiveWork,
    [JsonStringEnumMemberName("preference")] Preference,
    [JsonStringEnumMemberName("decision")] Decision,
    [JsonStringEnumMemberName("workflow")] Workflow,
}

[JsonConverter(typeof(JsonStringEnumConverter<AgentMemoryStatus>))]
public enum AgentMemoryStatus
{
    [JsonStringEnumMemberName("open")] Open,
    [JsonStringEnumMemberName("resolved")] Resolved,
}

/// <summary>Whether Ollama answers on this PC or from Ollama's hosted models. Unconfirmed routes get no automatic recall.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<OllamaInference>))]
public enum OllamaInference
{
    [JsonStringEnumMemberName("local")] Local,
    [JsonStringEnumMemberName("hosted")] Hosted,
    [JsonStringEnumMemberName("unconfirmed")] Unconfirmed,
}

public static class AgentMemoryKinds
{
    public static readonly AgentMemoryKind[] All = [AgentMemoryKind.ActiveWork, AgentMemoryKind.Preference, AgentMemoryKind.Decision, AgentMemoryKind.Workflow];

    public static string DisplayName(this AgentMemoryKind kind) => kind switch
    {
        AgentMemoryKind.ActiveWork => "Active work",
        AgentMemoryKind.Preference => "Preferences",
        AgentMemoryKind.Decision => "Decisions",
        _ => "Workflows",
    };

    /// <summary>The spelling the model reads and writes.</summary>
    public static string RawValue(this AgentMemoryKind kind) => kind switch
    {
        AgentMemoryKind.ActiveWork => "activeWork",
        AgentMemoryKind.Preference => "preference",
        AgentMemoryKind.Decision => "decision",
        _ => "workflow",
    };

    public static AgentMemoryKind? Parse(string? raw) => All.Cast<AgentMemoryKind?>().FirstOrDefault(kind => kind!.Value.RawValue() == raw);
}

/// <summary>
/// The computer and inference route that may receive a note: the Windows counterpart of macOS's route, built from
/// a <see cref="Conversation"/>'s target, harness, provider, computer and model.
/// </summary>
public sealed record AgentMemoryRoute
{
    public ConversationTarget Target { get; }
    /// <summary>Null for Sai; Claude Code for This PC and Box; the chosen harness for Private.</summary>
    public PrivateHarness? Harness { get; }
    /// <summary>Null for Sai, which picks its own model provider.</summary>
    public ModelProvider? Provider { get; }
    /// <summary>The Sai computer's id or the Box's name. Null elsewhere, and for the account's default box.</summary>
    public string? Computer { get; }
    public string? Model { get; }
    public OllamaInference? OllamaInference { get; }

    [JsonConstructor]
    public AgentMemoryRoute(ConversationTarget target, PrivateHarness? harness, ModelProvider? provider, string? computer = null,
        string? model = null, OllamaInference? ollamaInference = null)
    {
        Target = target;
        Harness = target switch
        {
            ConversationTarget.Sai => null,
            ConversationTarget.Private => harness ?? PrivateHarness.AgentS,
            _ => PrivateHarness.ClaudeCode,
        };
        Provider = target == ConversationTarget.Sai ? null : provider;
        Computer = target is ConversationTarget.Sai or ConversationTarget.Box ? Nonempty(computer) : null;
        Model = Nonempty(model);
        OllamaInference = Provider == ModelProvider.Ollama ? ollamaInference ?? Core.OllamaInference.Unconfirmed : null;
    }

    /// <summary>The route a conversation's next turn takes. Only the caller knows whether its Ollama model runs locally.</summary>
    public static AgentMemoryRoute For(Conversation conversation, OllamaInference? ollamaInference = null) => new(
        conversation.Target, conversation.Harness, conversation.Provider,
        conversation.Target == ConversationTarget.Box ? conversation.BoxName : conversation.ComputerId,
        conversation.ModelId, ollamaInference);

    /// <summary>Stable across app launches and distinct for each computer, provider, and model.</summary>
    [JsonIgnore]
    public string ContinuationKey => string.Join('/', new[]
    {
        Target.ToString(), Harness?.ToString(), Provider?.ToString(), Computer, Model, OllamaInference?.ToString(),
    }.Select(part => part is null ? "-" : $"{Encoding.UTF8.GetByteCount(part)}:{part}"));

    [JsonIgnore]
    public bool AllowsMemory => Target switch
    {
        ConversationTarget.Sai => Computer is not null,
        ConversationTarget.Private => Provider is not null && (Harness != PrivateHarness.AgentS || Provider == ModelProvider.Ollama),
        _ => Provider is not null,
    };

    [JsonIgnore]
    public bool AllowsAutomaticRecall => AllowsMemory &&
        (Provider != ModelProvider.Ollama || OllamaInference is Core.OllamaInference.Local or Core.OllamaInference.Hosted);

    [JsonIgnore]
    public string DisplayName
    {
        get
        {
            var location = Target switch
            {
                ConversationTarget.Box => Computer is null ? "Simular Box" : $"Simular Box · {Computer}",
                ConversationTarget.Sai => "Sai computer",
                ConversationTarget.ThisPc => "This PC",
                _ => "Private computer",
            };
            var inference = OllamaInference switch
            {
                Core.OllamaInference.Local => "Local Ollama",
                Core.OllamaInference.Hosted => "Hosted Ollama",
                Core.OllamaInference.Unconfirmed => "Ollama",
                _ => Provider?.DisplayName(),
            };
            return string.Join(" · ", new[] { location, Harness?.DisplayName(), inference, Model }.Where(part => part is not null));
        }
    }

    // The host uses exact identifiers for selected computers and model requests.
    private static string? Nonempty(string? value) => string.IsNullOrWhiteSpace(value) ? null : value;
}

/// <summary>Who may receive a note: nobody (<see cref="Private"/>) or one exact route.</summary>
public sealed record AgentMemoryScope
{
    public static readonly AgentMemoryScope Private = new((AgentMemoryRoute?)null);
    public AgentMemoryRoute? Route { get; }
    [JsonIgnore] public bool IsPrivate => Route is null;
    [JsonConstructor] private AgentMemoryScope(AgentMemoryRoute? route) => Route = route;
    public static AgentMemoryScope Of(AgentMemoryRoute route) => new(route);
    [JsonIgnore] public string DisplayName => Route?.DisplayName ?? "Private";
}

public sealed record AgentMemoryNote
{
    [JsonInclude] public Guid Id { get; internal init; }
    [JsonInclude] public AgentMemoryKind Kind { get; internal init; }
    [JsonInclude] public string Text { get; internal init; } = "";
    [JsonInclude] public AgentMemoryStatus Status { get; internal init; }
    [JsonInclude] public string? SourceChatItemId { get; internal init; }
    [JsonInclude] public DateTimeOffset UpdatedAt { get; internal init; }
    [JsonInclude] public bool IsPinned { get; internal init; }
    [JsonInclude] public AgentMemoryScope Scope { get; internal init; } = AgentMemoryScope.Private;
}

/// <summary>A change the model proposes in its trailer.</summary>
public abstract record AgentMemoryOperation
{
    private AgentMemoryOperation() { }
    public sealed record Add(AgentMemoryKind Kind, string Text) : AgentMemoryOperation;
    public sealed record Update(Guid Id, AgentMemoryKind? Kind, string Text) : AgentMemoryOperation;
    public sealed record Resolve(Guid Id) : AgentMemoryOperation;
    public sealed record Supersede(Guid Id, AgentMemoryKind Kind, string Text) : AgentMemoryOperation;
}

public enum AgentMemoryError { StaleRevision, TooManyOperations, InvalidNote, DuplicateNote, InvalidSource, NoteLimit, UnavailableNote, UnavailableRoute }

public sealed class AgentMemoryException(AgentMemoryError error) : Exception(Describe(error))
{
    public AgentMemoryError Error { get; } = error;

    private static string Describe(AgentMemoryError error) => error switch
    {
        AgentMemoryError.StaleRevision => "Memory changed during the turn.",
        AgentMemoryError.TooManyOperations => "Too many memory changes in one turn.",
        AgentMemoryError.InvalidNote => $"Enter a memory note of up to {AgentMemory.MaxNoteBytes.ToString("N0", CultureInfo.CurrentCulture)} bytes.",
        AgentMemoryError.DuplicateNote => "A matching memory note already exists.",
        AgentMemoryError.InvalidSource => "The memory note has an invalid source.",
        AgentMemoryError.NoteLimit => $"Memory holds up to {AgentMemory.MaxNoteCount.ToString("N0", CultureInfo.CurrentCulture)} notes.",
        AgentMemoryError.UnavailableNote => "The memory note is unavailable for this change.",
        _ => "Memory sharing requires a known model and computer route.",
    };
}

/// <summary>A separate document for one agent. A turn may commit proposed changes only against its input revision.</summary>
public sealed class AgentMemory
{
    public const int CurrentVersion = 1;
    public const int MaxNoteCount = 256;
    public const int MaxNoteBytes = 1_024;
    public const int MaxOperationCount = 8;
    public const int MaxContextBytes = 4_096;

    private List<AgentMemoryNote> notes;

    public int Version { get; }
    public Guid Revision { get; private set; }
    public IReadOnlyList<AgentMemoryNote> Notes => notes;

    public AgentMemory()
    {
        Version = CurrentVersion;
        Revision = Guid.NewGuid();
        notes = [];
    }

    [JsonConstructor]
    private AgentMemory(int version, Guid revision, IReadOnlyList<AgentMemoryNote>? notes)
    {
        if (version != CurrentVersion) throw new JsonException("Unsupported memory version.");
        if (notes is null || notes.Count > MaxNoteCount || notes.Any(note => note is null) ||
            notes.Select(note => note.Id).Distinct().Count() != notes.Count ||
            !notes.All(note => ValidatedTextOrNull(note.Text) == note.Text && IsValidSource(note.SourceChatItemId) &&
                Enum.IsDefined(note.Kind) && Enum.IsDefined(note.Status) && note.Scope is not null))
            throw new JsonException("Invalid memory notes.");
        Version = version;
        Revision = revision;
        this.notes = [.. notes];
    }

    /// <summary>An independent copy to change before the change is saved.</summary>
    public AgentMemory Copy()
    {
        var copy = new AgentMemory { Revision = Revision };
        copy.notes = [.. notes];
        return copy;
    }

    public IReadOnlyList<AgentMemoryNote> NotesFor(AgentMemoryRoute route)
    {
        if (!route.AllowsMemory) return [];
        var scope = AgentMemoryScope.Of(route);
        var found = notes.Where(note => note.Scope == scope).ToList();
        found.Sort(Compare);
        return found;
    }

    public IReadOnlyList<AgentMemoryNote> NotesIn(AgentMemoryKind kind)
    {
        var found = notes.Where(note => note.Kind == kind).ToList();
        found.Sort(Compare);
        return found;
    }

    /// <summary>Unfinished work the route's agent can pick up.</summary>
    public IReadOnlyList<AgentMemoryNote> OpenWork(AgentMemoryRoute route) =>
        NotesFor(route).Where(note => note.Kind == AgentMemoryKind.ActiveWork && note.Status == AgentMemoryStatus.Open).ToList();

    private static int Compare(AgentMemoryNote first, AgentMemoryNote second)
    {
        if (first.IsPinned != second.IsPinned) return first.IsPinned ? -1 : 1;
        if (first.UpdatedAt != second.UpdatedAt) return second.UpdatedAt.CompareTo(first.UpdatedAt);
        return string.CompareOrdinal(UuidString(first.Id), UuidString(second.Id));
    }

    /// <summary>JSON references for the model, bounded after encoding and scoped to its current route.</summary>
    public string? Context(AgentMemoryRoute route, string message = "")
    {
        if (!route.AllowsAutomaticRecall) return null;
        var terms = SearchTerms(PrefixElements(message, 4_096));
        var ranked = NotesFor(route).Where(note => note.Status == AgentMemoryStatus.Open)
            .Select(note => (Note: note, Score: SearchTerms(note.Text).Count(terms.Contains))).ToList();
        ranked.Sort((first, second) =>
        {
            if (first.Note.IsPinned != second.Note.IsPinned) return first.Note.IsPinned ? -1 : 1;
            if (first.Score != second.Score) return second.Score.CompareTo(first.Score);
            return Compare(first.Note, second.Note);
        });
        var entries = new List<string>();
        var byteCount = 2; // array brackets
        foreach (var (note, _) in ranked)
        {
            // Sorted keys, unescaped slashes: the same bytes as the macOS app.
            var item = $"{{\"id\":{JsonString(UuidString(note.Id))},\"kind\":{JsonString(note.Kind.RawValue())},\"text\":{JsonString(note.Text)}}}";
            var length = Encoding.UTF8.GetByteCount(item) + (entries.Count == 0 ? 0 : 1);
            if (byteCount + length > MaxContextBytes) continue;
            entries.Add(item);
            byteCount += length;
        }
        return entries.Count == 0 ? null : "[" + string.Join(',', entries) + "]";
    }

    /// <summary>Applies a turn's proposal all at once, or throws and changes nothing.</summary>
    public bool Apply(IReadOnlyList<AgentMemoryOperation> operations, Guid expecting, AgentMemoryRoute route,
        string? sourceChatItemId = null, DateTimeOffset? now = null)
    {
        if (expecting != Revision) throw new AgentMemoryException(AgentMemoryError.StaleRevision);
        if (!route.AllowsMemory) throw new AgentMemoryException(AgentMemoryError.UnavailableRoute);
        if (operations.Count > MaxOperationCount) throw new AgentMemoryException(AgentMemoryError.TooManyOperations);
        if (!IsValidSource(sourceChatItemId)) throw new AgentMemoryException(AgentMemoryError.InvalidSource);
        var time = now ?? DateTimeOffset.UtcNow;
        var candidate = new List<AgentMemoryNote>(notes);
        var scope = AgentMemoryScope.Of(route);
        foreach (var operation in operations)
        {
            switch (operation)
            {
                case AgentMemoryOperation.Add add:
                    AddNote(add.Kind, add.Text, sourceChatItemId, scope, time, candidate);
                    break;
                case AgentMemoryOperation.Update update:
                {
                    var index = ScopedIndex(update.Id, candidate, scope);
                    var text = ValidatedText(update.Text);
                    var kind = update.Kind ?? candidate[index].Kind;
                    if (candidate[index].Text == text && candidate[index].Kind == kind) continue;
                    if (candidate.Any(note => note.Id != update.Id && Matches(note, kind, text, scope)))
                        throw new AgentMemoryException(AgentMemoryError.DuplicateNote);
                    candidate[index] = candidate[index] with { Kind = kind, Text = text, SourceChatItemId = sourceChatItemId, UpdatedAt = time };
                    break;
                }
                case AgentMemoryOperation.Resolve resolve:
                {
                    var index = ScopedIndex(resolve.Id, candidate, scope);
                    candidate[index] = Resolved(candidate[index], sourceChatItemId, time);
                    break;
                }
                case AgentMemoryOperation.Supersede supersede:
                {
                    var index = ScopedIndex(supersede.Id, candidate, scope);
                    var text = ValidatedText(supersede.Text);
                    if (candidate[index].Text == text && candidate[index].Kind == supersede.Kind) continue;
                    AddNote(supersede.Kind, text, sourceChatItemId, scope, time, candidate);
                    candidate[index] = Resolved(candidate[index], sourceChatItemId, time);
                    break;
                }
            }
        }
        return Commit(candidate);
    }

    /// <summary>
    /// The person adds (no <paramref name="id"/>) or edits a note. New notes are private unless given a route;
    /// an edit keeps the note's scope unless given a route.
    /// </summary>
    public bool SaveUserNote(Guid? id, AgentMemoryKind kind, string text, AgentMemoryRoute? route = null, DateTimeOffset? now = null)
    {
        var validated = ValidatedText(text);
        if (route is not null && !route.AllowsMemory) throw new AgentMemoryException(AgentMemoryError.UnavailableRoute);
        var time = now ?? DateTimeOffset.UtcNow;
        var candidate = new List<AgentMemoryNote>(notes);
        if (id is { } existing)
        {
            var index = candidate.FindIndex(note => note.Id == existing);
            if (index < 0) throw new AgentMemoryException(AgentMemoryError.UnavailableNote);
            var scope = route is null ? candidate[index].Scope : AgentMemoryScope.Of(route);
            var changedText = candidate[index].Text != validated || candidate[index].Kind != kind;
            if (!changedText && candidate[index].Scope == scope) return false;
            candidate[index] = candidate[index] with
            {
                SourceChatItemId = changedText ? null : candidate[index].SourceChatItemId,
                Kind = kind, Text = validated, Scope = scope, UpdatedAt = time,
            };
        }
        else AddNote(kind, validated, null, route is null ? AgentMemoryScope.Private : AgentMemoryScope.Of(route), time, candidate);
        return Commit(candidate);
    }

    public bool SetScope(AgentMemoryScope scope, Guid id, DateTimeOffset? now = null)
    {
        if (scope.Route is { AllowsMemory: false }) throw new AgentMemoryException(AgentMemoryError.UnavailableRoute);
        var index = notes.FindIndex(note => note.Id == id);
        if (index < 0) throw new AgentMemoryException(AgentMemoryError.UnavailableNote);
        if (notes[index].Scope == scope) return false;
        notes[index] = notes[index] with { Scope = scope, UpdatedAt = now ?? DateTimeOffset.UtcNow };
        Revision = Guid.NewGuid();
        return true;
    }

    public bool SetPinned(bool isPinned, Guid id, DateTimeOffset? now = null)
    {
        var index = notes.FindIndex(note => note.Id == id);
        if (index < 0 || notes[index].IsPinned == isPinned) return false;
        notes[index] = notes[index] with { IsPinned = isPinned, UpdatedAt = now ?? DateTimeOffset.UtcNow };
        Revision = Guid.NewGuid();
        return true;
    }

    public bool SetStatus(AgentMemoryStatus status, Guid id, DateTimeOffset? now = null)
    {
        var index = notes.FindIndex(note => note.Id == id);
        if (index < 0 || notes[index].Status == status) return false;
        notes[index] = notes[index] with { Status = status, UpdatedAt = now ?? DateTimeOffset.UtcNow };
        Revision = Guid.NewGuid();
        return true;
    }

    public bool Forget(Guid id) => Forget([id]);

    public bool Forget(IReadOnlyCollection<Guid> ids)
    {
        if (ids.Count == 0) return false;
        return Commit(notes.Where(note => !ids.Contains(note.Id)).ToList());
    }

    /// <summary>
    /// Continuation keys to drop after an edit of <paramref name="ids"/>: a resumed session on a note's route may
    /// still hold the note's old text, or a note that is no longer shared with it.
    /// </summary>
    public static IReadOnlySet<string> InvalidatedContinuations(AgentMemory before, AgentMemory after, IEnumerable<Guid> ids)
    {
        var set = ids.ToHashSet();
        bool Unchanged(AgentMemoryNote old) =>
            after.notes.FirstOrDefault(note => note.Id == old.Id) is { } kept && kept.Scope == old.Scope && kept.Text == old.Text;
        return before.notes.Where(old => set.Contains(old.Id) && old.Scope.Route is not null && !Unchanged(old))
            .Select(old => old.Scope.Route!.ContinuationKey).ToHashSet();
    }

    private bool Commit(List<AgentMemoryNote> candidate)
    {
        if (notes.SequenceEqual(candidate)) return false;
        notes = candidate;
        Revision = Guid.NewGuid();
        return true;
    }

    private static int ScopedIndex(Guid id, List<AgentMemoryNote> notes, AgentMemoryScope scope)
    {
        var index = notes.FindIndex(note => note.Id == id && note.Scope == scope && note.Status == AgentMemoryStatus.Open);
        return index >= 0 ? index : throw new AgentMemoryException(AgentMemoryError.UnavailableNote);
    }

    private static void AddNote(AgentMemoryKind kind, string text, string? source, AgentMemoryScope scope, DateTimeOffset now, List<AgentMemoryNote> notes)
    {
        var validated = ValidatedText(text);
        if (notes.Any(note => Matches(note, kind, validated, scope))) return;
        if (notes.Count >= MaxNoteCount) throw new AgentMemoryException(AgentMemoryError.NoteLimit);
        notes.Add(new AgentMemoryNote
        {
            Id = Guid.NewGuid(), Kind = kind, Text = validated, Status = AgentMemoryStatus.Open,
            SourceChatItemId = source, UpdatedAt = now, IsPinned = false, Scope = scope,
        });
    }

    private static bool Matches(AgentMemoryNote note, AgentMemoryKind kind, string text, AgentMemoryScope scope) =>
        note.Status == AgentMemoryStatus.Open && note.Kind == kind && note.Text == text && note.Scope == scope;

    private static AgentMemoryNote Resolved(AgentMemoryNote note, string? source, DateTimeOffset now) =>
        note with { Status = AgentMemoryStatus.Resolved, UpdatedAt = now, SourceChatItemId = source };

    private static bool IsValidSource(string? id) => id is null || (!string.IsNullOrWhiteSpace(id) && Encoding.UTF8.GetByteCount(id) <= 256);

    private static string ValidatedText(string text) => ValidatedTextOrNull(text) ?? throw new AgentMemoryException(AgentMemoryError.InvalidNote);

    internal static string? ValidatedTextOrNull(string? text)
    {
        var trimmed = text?.Trim();
        return string.IsNullOrEmpty(trimmed) || Encoding.UTF8.GetByteCount(trimmed) > MaxNoteBytes ? null : trimmed;
    }

    /// <summary>The id as the model sees it, matching macOS's uppercase UUID strings.</summary>
    internal static string UuidString(Guid id) => id.ToString("D").ToUpperInvariant();

    /// <summary>Lowercased words of 3 or more characters, split at anything that isn't a letter or number.</summary>
    private static HashSet<string> SearchTerms(string text)
    {
        var terms = new HashSet<string>(StringComparer.Ordinal);
        var word = new StringBuilder();
        var count = 0;
        void Flush()
        {
            if (count >= 3) terms.Add(word.ToString().Normalize(NormalizationForm.FormC));
            word.Clear(); count = 0;
        }
        var elements = StringInfo.GetTextElementEnumerator(text.ToLowerInvariant());
        while (elements.MoveNext())
        {
            var element = elements.GetTextElement();
            var rune = Rune.GetRuneAt(element, 0);
            if (Rune.IsLetter(rune) || Rune.IsNumber(rune)) { word.Append(element); count++; }
            else Flush();
        }
        Flush();
        return terms;
    }

    private static string PrefixElements(string text, int count)
    {
        var info = new StringInfo(text);
        return info.LengthInTextElements <= count ? text : info.SubstringByTextElements(0, count);
    }

    private static string JsonString(string value)
    {
        var builder = new StringBuilder(value.Length + 2).Append('"');
        foreach (var c in value)
        {
            switch (c)
            {
                case '"': builder.Append("\\\""); break;
                case '\\': builder.Append("\\\\"); break;
                case '\n': builder.Append("\\n"); break;
                case '\r': builder.Append("\\r"); break;
                case '\t': builder.Append("\\t"); break;
                case '\b': builder.Append("\\b"); break;
                case '\f': builder.Append("\\f"); break;
                case < ' ': builder.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture)); break;
                default: builder.Append(c); break;
            }
        }
        return builder.Append('"').ToString();
    }
}
