using System.Text.Json;

namespace AgentS.Core;

/// <summary>One agent's saved notes and its resumable sessions, keyed by <see cref="AgentMemoryRoute.ContinuationKey"/>.</summary>
public sealed class AgentMemoryDocument
{
    public AgentMemory Notes { get; init; } = new();
    public Dictionary<string, string> Sessions { get; init; } = [];

    public string? SessionFor(AgentMemoryRoute route) => Sessions.GetValueOrDefault(route.ContinuationKey);

    /// <summary>A copy that resumes <paramref name="sessionId"/> on the route next time.</summary>
    public AgentMemoryDocument WithSession(AgentMemoryRoute route, string sessionId) =>
        new() { Notes = Notes, Sessions = new(Sessions) { [route.ContinuationKey] = sessionId } };

    /// <summary>A copy with the same notes and no sessions, for a fresh conversation.</summary>
    public AgentMemoryDocument WithoutSessions() => new() { Notes = Notes };

    /// <summary>
    /// Runs <paramref name="change"/> on a copy of the notes. Returns null when nothing changed; otherwise the
    /// document to save, without the sessions that may hold an old copy of an edited note in <paramref name="ids"/>.
    /// </summary>
    public AgentMemoryDocument? Edited(Func<AgentMemory, bool> change, IEnumerable<Guid>? ids = null)
    {
        var updated = Notes.Copy();
        if (!change(updated)) return null;
        var sessions = new Dictionary<string, string>(Sessions);
        foreach (var key in AgentMemory.InvalidatedContinuations(Notes, updated, ids ?? [])) sessions.Remove(key);
        return new() { Notes = updated, Sessions = sessions };
    }
}

/// <summary>The app owns the memory file on the PC, outside anything shared with an agent's computer: one JSON file per agent.</summary>
public sealed class AgentMemoryStore
{
    public const int MaxFileSize = 2 * 1024 * 1024;

    internal static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public string FilePath { get; }

    /// <param name="folder">The folder holding every agent's memory file.</param>
    /// <param name="agentId">The agent's <see cref="Conversation.Id"/>, used as the file name.</param>
    public AgentMemoryStore(string folder, string agentId)
    {
        if (string.IsNullOrEmpty(agentId) || agentId.Length > 128 || !agentId.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_'))
            throw new ArgumentException("The agent id isn't a safe file name.", nameof(agentId));
        FilePath = Path.Combine(folder, agentId + ".json");
    }

    /// <summary>The saved document, or an empty one when none was saved. Throws on an unreadable, invalid or oversized file.</summary>
    public AgentMemoryDocument Load()
    {
        if (!File.Exists(FilePath)) return new();
        byte[] data;
        using (var reader = new FileStream(FilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            if (reader.Length > MaxFileSize) throw new IOException("The memory file is too large.");
            data = new byte[MaxFileSize + 1];
            var count = 0;
            int read;
            while (count < data.Length && (read = reader.Read(data, count, data.Length - count)) > 0) count += read;
            if (count > MaxFileSize) throw new IOException("The memory file is too large.");
            Array.Resize(ref data, count);
        }
        var document = JsonSerializer.Deserialize<AgentMemoryDocument>(data, Json);
        if (document?.Notes is null || document.Sessions is null || document.Sessions.Values.Any(value => value is null))
            throw new JsonException("Invalid memory file.");
        return document;
    }

    /// <summary>Replaces the file atomically.</summary>
    public void Save(AgentMemoryDocument document)
    {
        var data = JsonSerializer.SerializeToUtf8Bytes(document, Json);
        if (data.Length > MaxFileSize) throw new IOException("The memory file would be too large.");
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        var temporary = FilePath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllBytes(temporary, data);
            File.Move(temporary, FilePath, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    public void Delete()
    {
        if (File.Exists(FilePath)) File.Delete(FilePath);
    }
}
