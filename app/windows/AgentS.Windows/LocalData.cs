using System.Text.Json;
using AgentS.Core;
using global::Windows.Security.Credentials;

namespace AgentS.Windows;

internal static class Secrets
{
    private const string Resource = "ai.simular.agent-s.windows.sai";
    private const string OpenRouterResource = "ai.simular.agent-s.windows.openrouter";
    public static string? Read() => Read(Resource);
    public static void Save(string key) => new PasswordVault().Add(new PasswordCredential(Resource, "api-key", key));
    public static void Delete() => Delete(Resource);

    /// <summary>The OpenRouter key; only the app's loopback proxy uses it, agents never see it.</summary>
    public static string? ReadOpenRouter() => Read(OpenRouterResource);
    public static void SaveOpenRouter(string key) => new PasswordVault().Add(new PasswordCredential(OpenRouterResource, "api-key", key));
    public static void DeleteOpenRouter() => Delete(OpenRouterResource);

    /// <summary>The Simular Box key; given to Claude Code only through its environment for Box's MCP server.</summary>
    private const string BoxResource = "ai.simular.agent-s.windows.box";
    public static string? ReadBox() => Read(BoxResource);
    public static void SaveBox(string key) => new PasswordVault().Add(new PasswordCredential(BoxResource, "api-key", key));
    public static void DeleteBox() => Delete(BoxResource);

    private static string? Read(string resource)
    {
        try
        {
            var credential = new PasswordVault().Retrieve(resource, "api-key");
            credential.RetrievePassword();
            return credential.Password;
        }
        catch (System.Runtime.InteropServices.COMException error) when ((uint)error.HResult == 0x80070490)
        { return null; }
    }
    private static void Delete(string resource)
    {
        var key = Read(resource);
        if (key is not null) new PasswordVault().Remove(new PasswordCredential(resource, "api-key", key));
    }
}

internal sealed class HistoryStore
{
    /// <summary>Conversations and settings. AGENT_S_DATA_DIR points them elsewhere (fresh-install previews and smoke tests); the Private computer, CLI profiles and keys are unaffected.</summary>
    public static string DataFolder => Environment.GetEnvironmentVariable("AGENT_S_DATA_DIR") is { Length: > 0 } custom
        ? custom : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Simular", "Agent S");
    private readonly string path = Path.Combine(DataFolder, "conversations.json");
    private readonly SemaphoreSlim gate = new(1, 1);
    public async Task<List<Conversation>> Read()
    {
        if (!File.Exists(path)) return [];
        // Refuse a corrupt/unreasonably large file without silently overwriting it.
        if (new FileInfo(path).Length > 20_000_000) throw new IOException("Conversation history is too large to open.");
        return JsonSerializer.Deserialize<List<Conversation>>(await File.ReadAllTextAsync(path)) ?? [];
    }
    public async Task Save(IEnumerable<Conversation> conversations)
    {
        var json = JsonSerializer.Serialize(conversations);
        if (System.Text.Encoding.UTF8.GetByteCount(json) > 20_000_000) throw new IOException("Conversation history exceeds the storage limit.");
        await gate.WaitAsync();
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await File.WriteAllTextAsync(path + ".tmp", json);
            File.Move(path + ".tmp", path, true);
        }
        finally { gate.Release(); }
    }
}
