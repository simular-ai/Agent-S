using System.Text.Json;

namespace AgentS.Windows;

/// <summary>App preferences, like the macOS app's UserDefaults: one small JSON file beside the conversations.</summary>
internal sealed class AppSettings
{
    private static readonly string FilePath = Path.Combine(
        HistoryStore.DataFolder, "settings.json");

    /// <summary>Ask OpenRouter for Zero Data Retention endpoints only; on by default, for every OpenRouter agent.</summary>
    public bool OpenRouterRequiresZdr { get; set; } = true;
    /// <summary>First-launch setup was finished or skipped ("Open chat" / "Finish later").</summary>
    public bool FinishedFirstAgentSetup { get; set; }
    /// <summary>The computer panel's width; the split is resizable, as on macOS.</summary>
    public double ComputerPanelWidth { get; set; } = 520;
    /// <summary>The Hyper-V Private computer's memory (2, 4 or 6 GB), applied when it next starts. Null: the setup choice.</summary>
    public int? ComputerMemoryGB { get; set; }

    private static AppSettings? current;
    public static AppSettings Current => current ??= Load();

    private static AppSettings Load()
    {
        try
        {
            if (File.Exists(FilePath) && new FileInfo(FilePath).Length < 64_000)
                return JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(FilePath)) ?? new();
        }
        catch (Exception error) when (error is IOException or JsonException or UnauthorizedAccessException) { }
        return new();
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
            File.WriteAllText(FilePath + ".tmp", JsonSerializer.Serialize(this));
            File.Move(FilePath + ".tmp", FilePath, true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException) { }
    }
}
