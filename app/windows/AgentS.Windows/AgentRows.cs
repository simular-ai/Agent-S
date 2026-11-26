using System.ComponentModel;
using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>A sidebar row: colored initial avatar, name, inference badge and the latest message, like the macOS app.</summary>
public sealed class AgentRow(Conversation conversation) : INotifyPropertyChanged
{
    /// <summary>The macOS avatar colors, by name (Color menu); an agent without a choice gets one from its id.</summary>
    public static readonly (string Name, global::Windows.UI.Color Color)[] Colors =
    [
        ("Black", global::Windows.UI.Color.FromArgb(255, 17, 17, 17)),
        ("Blue", global::Windows.UI.Color.FromArgb(255, 59, 110, 240)),
        ("Green", global::Windows.UI.Color.FromArgb(255, 36, 150, 84)),
        ("Orange", global::Windows.UI.Color.FromArgb(255, 236, 116, 32)),
        ("Purple", global::Windows.UI.Color.FromArgb(255, 139, 76, 222)),
        ("Red", global::Windows.UI.Color.FromArgb(255, 221, 54, 72)),
        ("Teal", global::Windows.UI.Color.FromArgb(255, 20, 150, 160)),
    ];

    private bool working;
    private string? badgeGlyph, badgeTip;
    public Conversation Conversation { get; } = conversation;
    public string Name => Conversation.Name;
    public string Initial => Conversation.Name.Trim() is { Length: > 0 } name ? char.ToUpperInvariant(name[0]).ToString() : "A";
    public Brush Avatar => new SolidColorBrush(Colors.Any(c => c.Name == Conversation.Color)
        ? Colors.First(c => c.Name == Conversation.Color).Color
        : Colors[(uint)StableHash(Conversation.Id) % Colors.Length].Color);
    public string Preview
    {
        get
        {
            var last = Conversation.Entries.LastOrDefault(entry => entry.Role is not "Activity" && !string.IsNullOrWhiteSpace(entry.Text));
            // Before there's a conversation: "Claude Code · gemma4", as on macOS.
            if (last is null) return Conversation.Summary();
            return (last.Role is "Agent" or "Sai" ? Markdown.Plain(last.Text) : last.Text).ReplaceLineEndings(" ").Trim();
        }
    }
    public bool Working { get => working; set { if (working == value) return; working = value; Changed(nameof(Working)); } }
    public string BadgeGlyph => badgeGlyph ?? "";
    public string BadgeTip => badgeTip ?? "";
    public Visibility BadgeVisibility => badgeGlyph is null ? Visibility.Collapsed : Visibility.Visible;

    /// <summary>The inference badge: where the model runs, with "{label} · {provider · model}" as its tooltip.</summary>
    public void SetBadge(string? glyph, string? tip)
    {
        if (glyph == badgeGlyph && tip == badgeTip) return;
        badgeGlyph = glyph; badgeTip = tip;
        Changed(nameof(BadgeGlyph)); Changed(nameof(BadgeTip)); Changed(nameof(BadgeVisibility));
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    public void Refresh() { Changed(nameof(Preview)); Changed(nameof(Name)); Changed(nameof(Initial)); Changed(nameof(Avatar)); }
    private void Changed(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private static int StableHash(string value)
    {
        unchecked
        {
            int hash = 17;
            foreach (var c in value) hash = hash * 31 + c;
            return hash & int.MaxValue;
        }
    }
}

/// <summary>A sidebar section: Private, This PC or Cloud. Empty, it links to creating an agent there.</summary>
public sealed class AgentGroup(string title, string glyph, string kind, string emptyLink, IEnumerable<AgentRow> rows) : List<AgentRow>(rows)
{
    public string Title { get; } = title;
    public string Glyph { get; } = glyph;
    public string Kind { get; } = kind;
    public string EmptyLink { get; } = emptyLink;
    public Visibility EmptyVisibility => Count == 0 ? Visibility.Visible : Visibility.Collapsed;
}
