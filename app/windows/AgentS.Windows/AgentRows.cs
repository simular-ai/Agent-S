using System.ComponentModel;
using AgentS.Core;
using Microsoft.UI;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>A sidebar row: colored initial avatar, name and the latest message, like the macOS app.</summary>
public sealed class AgentRow(Conversation conversation) : INotifyPropertyChanged
{
    private static readonly global::Windows.UI.Color[] Palette =
    [
        global::Windows.UI.Color.FromArgb(255, 17, 17, 17),    // black
        global::Windows.UI.Color.FromArgb(255, 59, 110, 240),  // blue
        global::Windows.UI.Color.FromArgb(255, 139, 76, 222),  // purple
        global::Windows.UI.Color.FromArgb(255, 221, 54, 72),   // red
        global::Windows.UI.Color.FromArgb(255, 36, 150, 84),   // green
        global::Windows.UI.Color.FromArgb(255, 236, 116, 32),  // orange
    ];

    private bool working;
    public Conversation Conversation { get; } = conversation;
    public string Name => Conversation.Name;
    public string Initial => Conversation.Name.Trim() is { Length: > 0 } name ? char.ToUpperInvariant(name[0]).ToString() : "A";
    public Brush Avatar { get; } = new SolidColorBrush(Palette[(uint)StableHash(conversation.Id) % Palette.Length]);
    public string Preview
    {
        get
        {
            var last = Conversation.Entries.LastOrDefault(entry => entry.Role is not "Activity" && !string.IsNullOrWhiteSpace(entry.Text));
            if (last is null) return Conversation.Target switch
            {
                ConversationTarget.Private => "Private · this PC",
                ConversationTarget.ThisPc => "This PC · asks before each action",
                ConversationTarget.Box => "Simular Box",
                _ => "Sai cloud computer",
            };
            return (last.Role is "Agent" or "Sai" ? Markdown.Plain(last.Text) : last.Text).ReplaceLineEndings(" ").Trim();
        }
    }
    public bool Working { get => working; set { if (working == value) return; working = value; Changed(nameof(Working)); } }

    public event PropertyChangedEventHandler? PropertyChanged;
    public void Refresh() { Changed(nameof(Preview)); Changed(nameof(Name)); }
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

/// <summary>A sidebar section: Private (this PC) or Cloud.</summary>
public sealed class AgentGroup(string title, string glyph, IEnumerable<AgentRow> rows) : List<AgentRow>(rows)
{
    public string Title { get; } = title;
    public string Glyph { get; } = glyph;
}
