using AgentS.Core;
using Microsoft.UI.Xaml.Controls;

namespace AgentS.Windows;

public sealed partial class MainWindow
{
    /// <summary>
    /// A place a conversation is drawn: the main chat, or the floating Agent Activity window. Both draw the same
    /// rows with the same code (Draw/Change), so the small window shows exactly what the big one does.
    /// </summary>
    private sealed class ChatSurface(StackPanel panel, ScrollViewer scroll, Func<Conversation?> owner, bool compact)
    {
        public StackPanel Panel { get; } = panel;
        public ScrollViewer Scroll { get; } = scroll;
        /// <summary>Whose conversation this surface shows now.</summary>
        public Func<Conversation?> Owner { get; } = owner;
        /// <summary>The small window: smaller text and narrower bubbles.</summary>
        public bool Compact { get; } = compact;
        public Dictionary<string, TextBlock> Text { get; } = [];
        public Dictionary<string, StackPanel> Markdown { get; } = [];
        public Dictionary<string, StackPanel> Approvals { get; } = [];
        public Dictionary<string, ActivityGroup> Groups { get; } = [];
        /// <summary>The tool-step group the next step joins, while steps follow each other; any other row ends it.</summary>
        public ActivityGroup? OpenGroup { get; set; }
        public bool Following => Scroll.ScrollableHeight - Scroll.VerticalOffset < 64;

        public void Clear()
        {
            Panel.Children.Clear(); Text.Clear(); Markdown.Clear(); Approvals.Clear(); Groups.Clear(); OpenGroup = null;
        }
    }

    private ChatSurface? mainChat;
    private ChatSurface Chat => mainChat ??= new ChatSurface(Messages, Transcript, () => selected, compact: false);
    /// <summary>The floating window's surface, once it exists.</summary>
    private ChatSurface? activityChat;
    private IEnumerable<ChatSurface> Surfaces => activityChat is null ? [Chat] : [Chat, activityChat];

    // The main chat's state, by the names the rest of the window uses.
    private Dictionary<string, TextBlock> entryText => Chat.Text;
    private Dictionary<string, StackPanel> entryMarkdown => Chat.Markdown;
    private Dictionary<string, StackPanel> approvalControls => Chat.Approvals;
    private Dictionary<string, ActivityGroup> stepGroups => Chat.Groups;
    private ActivityGroup? openGroup { get => Chat.OpenGroup; set => Chat.OpenGroup = value; }

    /// <summary>Redraws a surface from its owner's whole conversation.</summary>
    private void Redraw(ChatSurface surface)
    {
        surface.Clear();
        if (surface.Owner() is { } owner) foreach (var entry in owner.Entries) Draw(entry, surface);
        surface.OpenGroup?.Refresh();
        DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () =>
        { surface.Scroll.UpdateLayout(); surface.Scroll.ChangeView(null, surface.Scroll.ScrollableHeight, null, true); });
    }
}
