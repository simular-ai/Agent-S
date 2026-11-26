using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace AgentS.Windows;

/// <summary>
/// Follow-ups sent while a turn runs, as on macOS: they wait above the composer ("Next turn") and are
/// sent when the turn ends. A failed or stopped turn pauses them ("Send next"); "Stop & send" stops the
/// current turn and sends that follow-up as soon as the stop is confirmed. One turn runs at a time, so a
/// message for another agent waits too ("Waiting for …").
/// </summary>
public sealed partial class MainWindow
{
    private sealed record FollowUp(string Id, string Text);
    private sealed class FollowUpQueue
    {
        public List<FollowUp> Items { get; } = [];
        public bool Paused { get; set; }
        /// <summary>The follow-up "Stop & send" chose; it goes once the turn has stopped.</summary>
        public string? Forced { get; set; }
    }
    private readonly Dictionary<Conversation, FollowUpQueue> followUps = [];

    private FollowUpQueue QueueFor(Conversation owner) =>
        followUps.TryGetValue(owner, out var queue) ? queue : followUps[owner] = new FollowUpQueue();

    private void Enqueue(Conversation owner, string text)
    {
        QueueFor(owner).Items.Add(new FollowUp(Guid.NewGuid().ToString("N"), text));
        RenderQueue();
    }

    /// <summary>After a turn: a forced follow-up goes regardless; otherwise the next one only after success.</summary>
    private async Task ContinueQueue(Conversation owner, bool succeeded)
    {
        var queue = QueueFor(owner);
        FollowUp? next = null;
        if (queue.Forced is { } forced && queue.Items.FirstOrDefault(i => i.Id == forced) is { } chosen)
        { next = chosen; queue.Forced = null; queue.Paused = false; }
        else if (queue.Items.Count > 0)
        {
            if (succeeded && !queue.Paused) next = queue.Items[0];
            else queue.Paused = true;
        }
        queue.Forced = null;
        if (next is null)
        {
            // Another agent's message that waited for this turn goes next.
            var waiting = followUps.FirstOrDefault(pair => pair.Key != owner && conversations.Contains(pair.Key) && pair.Value.Items.Count > 0 && !pair.Value.Paused);
            if (waiting.Key is { } other) { owner = other; next = waiting.Value.Items[0]; queue = waiting.Value; }
        }
        RenderQueue();
        if (next is null || running) return;
        if (!CanRun(owner)) { queue.Paused = true; RenderQueue(); return; }
        queue.Items.Remove(next);
        RenderQueue();
        await RunTurn(owner, next.Text);
    }

    private void StopAndSend(Conversation owner, FollowUp item)
    {
        var queue = QueueFor(owner);
        queue.Forced = item.Id;
        RenderQueue();
        if (running && runOwner == owner) StopTurn(this, new RoutedEventArgs());
    }

    private async void SendNext(Conversation owner)
    {
        var queue = QueueFor(owner);
        if (running || queue.Items.Count == 0 || !CanRun(owner)) return;
        queue.Paused = false;
        var next = queue.Items[0];
        queue.Items.RemoveAt(0);
        RenderQueue();
        await RunTurn(owner, next.Text);
    }

    /// <summary>The selected agent's follow-ups, above the composer.</summary>
    private void RenderQueue()
    {
        QueueStrip.Children.Clear();
        if (selected is not { } owner || !followUps.TryGetValue(owner, out var queue) || queue.Items.Count == 0)
        { QueueStrip.Visibility = Visibility.Collapsed; return; }
        QueueStrip.Visibility = Visibility.Visible;
        for (var index = 0; index < queue.Items.Count; index++)
        {
            var item = queue.Items[index];
            bool mine = running && runOwner == owner;
            string status = queue.Forced == item.Id ? "Stopping…"
                : mine ? "Next turn"
                : running && runOwner is { } other ? $"Waiting for {other.Name}"
                : queue.Paused ? "Paused" : "Next turn";
            var grid = new Grid { ColumnSpacing = 10 };
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var text = new StackPanel { Spacing = 1, VerticalAlignment = VerticalAlignment.Center };
            text.Children.Add(new TextBlock { Text = item.Text.ReplaceLineEndings(" "), TextTrimming = TextTrimming.CharacterEllipsis, MaxLines = 1 });
            text.Children.Add(new TextBlock { Text = status, FontSize = 12, Foreground = Theme("MutedText") });
            grid.Children.Add(text);
            var actions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6, VerticalAlignment = VerticalAlignment.Center };
            if (mine && queue.Forced is null)
            {
                var force = new Button { Content = "Stop & send", Padding = new Thickness(10, 4, 10, 4) };
                ToolTipService.SetToolTip(force, "Stop this turn and send this follow-up once it has stopped");
                force.Click += (_, _) => StopAndSend(owner, item);
                actions.Children.Add(force);
            }
            if (!running && queue.Paused && index == 0)
            {
                var sendNext = new Button { Content = "Send next", Padding = new Thickness(10, 4, 10, 4) };
                sendNext.Click += (_, _) => SendNext(owner);
                actions.Children.Add(sendNext);
            }
            Grid.SetColumn(actions, 1); grid.Children.Add(actions);
            var remove = new Button { Content = new FontIcon { Glyph = "", FontSize = 10 }, Padding = new Thickness(8), Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0) };
            ToolTipService.SetToolTip(remove, "Remove this follow-up");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(remove, "Remove this follow-up");
            remove.Click += (_, _) =>
            {
                queue.Items.Remove(item);
                if (queue.Forced == item.Id) queue.Forced = null;
                if (queue.Items.Count == 0) queue.Paused = false;
                RenderQueue();
            };
            Grid.SetColumn(remove, 2); grid.Children.Add(remove);
            QueueStrip.Children.Add(new Border
            {
                Child = grid, CornerRadius = new CornerRadius(14), Padding = new Thickness(14, 8, 6, 8),
                Background = Theme("ChipBackground"), MaxWidth = 760,
            });
        }
    }
}
