using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>
/// The macOS ChatActivityBar: one folded row for an uninterrupted stretch of tool steps ("3 steps",
/// "1 failed"), each step expanding to its Tool, Input and Output.
/// </summary>
internal sealed class ActivityGroup : StackPanel
{
    private readonly List<(ChatEntry Entry, StepRow Row)> steps = [];
    private readonly Func<string, Brush> theme;
    private readonly Func<ActivityGroup, bool> isRunning;
    private readonly StackPanel body = new() { Spacing = 6, Margin = new Thickness(10, 0, 0, 0), Visibility = Visibility.Collapsed };
    private readonly FontIcon icon = new() { FontSize = 14 };
    private readonly ProgressRing ring = new() { Width = 14, Height = 14, IsActive = false, Visibility = Visibility.Collapsed };
    private readonly TextBlock title = new() { FontSize = 14, TextTrimming = TextTrimming.CharacterEllipsis, MaxLines = 1 };
    private readonly TextBlock count = new() { FontSize = 12, Opacity = 0.65, VerticalAlignment = VerticalAlignment.Center };
    private readonly TextBlock issue = new() { FontSize = 12, VerticalAlignment = VerticalAlignment.Center };
    private readonly FontIcon chevron = new() { Glyph = "", FontSize = 9, Opacity = 0.5 };
    private bool expanded;

    public ActivityGroup(Func<string, Brush> theme, Func<ActivityGroup, bool> isRunning)
    {
        this.theme = theme; this.isRunning = isRunning;
        Spacing = 8;
        HorizontalAlignment = HorizontalAlignment.Left;
        var row = new Grid { ColumnSpacing = 8 };
        foreach (var width in new[] { GridLength.Auto, new GridLength(1, GridUnitType.Star), GridLength.Auto, GridLength.Auto, GridLength.Auto })
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = width });
        var status = new Grid(); status.Children.Add(icon); status.Children.Add(ring);
        row.Children.Add(status);
        Grid.SetColumn(title, 1); row.Children.Add(title);
        Grid.SetColumn(count, 2); row.Children.Add(count);
        Grid.SetColumn(issue, 3); row.Children.Add(issue);
        chevron.VerticalAlignment = VerticalAlignment.Center;
        Grid.SetColumn(chevron, 4); row.Children.Add(chevron);
        var header = new Button
        {
            Content = row, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderBrush = theme("CardBorder"),
            BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(14), Padding = new Thickness(14, 8, 14, 8),
            HorizontalContentAlignment = HorizontalAlignment.Stretch, MaxWidth = 560, MinWidth = 220,
        };
        header.Click += (_, _) =>
        {
            expanded = !expanded;
            body.Visibility = expanded ? Visibility.Visible : Visibility.Collapsed;
            chevron.Glyph = expanded ? "" : "";
            Refresh();
        };
        Children.Add(header);
        Children.Add(body);
    }

    public bool Contains(ChatEntry entry) => steps.Any(s => s.Entry == entry);

    public void Add(ChatEntry entry)
    {
        var row = new StepRow(entry, theme);
        steps.Add((entry, row));
        body.Children.Add(row);
        Refresh();
    }

    public static string StepTitle(ChatEntry entry) => ToolActivity.Title(entry.Tool ?? "", entry.Text, entry.Input);

    /// <summary>The same summary as macOS: the running step's title, else the single step's, else "N steps"; plus failures.</summary>
    public void Refresh()
    {
        bool running = isRunning(this);
        int failures = steps.Count(s => s.Entry.Failed);
        int unfinished = running ? 0 : steps.Count(s => !s.Entry.Done && !s.Entry.Failed);
        var active = running ? steps.LastOrDefault(s => !s.Entry.Done && !s.Entry.Failed).Entry : null;
        var stepCount = steps.Count == 1 ? "1 step" : $"{steps.Count} steps";
        var action = active is not null ? StepTitle(active) : running ? "Working…" : null;
        var single = steps.Count == 1 ? StepTitle(steps[0].Entry) : null;
        title.Text = action ?? single ?? stepCount;
        count.Text = action is null ? "" : stepCount;
        var issues = new[] { failures > 0 ? $"{failures} failed" : null, unfinished > 0 ? $"{unfinished} unfinished" : null }.OfType<string>().ToArray();
        issue.Text = string.Join(", ", issues);
        issue.Foreground = failures > 0 ? new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72)) : theme("MutedText");
        ring.IsActive = running; ring.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        icon.Visibility = running ? Visibility.Collapsed : Visibility.Visible;
        icon.Glyph = failures > 0 ? "" : issues.Length == 0 ? "" : "";
        icon.Foreground = failures > 0 ? new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72))
            : issues.Length == 0 ? new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 52, 199, 89)) : theme("MutedText");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(this, title.Text);
        foreach (var (_, row) in steps) row.Refresh(running);
    }

    /// <summary>One step: icon, title and a chevron; open, its Tool / Input / Output in monospace.</summary>
    private sealed class StepRow : StackPanel
    {
        private readonly ChatEntry entry;
        private readonly FontIcon icon = new() { FontSize = 13 };
        private readonly TextBlock title = new() { FontSize = 14, TextTrimming = TextTrimming.CharacterEllipsis, MaxLines = 1 };
        private readonly StackPanel details = new() { Spacing = 8 };
        private readonly Border detailCard;
        private readonly Func<string, Brush> theme;

        public StepRow(ChatEntry entry, Func<string, Brush> theme)
        {
            this.entry = entry; this.theme = theme;
            Spacing = 6;
            var head = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            var chevron = new FontIcon { Glyph = "", FontSize = 9, Opacity = 0.5, VerticalAlignment = VerticalAlignment.Center };
            head.Children.Add(icon); head.Children.Add(title); head.Children.Add(chevron);
            var button = new Button
            {
                Content = head, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderBrush = theme("CardBorder"),
                BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(14), Padding = new Thickness(14, 6, 14, 6), MaxWidth = 540,
                HorizontalAlignment = HorizontalAlignment.Left,
            };
            detailCard = new Border
            {
                Child = new ScrollViewer { Content = details, MaxHeight = 180, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled },
                Background = theme("ChipBackground"), CornerRadius = new CornerRadius(10), Padding = new Thickness(10), MaxWidth = 540,
                HorizontalAlignment = HorizontalAlignment.Left, Visibility = Visibility.Collapsed,
            };
            button.Click += (_, _) =>
            {
                bool open = detailCard.Visibility != Visibility.Visible;
                detailCard.Visibility = open ? Visibility.Visible : Visibility.Collapsed;
                chevron.Glyph = open ? "" : "";
                if (open) FillDetails();
            };
            Children.Add(button); Children.Add(detailCard);
        }

        public void Refresh(bool running)
        {
            title.Text = StepTitle(entry);
            icon.Glyph = entry.Failed ? "" : entry.Done ? "" : running ? "" : "";
            icon.Foreground = entry.Failed ? new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72))
                : entry.Done ? new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 52, 199, 89)) : theme("MutedText");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(this,
                $"{title.Text}, {(entry.Failed ? "Failed" : entry.Done ? "Completed" : running ? "Running" : "Unfinished")}");
            if (detailCard.Visibility == Visibility.Visible) FillDetails();
        }

        private void FillDetails()
        {
            details.Children.Clear();
            var input = entry.Input ?? (entry.Text == entry.Tool || entry.Text == title.Text ? "" : entry.Text);
            foreach (var (label, value) in new[] { ("Tool", entry.Tool ?? ""), ("Input", input), ("Output", entry.Output ?? "") })
            {
                if (string.IsNullOrEmpty(value)) continue;
                var part = new StackPanel { Spacing = 3 };
                part.Children.Add(new TextBlock { Text = label, FontSize = 12, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Opacity = 0.7 });
                part.Children.Add(new TextBlock
                {
                    Text = value, FontSize = 12, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true,
                    FontFamily = new FontFamily("Cascadia Mono, Consolas"),
                });
                details.Children.Add(part);
            }
        }
    }
}
