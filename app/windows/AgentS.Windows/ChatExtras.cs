using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace AgentS.Windows;

/// <summary>macOS chat details: the pill text, empty states with sample tasks, the status banner, jump to latest.</summary>
public sealed partial class MainWindow
{
    /// <summary>"{harness} · {model}", as the macOS pill reads.</summary>
    private string PillText(Conversation agent) => agent.Target switch
    {
        ConversationTarget.Sai => $"Sai · {(agent.ModelId is null or "auto" ? "auto" : models.FirstOrDefault(m => m.Id == agent.ModelId)?.Name ?? agent.ModelId)}",
        ConversationTarget.ThisPc or ConversationTarget.Box => $"Claude Code · {ModelName(agent)}",
        _ => $"{(agent.Harness is { } h && h.RunsInComputer() ? h.DisplayName() : "Agent S")} · {ModelName(agent)}",
    };

    private static string ModelName(Conversation agent) => agent.Provider == ModelProvider.OpenRouter
        ? agent.ModelId ?? "OpenRouter" : OllamaModels.DisplayName(agent.ModelId ?? OllamaModels.Default);

    private static readonly string[] StarterPrompts =
    [
        "Open example.com and summarize the page.",
        "Compare the Moon and Mars Wikipedia pages in two tabs.",
        "Save a short checklist to ~/work/checklist.txt and open it in the private computer's browser.",
    ];

    /// <summary>The macOS empty states, per kind of agent; Private agents also get three sample tasks.</summary>
    private void UpdateEmptyChat()
    {
        EmptyChat.Visibility = selected is null || selected.Entries.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        SamplePrompts.Children.Clear();
        if (selected is not { } agent)
        {
            EmptyChatTitle.Text = "Create an agent to get started";
            EmptyChatText.Text = "Agents work on a private computer on this PC, on this PC itself, or on a computer in the cloud. Choose + to create one.";
            return;
        }
        bool openRouter = agent.Provider == ModelProvider.OpenRouter;
        (EmptyChatTitle.Text, EmptyChatText.Text) = agent.Target switch
        {
            ConversationTarget.Sai => ("Sai works on your Sai computer", "All in one: Sai's agent, model and cloud computer. Nothing runs on this PC."),
            ConversationTarget.ThisPc => ("Works on this PC", "It uses your PC's apps, terminal and files. Before each step it shows you exactly what it will do, and runs it only when you allow it."),
            ConversationTarget.Box => ("Working on your Simular Box", "The agent runs in this app and uses only your box's tools."),
            _ => ("Private computer on this PC", openRouter
                ? "The private computer runs on this PC. OpenRouter processes model requests."
                : "The agent works in a Linux computer on this PC. Open the computer to watch or take over."),
        };
        if (agent.Target != ConversationTarget.Private || !PrivateComputer.IsConfigured) return;
        foreach (var prompt in StarterPrompts)
        {
            var row = new Grid { ColumnSpacing = 8 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.Children.Add(new TextBlock { Text = prompt, TextWrapping = TextWrapping.Wrap, FontSize = 14 });
            var arrow = new FontIcon { Glyph = "", FontSize = 12, Foreground = Theme("MutedText"), VerticalAlignment = VerticalAlignment.Center };
            Grid.SetColumn(arrow, 1); row.Children.Add(arrow);
            var button = new Button
            {
                Content = row, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch,
                Background = Theme("ChipBackground"), BorderThickness = new Thickness(0), CornerRadius = new CornerRadius(10), Padding = new Thickness(12, 8, 12, 8),
            };
            ToolTipService.SetToolTip(button, "Use this prompt as your draft");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetHelpText(button, "Loads this prompt in the draft");
            // Fills the draft to edit before sending; it doesn't send.
            button.Click += (_, _) => { Composer.Text = prompt; Composer.Focus(FocusState.Programmatic); Composer.SelectionStart = prompt.Length; };
            SamplePrompts.Children.Add(button);
        }
    }

    // ---- Status banner above the composer ---------------------------------------------------------

    /// <summary>As on macOS: what keeps the selected agent from working right now, with the one action that fixes it.</summary>
    private void UpdateStatusBanner()
    {
        (string Text, string? Action)? banner = null;
        if (selected is { } agent && !(running && runOwner == agent))
        {
            if (agent.Target == ConversationTarget.Private)
            {
                if (!PrivateComputer.IsConfigured) banner = ("Set up the private computer to chat.", "Set up");
                else if (agent.Provider == ModelProvider.OpenRouter && Secrets.ReadOpenRouter() is null) banner = ("Add your OpenRouter key to chat.", "Settings");
                else if (!privateScreenShown) banner = ("The computer is off. It starts with your next message.", "Turn on");
            }
            else if (agent.Target == ConversationTarget.Box && Secrets.ReadBox() is null) banner = ("Add your Simular Box key to chat.", "Settings");
            else if (agent.Target == ConversationTarget.Sai && client is null) banner = ("Connect your Sai account to chat.", "Settings");
            else if (agent.Target is ConversationTarget.ThisPc or ConversationTarget.Box && agent.Provider == ModelProvider.OpenRouter && Secrets.ReadOpenRouter() is null)
                banner = ("Add your OpenRouter key to chat.", "Settings");
        }
        StatusBanner.Visibility = banner is null ? Visibility.Collapsed : Visibility.Visible;
        if (banner is not { } shown) return;
        StatusBannerText.Text = shown.Text;
        StatusBannerAction.Content = shown.Action;
        StatusBannerAction.Visibility = shown.Action is null ? Visibility.Collapsed : Visibility.Visible;
        StatusBannerAction.IsEnabled = !busy;
    }

    private async void StatusBannerClick(object sender, RoutedEventArgs e)
    {
        switch (StatusBannerAction.Content as string)
        {
            case "Set up": preferPrivate = true; NewAgent(sender, e); break;
            case "Turn on": if (!panelOpen) TogglePanel(this, e); await StartPrivateScreen(); break;
            case "Settings": Accounts(sender, e); break;
        }
    }

    /// <summary>
    /// Chats saved before steps recorded their tool kept only a line of text ("web snapshot", "Clicked (…)").
    /// Those become steps, so older conversations fold like new ones; status lines ("Claude Code · model") stay as they are.
    /// </summary>
    private static void UpgradeSteps(Conversation conversation)
    {
        foreach (var entry in conversation.Entries.Where(e => e.Role == "Activity" && e.Tool is null))
        {
            var text = entry.Text;
            if (text.Contains(" · ") || text.StartsWith("Starting", StringComparison.Ordinal) || text.StartsWith("Getting", StringComparison.Ordinal)
                || text.StartsWith("First use", StringComparison.Ordinal) || text.StartsWith("Private desktop ready", StringComparison.Ordinal)) continue;
            var lines = text.Split('\n');
            var failure = lines.Skip(1).FirstOrDefault(l => l.StartsWith("Failed", StringComparison.Ordinal) || l.StartsWith("Denied", StringComparison.Ordinal));
            bool desktop = lines[0].EndsWith("in the private desktop.", StringComparison.Ordinal) || lines[0].StartsWith("Opened the offline demo", StringComparison.Ordinal);
            entry.Tool = desktop ? "desktop" : "Bash";
            entry.Text = lines[0];
            if (!desktop) entry.Input = lines[0];
            entry.Failed = failure is not null;
            entry.Done = failure is null;
            if (failure is not null) entry.Output = failure;
        }
    }

    // ---- Jump to latest ---------------------------------------------------------------------------

    private void InitializeChatExtras()
    {
        Transcript.ViewChanged += (_, _) =>
            JumpButton.Visibility = Transcript.ScrollableHeight - Transcript.VerticalOffset > 240 ? Visibility.Visible : Visibility.Collapsed;
        // As on macOS, an awake Box's screen stays live while you look at it, not only during a turn.
        var boxScreen = DispatcherQueue.CreateTimer();
        boxScreen.Interval = TimeSpan.FromSeconds(5);
        boxScreen.Tick += async (_, _) =>
        {
            if (!IsBox || !panelOpen || busy || boxState != "Running" || (running && runOwner == selected)) return;
            if (selected is not { } owner || BoxRunner.Client() is not { } box) return;
            try { if (await box.Screenshot(owner.BoxName) is { } png && selected == owner) await ShowPrivateScreen(png, ScreenKindFor(owner)!); }
            catch (Exception error) when (error is BoxException or TaskCanceledException or HttpRequestException) { }
        };
        boxScreen.Start();
        // The sidebar's inference badges follow Ollama (installed models, running or not).
        var badges = DispatcherQueue.CreateTimer();
        badges.Interval = TimeSpan.FromSeconds(60);
        badges.Tick += (_, _) => UpdateBadges();
        badges.Start();
    }

    private void JumpToLatest(object sender, RoutedEventArgs e) => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, false);
}
