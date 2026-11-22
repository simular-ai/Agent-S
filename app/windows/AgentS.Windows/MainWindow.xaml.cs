using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media.Imaging;
using Microsoft.UI.Windowing;
using global::Windows.Storage.Streams;
using global::Windows.System;

namespace AgentS.Windows;

public sealed partial class MainWindow : Window
{
    private readonly ObservableCollection<Conversation> conversations = [];
    private readonly HistoryStore store = new();
    private readonly Dictionary<string, TextBlock> entryText = [];
    private readonly Dictionary<string, StackPanel> approvalControls = [];
    private readonly Dictionary<string, (Uri Uri, DateTime Until)> liveLinks = [];
    private Computer[] computers = [];
    private SaiModel[] models = [];
    private SaiClient? client;
    private Conversation? selected;
    private CancellationTokenSource? run;
    private bool busy, running, stopRequested, stopped, allowClose, storageReady, closing;
    private string turn = "";
    private string? currentSession;
    private readonly Dictionary<Conversation, AgentRow> rows = [];
    private readonly Microsoft.UI.Xaml.Data.CollectionViewSource agentGroups = new() { IsSourceGrouped = true };
    private bool panelOpen = true, rebuilding, privateScreenShown, preferPrivate;
    private HarnessUsage? usage;
    /// <summary>Whether the screen panel's image came from This PC (a pc screenshot) rather than the Private computer.</summary>
    private bool screenFromPc;
    /// <summary>This PC: approve the rest of the current turn's actions without asking (reset every turn).</summary>
    private bool approveRestOfTurn;
    private Uri? liveViewUri;

    public MainWindow()
    {
        InitializeComponent();
        conversations.CollectionChanged += (_, _) => RebuildAgents();
        var logo = Path.Combine(AppContext.BaseDirectory, "Assets", "AgentS.png");
        if (File.Exists(logo)) EmptyChatLogo.Source = new BitmapImage(new Uri(logo));
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        var width = Math.Min(1360, area.Width);
        var height = Math.Min(860, area.Height);
        AppWindow.Resize(new global::Windows.Graphics.SizeInt32(width, height));
        AppWindow.Move(new global::Windows.Graphics.PointInt32(area.X + (area.Width - width) / 2,
            area.Y + (area.Height - height) / 2));
        var icon = Path.Combine(AppContext.BaseDirectory, "Assets", "AgentS.ico");
        if (File.Exists(icon)) AppWindow.SetIcon(icon);
        AppWindow.Closing += Closing;
        Closed += (_, _) => { run?.Cancel(); client?.Dispose(); };
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            foreach (var conversation in await store.Read()) conversations.Add(conversation);
            storageReady = true;
            var key = Secrets.Read();
            if (key is not null) client = new SaiClient(key);
            Select(conversations.FirstOrDefault(item => item.Target == ConversationTarget.Private) ?? conversations.FirstOrDefault());
        }
        catch (Exception error) { Error(error); }
        Controls();
    }

    private bool IsPrivate => selected?.Target == ConversationTarget.Private;
    private bool IsThisPc => selected?.Target == ConversationTarget.ThisPc;

    // ---- Sidebar ---------------------------------------------------------------------------------

    private AgentRow Row(Conversation conversation)
    {
        if (!rows.TryGetValue(conversation, out var row)) rows[conversation] = row = new AgentRow(conversation);
        return row;
    }
    private void RebuildAgents()
    {
        foreach (var gone in rows.Keys.Except(conversations).ToArray()) rows.Remove(gone);
        var groups = new List<AgentGroup>();
        var local = conversations.Where(c => c.Target == ConversationTarget.Private).Select(Row).ToArray();
        var cloud = conversations.Where(c => c.Target == ConversationTarget.Sai).Select(Row).ToArray();
        var pc = conversations.Where(c => c.Target == ConversationTarget.ThisPc).Select(Row).ToArray();
        if (local.Length > 0) groups.Add(new AgentGroup("Private", "", local));
        if (pc.Length > 0) groups.Add(new AgentGroup("This PC", "", pc));
        if (cloud.Length > 0) groups.Add(new AgentGroup("Cloud", "", cloud));
        rebuilding = true;
        try
        {
            agentGroups.Source = groups;
            Agents.ItemsSource = agentGroups.View;
            Agents.SelectedItem = selected is null ? null : Row(selected);
        }
        finally { rebuilding = false; }
        EmptyAgents.Visibility = conversations.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }
    private void Select(Conversation? conversation)
    {
        if (conversation is null) { Agents.SelectedItem = null; SelectAgent(Agents, null!); return; }
        Agents.SelectedItem = Row(conversation);
    }
    private async void DeleteAgentFromMenu(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.Tag is not AgentRow row || busy) return;
        if (await Dialog($"Delete {row.Name}?", "This deletes the local conversation. It does not delete the computer or remote task history.", "Delete").ShowAsync() != ContentDialogResult.Primary) return;
        bool wasSelected = selected == row.Conversation;
        conversations.Remove(row.Conversation);
        if (wasSelected) { selected = null; Select(conversations.FirstOrDefault()); }
        await Persist();
    }

    // ---- Chat chrome -----------------------------------------------------------------------------

    private void ComposerPreviewKeyDown(object sender, Microsoft.UI.Xaml.Input.KeyRoutedEventArgs e)
    {
        // Enter sends; Shift+Enter adds a new line, like the macOS composer.
        if (e.Key != VirtualKey.Enter) return;
        var shift = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift)
            .HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
        if (shift) return;
        e.Handled = true;
        if (SendButton.IsEnabled) Send(SendButton, new RoutedEventArgs());
    }
    private void ComposerChanged(object sender, TextChangedEventArgs e) => Controls();

    private void TogglePanel(object sender, RoutedEventArgs e)
    {
        panelOpen = !panelOpen;
        ComputerColumn.Width = panelOpen ? new GridLength(380) : new GridLength(0);
        ComputerPanel.Visibility = panelOpen ? Visibility.Visible : Visibility.Collapsed;
        if (panelOpen) _ = UpdateScreen();
    }

    private async void OpenModelMenu(object sender, RoutedEventArgs e)
    {
        if (selected is not { } owner || busy) return;
        var menu = new MenuFlyout { Placement = Microsoft.UI.Xaml.Controls.Primitives.FlyoutPlacementMode.Bottom };
        IEnumerable<(string Id, string Label)> choices;
        if (owner.Target == ConversationTarget.ThisPc)
        {
            // Claude Code on this PC: a model from Ollama, or from OpenRouter with the saved key.
            string[] installed = [];
            try { using var agent = new LocalAgent(); installed = await agent.Models(); } catch { }
            void AddChoice(ModelProvider provider, string id)
            {
                var item = new ToggleMenuFlyoutItem { Text = id, IsChecked = owner.Provider == provider && owner.ModelId == id };
                item.Click += async (_, _) => { owner.Provider = provider; owner.ModelId = id; Controls(); await Persist(); };
                menu.Items.Add(item);
            }
            menu.Items.Add(new MenuFlyoutItem { Text = "Ollama on this PC", IsEnabled = false });
            if (installed.Length == 0) menu.Items.Add(new MenuFlyoutItem { Text = "Ollama isn't running or has no models", IsEnabled = false });
            foreach (var id in installed) AddChoice(ModelProvider.Ollama, id);
            menu.Items.Add(new MenuFlyoutSeparator());
            menu.Items.Add(new MenuFlyoutItem { Text = Secrets.ReadOpenRouter() is null ? "OpenRouter (add a key in Accounts)" : "OpenRouter", IsEnabled = false });
            var openRouterChoices = ThisPcAgent.SuggestedOpenRouterModels.ToList();
            if (owner.Provider == ModelProvider.OpenRouter && owner.ModelId is { } current && !openRouterChoices.Contains(current)) openRouterChoices.Insert(0, current);
            if (Secrets.ReadOpenRouter() is not null) foreach (var id in openRouterChoices) AddChoice(ModelProvider.OpenRouter, id);
            menu.ShowAt(ModelPill);
            return;
        }
        if (owner.Target == ConversationTarget.Private)
        {
            string[] installed = [];
            try { using var agent = new LocalAgent(); installed = await agent.Models(); } catch { }
            choices = installed.Select(m => (m, m));
            if (!installed.Any())
                menu.Items.Add(new MenuFlyoutItem { Text = "Ollama isn't running or has no models", IsEnabled = false });
        }
        else
        {
            if (models.Length == 0 && client is not null) await Refresh();
            choices = new[] { ("auto", "Automatic") }.Concat(models.Where(m => m.Id != "auto").Select(m => (m.Id, m.Name)));
        }
        foreach (var (id, label) in choices)
        {
            var item = new ToggleMenuFlyoutItem { Text = label, IsChecked = (owner.ModelId ?? "auto") == id };
            item.Click += async (_, _) => { owner.ModelId = id; Controls(); await Persist(); };
            menu.Items.Add(item);
        }
        if (owner.Target == ConversationTarget.Private && HarnessRunner.Supported)
        {
            menu.Items.Add(new MenuFlyoutSeparator());
            menu.Items.Add(new MenuFlyoutItem { Text = "Harness", IsEnabled = false });
            foreach (var harness in Harnesses.All)
            {
                var item = new ToggleMenuFlyoutItem { Text = harness.DisplayName(), IsChecked = (owner.Harness ?? PrivateHarness.AgentS) == harness };
                item.Click += async (_, _) => { owner.Harness = harness; Controls(); await Persist(); };
                menu.Items.Add(item);
            }
        }
        menu.ShowAt(ModelPill);
    }

    private void Controls()
    {
        Agents.IsEnabled = AccountsButton.IsEnabled = !busy;
        NewAgentButton.IsEnabled = !busy && storageReady;
        Composer.IsEnabled = !busy && selected is not null;
        SendButton.IsEnabled = !busy && storageReady && !string.IsNullOrWhiteSpace(Composer.Text) && (selected?.Target switch
        {
            ConversationTarget.Private or ConversationTarget.ThisPc => true,
            ConversationTarget.Sai => client is not null,
            _ => false,
        });
        StopButton.IsEnabled = running && !stopRequested;
        SendButton.Visibility = running ? Visibility.Collapsed : Visibility.Visible;
        StopButton.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        WorkingChip.Visibility = running ? Visibility.Visible : Visibility.Collapsed;
        if (selected is not null) Row(selected).Working = running;

        // Model pill: "☁ Sai · auto" or "🔒 Private · gemma4", as on macOS.
        ModelPill.IsEnabled = selected is not null && !busy;
        ModelPillIcon.Glyph = IsPrivate ? "" : "";
        ModelPillText.Text = selected is null ? "No agent selected"
            : IsPrivate ? selected.Harness is { } h && h.RunsInComputer()
                ? $"Private · {h.DisplayName()} · {selected.ModelId ?? "gemma4"}" : $"Private · {selected.ModelId ?? "gemma4"}"
            : $"Sai · {(selected.ModelId is null or "auto" ? "auto" : models.FirstOrDefault(m => m.Id == selected.ModelId)?.Name ?? selected.ModelId)}";
        Composer.PlaceholderText = selected is null ? "Create an agent to start…"
            : IsPrivate ? $"Ask {selected.Name} to do something on its private computer…"
            : $"Ask {selected.Name} to do something on {CurrentComputer?.Name ?? "its computer"}…";
        EmptyChat.Visibility = selected is null || selected.Entries.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        EmptyChatTitle.Text = selected is null ? "Create an agent to get started" : $"Start a task with {selected.Name}";
        EmptyChatText.Text = selected is null
            ? "Agents work on a private computer on this PC or on a Sai computer in the cloud. Choose + to create one."
            : IsPrivate ? "It works on the private Linux computer on this PC. Watch it on the right."
            : "It works on your Sai computer. Watch it on the right.";
        if (IsThisPc && selected is not null)
        {
            ModelPillIcon.Glyph = "";
            ModelPillText.Text = $"This PC · Claude Code · {(selected.Provider == ModelProvider.OpenRouter ? "OpenRouter · " : "")}{selected.ModelId}";
            Composer.PlaceholderText = $"Ask {selected.Name} to do something on this PC…";
            EmptyChatText.Text = "It works on this PC itself and asks you before every action.";
        }

        // Computer header and actions.
        ScreenButton.IsEnabled = IsPrivate ? !busy && PrivateComputer.IsConfigured : !busy && CurrentComputer is not null;
        ScreenButtonIcon.Glyph = IsPrivate ? "" : "";
        ToolTipService.SetToolTip(ScreenButton, IsPrivate ? "Refresh screen" : "Open in browser");
        WakeButton.Text = IsPrivate ? "Stop computer" : "Wake computer";
        WakeButton.IsEnabled = !busy && (IsPrivate ? PrivateComputer.IsConfigured : CurrentComputer?.CanWake == true);
        RefreshButton.Text = IsPrivate ? "Reset computer" : "Refresh computers";
        RefreshButton.IsEnabled = !busy && (IsPrivate ? PrivateComputer.IsConfigured : client is not null);
        MoreButton.IsEnabled = selected is not null;
        ComputerName.Text = selected is null ? "No computer" : IsPrivate ? "Private computer" : CurrentComputer?.Name ?? "Sai computer";
        bool online = IsPrivate ? privateScreenShown : CurrentComputer?.Online == true;
        ComputerState.Text = selected is null ? "" : running ? "Working…" : IsPrivate
            ? PrivateComputer.IsConfigured ? privateScreenShown ? "Running on this PC" : "Off · starts with the next task" : "Not set up"
            : CurrentComputer?.Availability ?? (client is null ? "Connect Sai in Accounts" : "Unavailable");
        StatusDot.Fill = new Microsoft.UI.Xaml.Media.SolidColorBrush(online || running
            ? global::Windows.UI.Color.FromArgb(255, 52, 199, 89) : global::Windows.UI.Color.FromArgb(255, 154, 154, 154));
        if (IsThisPc)
        {
            // The agent works on this desktop; there's no separate computer to start, stop or reset.
            ScreenButton.IsEnabled = WakeButton.IsEnabled = RefreshButton.IsEnabled = false;
            ComputerName.Text = "This PC";
            ComputerState.Text = running ? "Working… it asks before each action" : "Your apps and files";
        }
        UpdatePlaceholder();
        UpdateContext();
    }

    // ---- Right sidebar: computer screen and context -----------------------------------------------

    private void UpdatePlaceholder()
    {
        bool showImage = (IsPrivate && !screenFromPc || IsThisPc && screenFromPc) && PrivateScreen.Source is not null;
        bool showLive = !IsPrivate && !IsThisPc && liveViewUri is not null && CurrentComputer?.Online == true;
        PrivateScreen.Visibility = showImage ? Visibility.Visible : Visibility.Collapsed;
        LiveView.Visibility = showLive ? Visibility.Visible : Visibility.Collapsed;
        ScreenPlaceholder.Visibility = showImage || showLive ? Visibility.Collapsed : Visibility.Visible;
        PlaceholderAction.Visibility = Visibility.Collapsed;
        if (selected is null)
        { PlaceholderTitle.Text = "No computer yet"; PlaceholderText.Text = "Create an agent to see its computer here."; return; }
        if (IsThisPc)
        { PlaceholderTitle.Text = "This PC"; PlaceholderText.Text = "The agent works on this desktop. Screenshots it takes appear here."; return; }
        if (IsPrivate)
        {
            if (!PrivateComputer.IsConfigured)
            { PlaceholderTitle.Text = "Set up this PC"; PlaceholderText.Text = "A one-time setup creates the private Linux computer."; PlaceholderAction.Content = "Set up"; }
            else
            { PlaceholderTitle.Text = "Private computer is off"; PlaceholderText.Text = "It starts automatically with the next task."; PlaceholderAction.Content = "Start computer"; }
            PlaceholderAction.Visibility = Visibility.Visible;
            return;
        }
        if (client is null)
        { PlaceholderTitle.Text = "Connect your Sai account"; PlaceholderText.Text = "Add your Sai API key to see this computer."; PlaceholderAction.Content = "Accounts"; PlaceholderAction.Visibility = Visibility.Visible; }
        else if (CurrentComputer is not { } computer)
        { PlaceholderTitle.Text = "Computer not found"; PlaceholderText.Text = "Refresh computers, or create a new agent."; PlaceholderAction.Content = "Refresh"; PlaceholderAction.Visibility = Visibility.Visible; }
        else if (!computer.Online)
        {
            PlaceholderTitle.Text = $"{computer.Name} is {computer.Availability.ToLowerInvariant()}";
            PlaceholderText.Text = computer.CanWake ? "Wake it to watch the agent work." : "It isn't available right now.";
            if (computer.CanWake) { PlaceholderAction.Content = "Wake computer"; PlaceholderAction.Visibility = Visibility.Visible; }
        }
        else { PlaceholderTitle.Text = "Loading screen…"; PlaceholderText.Text = ""; }
    }

    private async void PlaceholderActionClick(object sender, RoutedEventArgs e)
    {
        switch (PlaceholderAction.Content as string)
        {
            case "Set up": preferPrivate = true; NewAgent(sender, e); break;
            case "Start computer": await StartPrivateScreen(); break;
            case "Accounts": Accounts(sender, e); break;
            case "Refresh": await Refresh(); await UpdateScreen(); break;
            case "Wake computer": WakeComputer(sender, e); break;
        }
    }

    private async Task StartPrivateScreen()
    {
        await UsePrivateComputer(async computer =>
        {
            await computer.Start(CancellationToken.None);
            await ShowPrivateScreen(await computer.Capture(CancellationToken.None));
        });
    }

    /// <summary>Shows the selected computer: a live Sai page in the panel, or the latest private screenshot.</summary>
    private async Task UpdateScreen()
    {
        if (!panelOpen || selected is null || IsPrivate || client is null || CurrentComputer is not { Online: true } computer)
        { Controls(); return; }
        try
        {
            if (!liveLinks.TryGetValue(computer.Id, out var cached) || cached.Until < DateTime.UtcNow)
            { cached = (await client.LiveLink(computer.Id), DateTime.UtcNow.AddMinutes(23)); liveLinks[computer.Id] = cached; }
            if (CurrentComputer?.Id != computer.Id) return;
            if (liveViewUri != cached.Uri) { liveViewUri = cached.Uri; LiveView.Source = cached.Uri; }
        }
        catch { liveViewUri = null; }
        Controls();
    }

    private void UpdateContext()
    {
        ContextPanel.Visibility = selected is null ? Visibility.Collapsed : Visibility.Visible;
        SessionGrid.Children.Clear(); SessionGrid.RowDefinitions.Clear();
        ConversationSummary.Children.Clear();
        if (selected is not { } owner) return;
        void Pair(string label, string value, bool mono = false)
        {
            SessionGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var key = new TextBlock { Text = label, Opacity = 0.6 };
            var text = new TextBlock { Text = value, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true };
            if (mono) text.FontFamily = new Microsoft.UI.Xaml.Media.FontFamily("Cascadia Mono, Consolas");
            Grid.SetRow(key, SessionGrid.RowDefinitions.Count - 1);
            Grid.SetRow(text, SessionGrid.RowDefinitions.Count - 1); Grid.SetColumn(text, 1);
            SessionGrid.Children.Add(key); SessionGrid.Children.Add(text);
        }
        if (owner.Target == ConversationTarget.Private)
        {
            var harness = owner.Harness ?? PrivateHarness.AgentS;
            Pair("Harness", harness.RunsInComputer() ? $"{harness.DisplayName()} · in the Private computer" : "Agent S · local");
            Pair("Model", $"{owner.ModelId ?? "gemma4"} · Ollama on this PC");
            Pair("Computer", PrivateComputer.UsesWsl || !PrivateComputer.IsConfigured ? "Private Linux (WSL)" : "Private Linux (Hyper-V)");
            if (harness.RunsInComputer())
                Pair("Session", owner.HarnessSessions.GetValueOrDefault(harness) ?? "New (starts with the next message)", mono: true);
            if (owner == selected && usage is { } used)
                Pair("Last turn", $"{used.Input:N0} tokens in · {used.Output:N0} out");
        }
        else if (owner.Target == ConversationTarget.ThisPc)
        {
            Pair("Harness", "Claude Code · on this PC");
            Pair("Model", owner.Provider == ModelProvider.OpenRouter ? $"{owner.ModelId} · OpenRouter" : $"{owner.ModelId} · Ollama on this PC");
            Pair("Computer", "This PC · asks before each action");
            Pair("Session", owner.HarnessSessions.GetValueOrDefault(PrivateHarness.ClaudeCode) ?? "New (starts with the next message)", mono: true);
            if (owner == selected && usage is { } used)
                Pair("Last turn", $"{used.Input:N0} tokens in · {used.Output:N0} out");
        }
        else
        {
            Pair("Harness", "Sai");
            Pair("Model", $"{(owner.ModelId is null or "auto" ? "auto" : owner.ModelId)} · Sai API");
            Pair("Computer", CurrentComputer?.Name ?? "Unknown");
            Pair("Session", owner.SessionId ?? "New (starts with the next message)", mono: true);
        }
        var items = owner.Entries.Where(entry => entry.Role != "Activity" && !string.IsNullOrWhiteSpace(entry.Text)).ToArray();
        ConversationLabel.Text = $"CONVERSATION · {items.Length} {(items.Length == 1 ? "ITEM" : "ITEMS")}";
        if (items.Length == 0)
            ConversationSummary.Children.Add(new TextBlock { Text = "Nothing yet.", Opacity = 0.6 });
        foreach (var entry in items.TakeLast(6))
        {
            var block = new StackPanel { Spacing = 4 };
            var role = new TextBlock { Text = entry.Role == "Needs you" ? "Approval" : entry.Role, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, FontSize = 12 };
            if (entry.Role is "Needs you") role.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 240, 140, 30));
            if (entry.Role is "Error") role.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72));
            block.Children.Add(role);
            block.Children.Add(new TextBlock { Text = entry.Text, TextWrapping = TextWrapping.Wrap, MaxLines = 4, TextTrimming = TextTrimming.CharacterEllipsis });
            ConversationSummary.Children.Add(block);
        }
    }
    private Computer? CurrentComputer => selected?.Target == ConversationTarget.Sai
        ? computers.FirstOrDefault(c => c.Id == selected.ComputerId)
        : null;
    private void Show(string text, bool error = true)
    { Notice.Message = text; Notice.Severity = error ? InfoBarSeverity.Error : InfoBarSeverity.Informational; Notice.IsOpen = true; }
    private void Error(Exception error) => Show(Describe(error));
    private static string Describe(Exception error) => error is SaiException or PrivateComputerException or LocalAgentException ? error.Message :
        error is OperationCanceledException ? "The request timed out or was interrupted. Its remote result is not confirmed." :
        "The operation failed. Check your connection, credential access and local storage, then retry.";
    private async Task Persist()
    {
        if (!storageReady) return;
        try { await store.Save(conversations); }
        catch { Show("Could not save conversation history. Keep this window open to avoid losing the current conversation."); }
    }
    private ContentDialog Dialog(string title, object content, string primary) => new()
    { Title = title, Content = content, PrimaryButtonText = primary, CloseButtonText = "Cancel", XamlRoot = Root.XamlRoot };

    /// <summary>Reloads Sai computers and models; returns a user-facing error, or null on success.</summary>
    private async Task<string?> Refresh()
    {
        if (client is null) return null;
        busy = true; Controls();
        try
        {
            var nextComputers = await client.Computers();
            var nextModels = await client.Models();
            computers = nextComputers; models = nextModels.Where(m => m.Allowed).ToArray();
            Notice.IsOpen = false;
            return null;
        }
        catch (Exception error) { Error(error); return Describe(error); }
        finally { busy = false; Controls(); }
    }

    /// <summary>Saves (or removes, when null) the Sai key and drops everything tied to the previous account.</summary>
    private async Task<string?> UseSaiKey(string? key)
    {
        if (key is null) { Secrets.Delete(); client?.Dispose(); client = null; }
        else { Secrets.Save(key); client?.Dispose(); client = new SaiClient(key); }
        // A different credential must never reuse the prior account's sessions or live links.
        foreach (var conversation in conversations.Where(item => item.Target == ConversationTarget.Sai))
            conversation.SessionId = null;
        liveLinks.Clear(); computers = []; models = [];
        await Persist();
        return client is null ? null : await Refresh();
    }

    private async void RefreshComputers(object sender, RoutedEventArgs e)
    {
        if (!IsPrivate) { await Refresh(); await UpdateScreen(); return; }
        if (busy) return;
        var dialog = Dialog("Reset the Private computer?",
            "This erases guest changes since setup. Conversation history on Windows is kept.", "Reset computer");
        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        await UsePrivateComputer(async computer =>
        {
            await computer.Reset();
            PrivateScreen.Source = null; privateScreenShown = false;
            // Harness sessions lived in the guest that was just erased.
            foreach (var conversation in conversations.Where(c => c.Target == ConversationTarget.Private)) conversation.HarnessSessions.Clear();
            await Persist();
        });
    }

    private async void Accounts(object sender, RoutedEventArgs e)
    {
        if (busy) return;
        var password = new PasswordBox { Header = "Sai API key", PlaceholderText = "Paste a key from platform.simular.ai" };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(password, "SaiApiKey");
        var content = new StackPanel { Spacing = 12 };
        content.Children.Add(new TextBlock { Text = "The key is saved in Windows Credential Locker. Conversations stay on this Windows account.", TextWrapping = TextWrapping.Wrap });
        content.Children.Add(password);
        // OpenRouter, for This PC agents: saved or removed right here, separately from the Sai key below.
        var openRouterKey = new PasswordBox { Header = "OpenRouter API key", PlaceholderText = Secrets.ReadOpenRouter() is null ? "Paste a key from openrouter.ai" : "A key is saved. Paste a new one to replace it." };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(openRouterKey, "OpenRouterApiKey");
        var openRouterStatus = new TextBlock { TextWrapping = TextWrapping.Wrap, FontSize = 12, Opacity = 0.7,
            Text = "Used by This PC agents you set to OpenRouter. Agent S adds it to requests itself, so agents never see it." };
        var saveOpenRouter = new Button { Content = "Save OpenRouter key" };
        var removeOpenRouter = new Button { Content = "Remove", IsEnabled = Secrets.ReadOpenRouter() is not null };
        saveOpenRouter.Click += (_, _) =>
        {
            if (string.IsNullOrWhiteSpace(openRouterKey.Password)) return;
            try { Secrets.SaveOpenRouter(openRouterKey.Password.Trim()); openRouterStatus.Text = "OpenRouter key saved."; removeOpenRouter.IsEnabled = true; }
            catch (Exception failure) { openRouterStatus.Text = Describe(failure); }
            openRouterKey.Password = "";
        };
        removeOpenRouter.Click += (_, _) =>
        {
            try { Secrets.DeleteOpenRouter(); openRouterStatus.Text = "OpenRouter key removed."; removeOpenRouter.IsEnabled = false; }
            catch (Exception failure) { openRouterStatus.Text = Describe(failure); }
        };
        var openRouterButtons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        openRouterButtons.Children.Add(saveOpenRouter); openRouterButtons.Children.Add(removeOpenRouter);
        content.Children.Add(new MenuFlyoutSeparator());
        content.Children.Add(openRouterKey); content.Children.Add(openRouterButtons); content.Children.Add(openRouterStatus);
        var dialog = Dialog("Accounts", content, "Save Sai key");
        dialog.SecondaryButtonText = "Remove Sai key";
        var result = await dialog.ShowAsync();
        try
        {
            if (result == ContentDialogResult.Primary)
            {
                if (string.IsNullOrWhiteSpace(password.Password)) { Show("Enter a Sai API key."); return; }
                await UseSaiKey(password.Password.Trim());
            }
            else if (result == ContentDialogResult.Secondary) await UseSaiKey(null);
        }
        catch (Exception error) { Error(error); }
        finally { password.Password = ""; openRouterKey.Password = ""; Controls(); }
    }

    private async void NewAgent(object sender, RoutedEventArgs e)
    {
        if (busy || !storageReady) return;
        bool privateReady = PrivateComputer.IsConfigured, loadingCloud = false, setupRunning = false;
        var wslState = WslComputer.Readiness.Ready;
        var secondary = Application.Current.Resources.TryGetValue("TextFillColorSecondaryBrush", out var brush)
            ? brush as Microsoft.UI.Xaml.Media.Brush : null;
        // Modes in the order of the "Where should it work?" options.
        const int PrivateMode = 0, PcMode = 1, CloudMode = 2;
        static string Suggested(int mode) => mode switch { CloudMode => "Cloud agent", PcMode => "PC agent", _ => "Private agent" };
        static void Id(UIElement element, string id) => Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(element, id);
        TextBlock Note(string text = "")
        {
            var note = new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, FontSize = 12 };
            if (secondary is not null) note.Foreground = secondary;
            return note;
        }

        // Prefer whatever works on this PC right now: a set-up Private computer, else a connected Sai account.
        bool startCloud = !preferPrivate && !privateReady && client is not null;
        preferPrivate = false;
        var name = new TextBox { Header = "Name", Text = Suggested(startCloud ? CloudMode : PrivateMode), MaxLength = 100 };

        RadioButton Option(string id, string title, string detail)
        {
            var text = new StackPanel { Spacing = 2 };
            text.Children.Add(new TextBlock { Text = title, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
            text.Children.Add(Note(detail));
            var option = new RadioButton { Content = text };
            Id(option, id); Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(option, title);
            return option;
        }
        var privateOption = Option("TargetPrivate", "Private · Linux computer on this PC",
            "Works in a private Linux computer on this PC, with an AI model that runs locally.");
        var pcOption = Option("TargetThisPc", "This PC · your apps and files",
            "Works on this PC itself. It asks you before every action, and you can stop it any time.");
        var cloudOption = Option("TargetCloud", "Cloud · Sai computer",
            "Works on a computer in your Sai account. Nothing to install.");
        var target = new RadioButtons { Header = "Where should it work?" };
        target.Items.Add(privateOption); target.Items.Add(pcOption); target.Items.Add(cloudOption);
        target.SelectedIndex = startCloud ? CloudMode : PrivateMode;

        // ---- This PC: one checklist, one button ----
        bool ollamaReady = false, modelReady = false, restartPending = false, checkedPc = false;
        var green = new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 52, 199, 89));
        var red = new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72));
        var steps = new StackPanel { Spacing = 14 };
        (FontIcon Icon, ProgressRing Ring, TextBlock Detail) Step(string title)
        {
            var row = new Grid { ColumnSpacing = 12 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(20) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            var icon = new FontIcon { Glyph = "", FontSize = 16, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) };
            var ring = new ProgressRing { Width = 16, Height = 16, IsActive = false, Visibility = Visibility.Collapsed, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) };
            var text = new StackPanel();
            text.Children.Add(new TextBlock { Text = title, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
            var detail = Note("Checking…");
            text.Children.Add(detail);
            row.Children.Add(icon); row.Children.Add(ring);
            Grid.SetColumn(text, 1); row.Children.Add(text);
            steps.Children.Add(row);
            return (icon, ring, detail);
        }
        void Mark((FontIcon Icon, ProgressRing Ring, TextBlock Detail) step, string state, string detail)
        {
            step.Detail.Text = detail;
            step.Ring.IsActive = state == "working";
            step.Ring.Visibility = state == "working" ? Visibility.Visible : Visibility.Collapsed;
            step.Icon.Visibility = state == "working" ? Visibility.Collapsed : Visibility.Visible;
            step.Icon.Glyph = state switch { "done" => "", "failed" => "", _ => "" };
            step.Icon.ClearValue(IconElement.ForegroundProperty);
            if (state == "done") step.Icon.Foreground = green;
            if (state == "failed") step.Icon.Foreground = red;
        }
        var ollamaStep = Step("Ollama");
        var modelStep = Step("AI model");
        var linuxStep = Step("Private computer");
        var setupButton = new Button { Content = "Set up this PC", HorizontalAlignment = HorizontalAlignment.Stretch };
        if (Application.Current.Resources.TryGetValue("AccentButtonStyle", out var setupAccent) && setupAccent is Style setupStyle) setupButton.Style = setupStyle;
        Id(setupButton, "PrivateSetup");
        var setupProgress = new ProgressBar { IsIndeterminate = true, Visibility = Visibility.Collapsed };
        var checklist = new StackPanel { Spacing = 14 };
        checklist.Children.Add(new TextBlock { Text = "Get this PC ready", FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, FontSize = 15 });
        checklist.Children.Add(Note("A one-time setup. Agent S installs everything for you; it can take a while on the first run."));
        checklist.Children.Add(steps); checklist.Children.Add(setupProgress); checklist.Children.Add(setupButton);
        var setupCard = new Border
        {
            Child = checklist, CornerRadius = new CornerRadius(12), BorderThickness = new Thickness(1), Padding = new Thickness(18, 16, 18, 16),
            BorderBrush = Theme("CardBorder"), Background = Theme("ChipBackground"),
        };
        var localModel = new ComboBox
        {
            Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch, Visibility = Visibility.Collapsed,
            Description = "Runs on this PC through Ollama.",
        };
        Id(localModel, "PrivateModel");
        // Like macOS: Claude Code, Codex or DeepSeek run inside the Private computer; Agent S is the built-in loop.
        var harnessChoices = HarnessRunner.Supported || !privateReady ? Harnesses.All : [PrivateHarness.AgentS];
        var localHarness = new ComboBox
        {
            Header = "Harness", HorizontalAlignment = HorizontalAlignment.Stretch, Visibility = Visibility.Collapsed,
            ItemsSource = harnessChoices.Select(h => h.DisplayName()).ToArray(), SelectedIndex = 0,
            Description = "Runs inside the Private computer. Installed on first use.",
        };
        Id(localHarness, "PrivateHarness");
        var local = new StackPanel { Spacing = 12 };
        local.Children.Add(setupCard); local.Children.Add(localModel); local.Children.Add(localHarness);

        // ---- Cloud ----
        var key = new PasswordBox { Header = "Sai API key", PlaceholderText = "Paste your key", Description = "Saved securely in Windows Credential Locker." };
        Id(key, "NewAgentSaiKey");
        var connect = new Button { Content = "Connect" };
        if (Application.Current.Resources.TryGetValue("AccentButtonStyle", out var accent) && accent is Style accentStyle) connect.Style = accentStyle;
        var getKey = new HyperlinkButton { Content = "Get a key", NavigateUri = new Uri("https://platform.simular.ai/api-keys") };
        var keyActions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        keyActions.Children.Add(connect); keyActions.Children.Add(getKey);
        var keyPanel = new StackPanel { Spacing = 8 };
        keyPanel.Children.Add(Note("Connect your Sai account to use its computers."));
        keyPanel.Children.Add(key); keyPanel.Children.Add(keyActions);
        var computer = new ComboBox { Header = "Computer", PlaceholderText = "Choose a computer", HorizontalAlignment = HorizontalAlignment.Stretch };
        var model = new ComboBox { Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch, Description = "Automatic picks a model for each task." };
        var retry = new Button { Content = "Try again" };
        var cloudNotice = new InfoBar { IsClosable = false, Severity = InfoBarSeverity.Warning, ActionButton = retry };
        var loading = new ProgressBar { IsIndeterminate = true, Visibility = Visibility.Collapsed };
        var choices = new StackPanel { Spacing = 12 };
        choices.Children.Add(computer); choices.Children.Add(model);
        var cloud = new StackPanel { Spacing = 12 };
        cloud.Children.Add(keyPanel); cloud.Children.Add(loading); cloud.Children.Add(cloudNotice); cloud.Children.Add(choices);

        // ---- This PC: Claude Code on the PC itself, Ollama or OpenRouter ----
        var claudePath = ThisPcRunner.FindClaude();
        bool hasGitBash = ThisPcRunner.FindGitBash() is not null;
        string[] openRouterModels = ThisPcAgent.SuggestedOpenRouterModels;
        var pcNotice = new InfoBar { IsClosable = false, Severity = InfoBarSeverity.Warning };
        var pcInstall = new HyperlinkButton { Content = "How to install Claude Code", NavigateUri = ThisPcRunner.ClaudeSetup };
        var pcProvider = new ComboBox
        {
            Header = "Model provider", HorizontalAlignment = HorizontalAlignment.Stretch,
            ItemsSource = new[] { ModelProvider.Ollama.DisplayName(), ModelProvider.OpenRouter.DisplayName() }, SelectedIndex = 0,
        };
        Id(pcProvider, "PcProvider");
        var pcModel = new ComboBox { Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch, IsEditable = true };
        Id(pcModel, "PcModel");
        var openRouterKey = new PasswordBox { Header = "OpenRouter API key", PlaceholderText = "Paste your key", Description = "Saved in Windows Credential Locker. Agents never see it." };
        var saveOpenRouter = new Button { Content = "Save key" };
        var getOpenRouter = new HyperlinkButton { Content = "Get a key", NavigateUri = ThisPcRunner.OpenRouterKeys };
        var openRouterActions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        openRouterActions.Children.Add(saveOpenRouter); openRouterActions.Children.Add(getOpenRouter);
        var openRouterPanel = new StackPanel { Spacing = 8 };
        openRouterPanel.Children.Add(openRouterKey); openRouterPanel.Children.Add(openRouterActions);
        var desktop = new StackPanel { Spacing = 12 };
        desktop.Children.Add(Note("Harness: Claude Code, installed on this PC. Agent S uses its own Claude Code profile, and asks you before every command, click or file change. Looking (a screenshot or a window's contents) runs without asking."));
        desktop.Children.Add(pcNotice); desktop.Children.Add(pcProvider); desktop.Children.Add(openRouterPanel); desktop.Children.Add(pcModel);
        pcNotice.IsOpen = claudePath is null || !hasGitBash;
        pcNotice.Message = claudePath is null ? "Claude Code isn't installed on this PC." : "Claude Code on Windows needs Git for Windows (Git Bash).";
        if (claudePath is null) pcNotice.ActionButton = pcInstall;
        bool UseOpenRouter() => pcProvider.SelectedIndex == 1;
        string PcModelText() => (pcModel.SelectedItem as string ?? pcModel.Text ?? "").Trim();
        void ShowPcModels()
        {
            var keep = PcModelText();
            var items = UseOpenRouter() ? openRouterModels : localModel.ItemsSource as string[] ?? [];
            pcModel.ItemsSource = items;
            pcModel.SelectedItem = items.Contains(keep) ? keep : items.FirstOrDefault();
            if (pcModel.SelectedItem is null) pcModel.Text = "";
            pcModel.Description = UseOpenRouter() ? "Any model id from openrouter.ai/models. Prompts and what the agent reads go to OpenRouter and the model's provider." : "Runs on this PC through Ollama.";
        }

        var hint = Note();
        var form = new StackPanel { Spacing = 16, MinWidth = 420 };
        form.Children.Add(name); form.Children.Add(target); form.Children.Add(local); form.Children.Add(desktop); form.Children.Add(cloud); form.Children.Add(hint);
        var dialog = Dialog("New agent", new ScrollViewer { Content = form, VerticalScrollBarVisibility = ScrollBarVisibility.Auto }, "Create");
        dialog.DefaultButton = ContentDialogButton.Primary;

        int Mode() => Math.Max(0, target.SelectedIndex);
        bool UseCloud() => Mode() == CloudMode;
        void SetMode()
        {
            int mode = Mode();
            bool useCloud = mode == CloudMode;
            local.Visibility = mode == PrivateMode ? Visibility.Visible : Visibility.Collapsed;
            desktop.Visibility = mode == PcMode ? Visibility.Visible : Visibility.Collapsed;
            cloud.Visibility = useCloud ? Visibility.Visible : Visibility.Collapsed;
            keyPanel.Visibility = client is null ? Visibility.Visible : Visibility.Collapsed;
            choices.Visibility = client is not null && computers.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
            loading.Visibility = loadingCloud ? Visibility.Visible : Visibility.Collapsed;
            bool hasOpenRouterKey = Secrets.ReadOpenRouter() is not null;
            openRouterPanel.Visibility = UseOpenRouter() && !hasOpenRouterKey ? Visibility.Visible : Visibility.Collapsed;
            string? blocker = name.Text.Trim().Length == 0 ? "Give the agent a name." : mode switch
            {
                CloudMode => client is null ? "Connect your Sai account to continue."
                    : loadingCloud ? "Loading your computers…"
                    : computer.SelectedItem is not Computer ? "Choose a computer to continue." : null,
                PcMode => claudePath is null ? "Install Claude Code on this PC to continue."
                    : !hasGitBash ? "Install Git for Windows to continue."
                    : UseOpenRouter() && !hasOpenRouterKey ? "Add your OpenRouter key to continue."
                    : !UseOpenRouter() && !checkedPc ? "Checking Ollama…"
                    : !UseOpenRouter() && !ollamaReady ? "Start Ollama on this PC, or choose OpenRouter."
                    : PcModelText().Length == 0 ? "Choose a model to continue." : null,
                _ => setupRunning ? "Setting up this PC…"
                    : !checkedPc ? "Checking this PC…"
                    : !(privateReady && ollamaReady && modelReady) ? "Choose Set up this PC, or pick Cloud."
                    : localModel.SelectedItem is not string ? "Choose a model to continue." : null,
            };
            hint.Text = blocker ?? "";
            // The setup checklist already explains itself; don't repeat it underneath.
            bool checklistShown = mode == PrivateMode && setupCard.Visibility == Visibility.Visible && checkedPc;
            hint.Visibility = blocker is null || checklistShown ? Visibility.Collapsed : Visibility.Visible;
            dialog.IsPrimaryButtonEnabled = blocker is null;
        }
        void SetCloudChoices(string? error)
        {
            computer.ItemsSource = computers;
            computer.SelectedIndex = computers.Length > 0 ? Array.FindIndex(computers, c => c.Online) is var i and >= 0 ? i : 0 : -1;
            model.ItemsSource = new[] { new SaiModel("auto", "Automatic", true) }
                .Concat(models.Where(item => item.Id != "auto")).ToArray();
            model.SelectedIndex = 0;
            cloudNotice.Message = error ?? (client is not null && computers.Length == 0 ? "No computers were found in your Sai account." : "");
            cloudNotice.IsOpen = client is not null && cloudNotice.Message.Length > 0;
        }
        async Task LoadCloud(Func<Task<string?>> load)
        {
            loadingCloud = true; cloudNotice.IsOpen = false; SetMode();
            string? error;
            try { error = await load(); }
            catch (Exception failure) { error = Describe(failure); }
            loadingCloud = false; SetCloudChoices(error); SetMode();
        }
        void ShowModels(string[] installed)
        {
            localModel.ItemsSource = installed;
            localModel.SelectedItem = installed.FirstOrDefault(m => OllamaSetup.HasModel([m], OllamaSetup.DefaultModel)) ?? installed.FirstOrDefault();
        }
        void ShowReadiness()
        {
            bool allReady = privateReady && ollamaReady && modelReady;
            setupCard.Visibility = allReady ? Visibility.Collapsed : Visibility.Visible;
            localModel.Visibility = allReady ? Visibility.Visible : Visibility.Collapsed;
            localHarness.Visibility = localModel.Visibility;
            SetMode();
        }
        async Task CheckPc()
        {
            var installed = await OllamaSetup.Models();
            ollamaReady = installed is not null;
            modelReady = installed is not null && OllamaSetup.HasModel(installed, OllamaSetup.DefaultModel);
            if (installed is { Length: > 0 }) ShowModels(installed);
            Mark(ollamaStep, ollamaReady ? "done" : "todo", ollamaReady ? "Installed and running."
                : OllamaSetup.IsInstalled ? "Installed. Agent S will start it." : "Runs the AI model on this PC. Agent S installs it for you.");
            Mark(modelStep, modelReady ? "done" : "todo", modelReady ? $"{OllamaSetup.DefaultModel} is ready."
                : $"Downloads {OllamaSetup.DefaultModel}, a model that can see the screen (a few GB).");
            if (!privateReady) wslState = await WslComputer.CheckWsl();
            Mark(linuxStep, privateReady ? "done" : "todo", privateReady ? "Ready." : wslState switch
            {
                WslComputer.Readiness.NeedsRestart => "Windows needs a restart to finish turning on its virtual machine feature.",
                WslComputer.Readiness.NeedsWsl => "Turns on Windows' virtual machine feature, then creates a private desktop. Needs one restart.",
                _ => "Creates a private Linux desktop for the agent (about 400 MB).",
            });
            // When only the restart is left, the one button becomes "Restart now".
            restartPending = !privateReady && wslState == WslComputer.Readiness.NeedsRestart && ollamaReady && modelReady;
            setupButton.Content = restartPending ? "Restart now" : "Set up this PC";
            checkedPc = true;
            ShowReadiness();
        }
        async Task RunSetup()
        {
            if (restartPending)
            {
                // Explicit user choice; Windows shows its own restart notice.
                Process.Start(new ProcessStartInfo("shutdown.exe", "/r /t 10 /c \"Restarting to finish setting up Agent S\"") { UseShellExecute = false, CreateNoWindow = true });
                return;
            }
            setupRunning = busy = true; setupButton.IsEnabled = false; Controls(); SetMode();
            var current = ollamaStep;
            // Each step gets its own progress reporter, so a late update can't overwrite a finished step's ✓.
            IProgress<(string Step, double? Percent)> ProgressFor((FontIcon Icon, ProgressRing Ring, TextBlock Detail) step) =>
                new Progress<(string Step, double? Percent)>(update =>
                {
                    if (step.Icon.Glyph == "" && step.Icon.Visibility == Visibility.Visible) return;
                    Mark(step, "working", update.Step);
                    setupProgress.IsIndeterminate = update.Percent is null;
                    if (update.Percent is { } value) setupProgress.Value = value;
                });
            setupProgress.Visibility = Visibility.Visible;
            try
            {
                // 1. Ollama: install silently with winget, else open the download page.
                if (!ollamaReady)
                {
                    if (!OllamaSetup.IsInstalled)
                    {
                        Mark(current, "working", "Installing Ollama…");
                        if (!await OllamaSetup.Install(CancellationToken.None))
                        {
                            await Launcher.LaunchUriAsync(OllamaSetup.DownloadPage);
                            Mark(current, "failed", "Install Ollama from the page that just opened, then choose Set up this PC again.");
                            setupButton.Content = "Continue setup";
                            return;
                        }
                    }
                    Mark(current, "working", "Starting Ollama…");
                    if (!await OllamaSetup.Start(CancellationToken.None))
                    {
                        Mark(current, "failed", "Ollama is installed but didn't start. Open Ollama from the Start menu, then choose Set up this PC again.");
                        setupButton.Content = "Continue setup";
                        return;
                    }
                    ollamaReady = true;
                    Mark(current, "done", "Installed and running.");
                }

                // 2. The model, downloaded through Ollama with progress.
                current = modelStep;
                if (!modelReady)
                {
                    Mark(current, "working", $"Downloading {OllamaSetup.DefaultModel}…");
                    await OllamaSetup.Pull(OllamaSetup.DefaultModel, ProgressFor(modelStep), CancellationToken.None);
                    modelReady = true;
                    Mark(current, "done", $"{OllamaSetup.DefaultModel} is ready.");
                }
                ShowModels(await OllamaSetup.Models() ?? [OllamaSetup.DefaultModel]);

                // 3. The private Linux computer (turning on WSL first if needed).
                current = linuxStep;
                if (!privateReady)
                {
                    wslState = await WslComputer.CheckWsl();
                    if (wslState == WslComputer.Readiness.NeedsWsl)
                    {
                        Mark(current, "working", "Turning on Windows' virtual machine feature. Approve the Windows prompt…");
                        await WslComputer.InstallWsl();
                        wslState = await WslComputer.CheckWsl();
                    }
                    if (wslState != WslComputer.Readiness.Ready)
                    {
                        Mark(current, "todo", "Restart your PC to finish turning on Windows' virtual machine feature. Afterwards, open Agent S and choose Set up this PC again.");
                        restartPending = true;
                        setupButton.Content = "Restart now";
                        return;
                    }
                    setupProgress.Value = 0;
                    Mark(current, "working", "Creating the private computer…");
                    await WslComputer.Setup(ProgressFor(linuxStep), CancellationToken.None);
                    privateReady = true;
                    Mark(current, "done", "Ready.");
                }
            }
            catch (Exception failure)
            {
                Mark(current, "failed", Describe(failure));
                setupButton.Content = "Try again";
            }
            finally
            {
                setupRunning = busy = false; setupButton.IsEnabled = true;
                setupProgress.Visibility = Visibility.Collapsed; Controls(); ShowReadiness();
            }
        }

        target.SelectionChanged += async (_, _) =>
        {
            if (name.Text.Trim() is "" or "Private agent" or "PC agent" or "Cloud agent") name.Text = Suggested(Mode());
            if (Mode() == PcMode) ShowPcModels();
            SetMode();
            if (UseCloud() && client is not null && computers.Length == 0 && !loadingCloud) await LoadCloud(Refresh);
        };
        computer.SelectionChanged += (_, _) => SetMode();
        localModel.SelectionChanged += (_, _) => SetMode();
        name.TextChanged += (_, _) => SetMode();
        pcProvider.SelectionChanged += async (_, _) =>
        {
            ShowPcModels(); SetMode();
            // The full list loads in the background; the suggestions show at once.
            if (UseOpenRouter() && openRouterModels == ThisPcAgent.SuggestedOpenRouterModels)
            {
                var all = await OpenRouterProxy.Models();
                if (all.Length > 0) { openRouterModels = ThisPcAgent.SuggestedOpenRouterModels.Concat(all.Except(ThisPcAgent.SuggestedOpenRouterModels)).ToArray(); if (UseOpenRouter()) ShowPcModels(); }
            }
        };
        pcModel.SelectionChanged += (_, _) => SetMode();
        pcModel.TextSubmitted += (_, _) => SetMode();
        pcModel.LostFocus += (_, _) => SetMode();
        saveOpenRouter.Click += (_, _) =>
        {
            var value = openRouterKey.Password.Trim();
            if (value.Length == 0) return;
            try { Secrets.SaveOpenRouter(value); } catch (Exception failure) { Error(failure); }
            openRouterKey.Password = ""; SetMode();
        };
        key.KeyDown += async (_, args) => { if (args.Key == VirtualKey.Enter) { args.Handled = true; await Connect(); } };
        connect.Click += async (_, _) => await Connect();
        retry.Click += async (_, _) => await LoadCloud(Refresh);
        setupButton.Click += async (_, _) => await RunSetup();
        dialog.Closing += (_, args) => { if (setupRunning) args.Cancel = true; };
        async Task Connect()
        {
            var value = key.Password.Trim();
            if (value.Length == 0 || loadingCloud) return;
            connect.IsEnabled = false;
            try { await LoadCloud(() => UseSaiKey(value)); }
            finally { key.Password = ""; connect.IsEnabled = true; }
        }

        SetCloudChoices(null);
        SetMode();
        _ = CheckThenShowPcModels();
        async Task CheckThenShowPcModels()
        {
            await CheckPc();
            ShowPcModels(); SetMode();
        }
        if (startCloud && computers.Length == 0) _ = LoadCloud(Refresh);

        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        var conversation = new Conversation { Name = name.Text.Trim() };
        if (Mode() == PcMode)
        {
            if (PcModelText() is not { Length: > 0 } pcChoice) return;
            conversation.Target = ConversationTarget.ThisPc;
            conversation.Harness = PrivateHarness.ClaudeCode;
            conversation.Provider = UseOpenRouter() ? ModelProvider.OpenRouter : ModelProvider.Ollama;
            conversation.ModelId = pcChoice;
        }
        else if (!UseCloud())
        {
            if (localModel.SelectedItem is not string localChoice) return;
            conversation.Target = ConversationTarget.Private;
            conversation.ModelId = localChoice;
            var harness = harnessChoices[Math.Max(0, localHarness.SelectedIndex)];
            conversation.Harness = harness.RunsInComputer() && !HarnessRunner.Supported ? PrivateHarness.AgentS : harness;
        }
        else
        {
            if (computer.SelectedItem is not Computer choice) return;
            conversation.Target = ConversationTarget.Sai;
            conversation.ComputerId = choice.Id;
            conversation.ModelId = (model.SelectedItem as SaiModel)?.Id;
        }
        conversations.Add(conversation); Select(conversation); await Persist();
    }
    private async void SelectAgent(object sender, SelectionChangedEventArgs e)
    {
        if (rebuilding) return;
        var next = (Agents.SelectedItem as AgentRow)?.Conversation;
        if (next is null && selected is not null && conversations.Contains(selected)) { Agents.SelectedItem = Row(selected); return; }
        selected = next; usage = null;
        Composer.Text = ""; Messages.Children.Clear(); entryText.Clear(); approvalControls.Clear();
        if (selected is not null) foreach (var entry in selected.Entries) Draw(entry);
        Controls();
        DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
        if (!busy && selected?.Target == ConversationTarget.Sai && client is not null && computers.Length == 0)
            await Refresh();
        await UpdateScreen();
    }

    private static void SetButtonsEnabled(StackPanel panel, bool enabled)
    {
        foreach (var control in panel.Children.OfType<Control>()) control.IsEnabled = enabled;
    }

    private Microsoft.UI.Xaml.Media.Brush Theme(string key) =>
        (Microsoft.UI.Xaml.Media.Brush)((ResourceDictionary)Application.Current.Resources.ThemeDictionaries[
            Root.ActualTheme == ElementTheme.Dark ? "Dark" : "Light"])[key];

    private void Draw(ChatEntry entry)
    {
        var text = new TextBlock { Text = entry.Text, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true, FontSize = 16, LineHeight = 24 };
        entryText[entry.Id] = text;
        FrameworkElement element;
        StackPanel? panel = null;
        switch (entry.Role)
        {
            case "You":
                // Black bubble on the right, as on macOS.
                text.Foreground = Theme("BubbleForeground");
                element = new Border
                {
                    Child = text, Background = Theme("BubbleBackground"), CornerRadius = new CornerRadius(22),
                    Padding = new Thickness(18, 10, 18, 10), HorizontalAlignment = HorizontalAlignment.Right, MaxWidth = 560,
                };
                break;
            case "Activity":
            {
                var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10 };
                row.Children.Add(new FontIcon { Glyph = "", FontSize = 14, Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 52, 199, 89)), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 4, 0, 0) });
                text.FontSize = 14; text.Foreground = Theme("MutedText"); text.MaxLines = 3; text.TextTrimming = TextTrimming.CharacterEllipsis;
                row.Children.Add(text);
                element = new Border
                {
                    Child = row, BorderBrush = Theme("CardBorder"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(20),
                    Padding = new Thickness(16, 8, 18, 8), HorizontalAlignment = HorizontalAlignment.Left, MaxWidth = 560,
                };
                break;
            }
            case "Error":
                text.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72));
                element = text;
                break;
            case "Needs you":
            {
                panel = new StackPanel { Spacing = 12 };
                var head = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10 };
                head.Children.Add(new FontIcon { Glyph = "", FontSize = 16, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 3, 0, 0) });
                head.Children.Add(text);
                panel.Children.Add(head);
                element = new Border
                {
                    Child = panel, BorderBrush = Theme("CardBorder"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(18),
                    Background = Theme("CardBackground"), Padding = new Thickness(18, 14, 18, 14), HorizontalAlignment = HorizontalAlignment.Left, MaxWidth = 600,
                };
                break;
            }
            default:
                element = text; // Agent replies are plain text, left aligned.
                break;
        }
        if (panel is not null && entry.LocalAnswer is { } localAnswer && !entry.Answered)
        {
            // This PC: the exact step is shown above; nothing runs until the person chooses.
            var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            void Answer(bool allowed, string note, bool rest = false)
            {
                if (entry.Answered) return;
                entry.Answered = true;
                if (rest) approveRestOfTurn = true;
                localAnswer(allowed);
                SetButtonsEnabled(buttons, false);
                text.Text = entry.Text += "\n" + note;
            }
            var allow = new Button { Content = "Allow" };
            if (Application.Current.Resources.TryGetValue("AccentButtonStyle", out var accentStyle) && accentStyle is Style style) allow.Style = style;
            allow.Click += (_, _) => Answer(true, "Allowed.");
            var allowRest = new Button { Content = "Allow for this task" };
            allowRest.Click += (_, _) => Answer(true, "Allowed, with the rest of this task.", rest: true);
            var deny = new Button { Content = "Deny" };
            deny.Click += (_, _) => Answer(false, "Denied.");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(allow, "ApproveStep");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(deny, "DenyStep");
            buttons.Children.Add(allow); buttons.Children.Add(allowRest); buttons.Children.Add(deny);
            panel.Children.Add(buttons);
            approvalControls[entry.Id] = buttons;
        }
        else if (panel is not null && entry.ApprovalId is not null && !entry.Answered)
        {
            var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            if (entry.LinkOnly)
            {
                if (entry.ApprovalUrl is { } url)
                {
                    var open = new Button { Content = "Open approval" };
                    open.Click += async (_, _) => { if (!await Launcher.LaunchUriAsync(url)) Show("Could not open the approval in your browser."); };
                    buttons.Children.Add(open);
                }
                else panel.Children.Add(new TextBlock { Text = "No valid approval link was supplied.", TextWrapping = TextWrapping.Wrap });
            }
            else
            {
                foreach (var (label, answer) in new[] { ("Approve", "approve"), ("Deny", "deny") })
                {
                    var button = new Button { Content = label };
                    button.Click += async (_, _) =>
                    {
                        SetButtonsEnabled(buttons, false);
                        try
                        {
                            if (client is null) throw new SaiException("Connect your Sai account first.");
                            await client.Approve(entry.ApprovalId, answer);
                            entry.Answered = true; text.Text = entry.Text += $"\n{label} confirmed.";
                        }
                        catch (Exception error)
                        {
                            if (!entry.Answered && running)
                            { Error(error); SetButtonsEnabled(buttons, true); }
                        }
                    };
                    buttons.Children.Add(button);
                }
            }
            panel.Children.Add(buttons);
            approvalControls[entry.Id] = buttons;
        }
        var following = Transcript.ScrollableHeight - Transcript.VerticalOffset < 64;
        Messages.Children.Add(element);
        if (following) DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
    }
    private void Add(Conversation owner, ChatEntry entry)
    {
        owner.Entries.Add(entry);
        Row(owner).Refresh();
        if (selected == owner) { Draw(entry); EmptyChat.Visibility = Visibility.Collapsed; UpdateContext(); }
    }
    private void Change(ChatEntry entry)
    {
        var following = Transcript.ScrollableHeight - Transcript.VerticalOffset < 64;
        if (selected is not null) Row(selected).Refresh();
        if (entryText.TryGetValue(entry.Id, out var text)) text.Text = entry.Text;
        if (following) DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
    }
    private static string? Str(JsonElement value, string field) => value.ValueKind == JsonValueKind.Object && value.TryGetProperty(field, out var found) && found.ValueKind == JsonValueKind.String ? found.GetString() : null;

    private async Task Apply(Conversation owner, JsonElement value)
    {
        var type = Str(value, "type");
        var data = value.TryGetProperty("data", out var d) ? d : default;
        switch (type)
        {
            case "data-session":
                owner.SessionId = currentSession = Str(data, "sessionId");
                if (stopRequested) await Stop(owner);
                break;
            case "text-delta":
                var id = turn + "/text/" + Str(value, "id");
                var text = owner.Entries.FirstOrDefault(entry => entry.Id == id);
                if (text is null) { text = new ChatEntry { Id = id }; Add(owner, text); }
                text.Text += Str(value, "delta"); Change(text);
                break;
            case "tool-input-start": case "tool-input-available":
                var toolId = turn + "/tool/" + Str(value, "toolCallId");
                var tool = owner.Entries.FirstOrDefault(entry => entry.Id == toolId);
                if (tool is null) { tool = new ChatEntry { Id = toolId, Role = "Activity" }; Add(owner, tool); }
                tool.Text = value.TryGetProperty("input", out var input) ? Str(input, "title") ?? Str(value, "toolName") ?? "Working" : Str(value, "toolName") ?? "Working";
                Change(tool); break;
            case "data-progress":
                var progress = owner.Entries.LastOrDefault(entry => entry.Role == "Activity");
                if (progress is not null) { progress.Text += "\n" + Str(data, "text"); Change(progress); }
                break;
            case "tool-output-error":
                Add(owner, new ChatEntry { Role = "Error", Text = Str(value, "errorText") ?? "An action failed." }); break;
            case "data-approval-request":
                var approvalId = Str(data, "approvalId");
                if (approvalId is null || owner.Entries.Any(entry => entry.ApprovalId == approvalId)) break;
                Add(owner, new ChatEntry { Role = "Needs you", ApprovalId = approvalId,
                    Text = string.Join("\n", new[] { Str(data, "title"), Str(data, "description") }.Where(s => !string.IsNullOrWhiteSpace(s))),
                    LinkOnly = data.TryGetProperty("isLinkOnly", out var link) && link.ValueKind == JsonValueKind.True,
                    ApprovalUrl = SaiClient.SafeWebLink(Str(data, "approvalUrl")) }); break;
            case "error": throw new SaiException("Sai reported a turn error. Check the task in Sai before retrying.");
        }
    }

    private async void Send(object sender, RoutedEventArgs e)
    {
        if (busy || selected is null || string.IsNullOrWhiteSpace(Composer.Text)) return;
        var owner = selected;
        bool local = owner.Target == ConversationTarget.Private;
        bool pc = owner.Target == ConversationTarget.ThisPc;
        if (!local && !pc && (owner.Target != ConversationTarget.Sai || client is null)) return;
        if (!local && !pc && CurrentComputer is null) { Show("The saved computer is not available on this account. Refresh computers or create a new agent."); return; }
        if (local && !PrivateComputer.IsConfigured)
        {
            Show("This PC's Private computer isn't set up yet. Choose New agent › This PC › Set up.");
            return;
        }
        var history = owner.Entries.Where(entry => entry.Role is "You" or "Agent").TakeLast(12)
            .Select(entry => new PrivateMessage(entry.Role == "You" ? "user" : "assistant", entry.Text)).ToArray();
        var message = Composer.Text.Trim(); Composer.Text = "";
        busy = running = true; stopped = stopRequested = false; Notice.IsOpen = false;
        currentSession = null;
        var cancellation = new CancellationTokenSource();
        turn = Guid.NewGuid().ToString("N"); run = cancellation; Controls();
        Add(owner, new ChatEntry { Role = "You", Text = message });
        DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
        try
        {
            await Persist();
            if (pc)
            {
                await RunThisPc(owner, message, cancellation.Token);
            }
            else if (local && (owner.Harness ?? PrivateHarness.AgentS) is var harness && harness.RunsInComputer())
            {
                if (!panelOpen) TogglePanel(this, new RoutedEventArgs());
                await RunHarness(owner, harness, message, cancellation.Token);
            }
            else if (local)
            {
                var computer = new PrivateComputer();
                using var agent = new LocalAgent();
                if (!panelOpen) TogglePanel(this, new RoutedEventArgs());
                Controls();
                await computer.Start(cancellation.Token);
                await foreach (var progress in agent.Run(message, owner.ModelId ?? "gemma4", history, computer, cancellation.Token))
                {
                    if (progress.Screenshot is { } screenshot) await ShowPrivateScreen(screenshot);
                    if (!string.IsNullOrWhiteSpace(progress.Text))
                        Add(owner, new ChatEntry { Role = progress.Role, Text = progress.Text });
                }
            }
            else
            {
                await foreach (var value in client!.Run(message, owner.ComputerId, owner.SessionId, owner.ModelId, cancellation.Token))
                    await Apply(owner, value);
                if (stopRequested && !stopped) Show("The stream ended without confirming that the remote task stopped.");
            }
        }
        catch (OperationCanceledException) when (stopped)
        { Show(local ? "Private turn stopped." : pc ? "Stopped. Nothing else runs on this PC." : "Sai confirmed the stop request.", false); }
        catch (Exception error) { Error(error); }
        finally
        {
            // A This PC step still waiting when the turn ends is denied, never left to run later.
            foreach (var entry in owner.Entries.Where(entry => entry.LocalAnswer is not null && !entry.Answered))
            {
                entry.Answered = true; entry.LocalAnswer!(false); entry.Text += "\nNot run: the turn ended."; Change(entry);
            }
            pcTurnOwner = null; approveRestOfTurn = false;
            foreach (var entry in owner.Entries.Where(entry => entry.ApprovalId is not null && !entry.Answered))
            {
                entry.Answered = true; entry.Text += "\nTurn ended. Check pending approvals in Sai."; Change(entry);
            }
            foreach (var controls in approvalControls.Values) SetButtonsEnabled(controls, false);
            run = null; running = false; cancellation.Dispose(); Controls();
            await Persist();
            busy = false; Controls();
        }
    }

    /// <summary>One Claude Code turn on this PC itself, every action approved by the person, like "This Mac".</summary>
    private async Task RunThisPc(Conversation owner, string message, CancellationToken token)
    {
        ThisPcRunner.UseApprovals(Decide);
        pcTurnOwner = owner; approveRestOfTurn = false;
        var model = owner.ModelId ?? "gemma4";
        Add(owner, new ChatEntry { Role = "Activity", Text = $"Claude Code · {(owner.Provider == ModelProvider.OpenRouter ? "OpenRouter · " : "")}{model}" });
        var calls = new Dictionary<string, (ChatEntry Entry, bool Screenshot)>();
        var resume = owner.HarnessSessions.GetValueOrDefault(PrivateHarness.ClaudeCode);
        bool started = false, reportedError = false;
        try
        {
            await foreach (var item in ThisPcRunner.Run(owner, message, resume, model, token))
            {
                switch (item)
                {
                    case HarnessStarted begun:
                        started = true; owner.HarnessSessions[PrivateHarness.ClaudeCode] = begun.SessionId; UpdateContext(); break;
                    case HarnessText said:
                        Add(owner, new ChatEntry { Role = "Agent", Text = said.Text }); break;
                    case HarnessToolCall call:
                        // "pc apps" reads better than the tool's full path.
                        var activity = new ChatEntry { Role = "Activity", Text = ShortPc(call.Summary) };
                        calls[call.Id] = (activity, call.Summary.EndsWith(" screenshot", StringComparison.Ordinal));
                        Add(owner, activity); break;
                    case HarnessToolResult result when calls.TryGetValue(result.Id, out var done):
                        if (result.IsError)
                        {
                            var first = result.Output.Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "";
                            done.Entry.Text += $"\n{(first.Contains("denied", StringComparison.OrdinalIgnoreCase) ? "Denied" : "Failed")}{(first.Length > 0 ? ": " + first : "")}";
                            Change(done.Entry);
                        }
                        else if (done.Screenshot && File.Exists(ThisPcRunner.ScreenshotPath))
                        {
                            try { await ShowPrivateScreen(await File.ReadAllBytesAsync(ThisPcRunner.ScreenshotPath, token), fromPc: true); } catch (IOException) { }
                        }
                        break;
                    case HarnessUsage used:
                        usage = used; UpdateContext(); break;
                    case HarnessFinished { IsError: true } end when !reportedError:
                        reportedError = true;
                        Add(owner, new ChatEntry { Role = "Error", Text = Readable(end.Result) ?? "Claude Code reported an error." }); break;
                }
            }
        }
        catch (PrivateComputerException) when (!started && resume is not null)
        {
            owner.HarnessSessions.Remove(PrivateHarness.ClaudeCode);
            throw new PrivateComputerException("Claude Code couldn't continue its earlier session, so the next message starts a new one. Send it again.");
        }
    }

    private static string ShortPc(string text) => text.Replace(ThisPcAgent.ToolCommand(ThisPcRunner.PcToolPath) + " ", "pc ", StringComparison.Ordinal);

    /// <summary>The This PC agent whose turn is running; approval requests for any other agent are denied.</summary>
    private Conversation? pcTurnOwner;

    /// <summary>Called by the approval hook (off the UI thread) for each This PC tool use.</summary>
    private Task<bool> Decide(ApprovalRequest request, CancellationToken cancellation)
    {
        var answer = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        cancellation.Register(() => answer.TrySetResult(false));
        if (!DispatcherQueue.TryEnqueue(() =>
        {
            var owner = pcTurnOwner;
            if (owner is null || owner.Id != request.Agent || !running) { answer.TrySetResult(false); return; }
            // Looking runs without asking; so does the rest of a turn the person approved as a whole.
            if (ThisPcAgent.IsLookOnly(request.Tool, request.Input, ThisPcRunner.PcToolPath) || approveRestOfTurn) { answer.TrySetResult(true); return; }
            var entry = new ChatEntry { Role = "Needs you", Text = ShortPc(ThisPcAgent.Describe(request.Tool, request.Input)) };
            entry.LocalAnswer = allowed => answer.TrySetResult(allowed);
            Add(owner, entry);
        })) answer.TrySetResult(false);
        return answer.Task;
    }

    /// <summary>One Claude Code, Codex or DeepSeek turn inside the Private computer, as on macOS.</summary>
    private async Task RunHarness(Conversation owner, PrivateHarness harness, string message, CancellationToken token)
    {
        if (!HarnessRunner.Supported)
            throw new PrivateComputerException($"{harness.DisplayName()} needs the WSL Private computer. Choose Agent S (built-in) in the model menu.");
        var model = owner.ModelId ?? "gemma4";
        var computer = new PrivateComputer();
        var status = new ChatEntry { Role = "Activity", Text = "Starting the Private computer…" };
        Add(owner, status);
        await computer.Start(token);
        try { await ShowPrivateScreen(await computer.Capture(token)); } catch (PrivateComputerException) { }
        status.Text = $"Getting {harness.DisplayName()} ready…"; Change(status);
        await HarnessRunner.EnsureInstalled(new Progress<string>(line =>
        {
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("Harnesses are installed", StringComparison.Ordinal)) return;
            status.Text = $"First use: {line.Trim()} (this takes a few minutes)"; Change(status);
        }), token);
        await HarnessRunner.EnsureRelay(token);
        var runModel = await HarnessRunner.ContextModel(model, token);
        status.Text = $"{harness.DisplayName()} · {model}"; Change(status);

        using var screens = CancellationTokenSource.CreateLinkedTokenSource(token);
        var watching = WatchScreen(computer, screens.Token);
        var calls = new Dictionary<string, ChatEntry>();
        var resume = owner.HarnessSessions.GetValueOrDefault(harness);
        bool started = false, reportedError = false;
        try
        {
            await foreach (var item in HarnessRunner.Run(harness, message, resume, runModel, token))
            {
                switch (item)
                {
                    case HarnessStarted begun:
                        started = true; owner.HarnessSessions[harness] = begun.SessionId; UpdateContext(); break;
                    case HarnessText said:
                        Add(owner, new ChatEntry { Role = "Agent", Text = said.Text }); break;
                    case HarnessToolCall call:
                        var activity = new ChatEntry { Role = "Activity", Text = call.Summary };
                        calls[call.Id] = activity; Add(owner, activity); break;
                    case HarnessToolResult { IsError: true } result when calls.TryGetValue(result.Id, out var failed):
                        var first = result.Output.Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "";
                        failed.Text += $"\nFailed{(first.Length > 0 ? ": " + first : "")}"; Change(failed); break;
                    case HarnessUsage used:
                        usage = used; UpdateContext(); break;
                    case HarnessFinished { IsError: true } end when !reportedError:
                        // Codex reports a failure twice (error, then turn.failed); show it once.
                        reportedError = true;
                        Add(owner, new ChatEntry { Role = "Error", Text = Readable(end.Result) ?? $"{harness.DisplayName()} reported an error." }); break;
                }
            }
        }
        catch (PrivateComputerException) when (!started && resume is not null)
        {
            // A session from before Reset computer (or another install) can't be resumed; start fresh next time.
            owner.HarnessSessions.Remove(harness);
            throw new PrivateComputerException($"{harness.DisplayName()} couldn't continue its earlier session, so the next message starts a new one. Send it again.");
        }
        finally
        {
            screens.Cancel();
            try { await watching; } catch (OperationCanceledException) { }
        }
        try { await ShowPrivateScreen(await computer.Capture(CancellationToken.None)); } catch (Exception) { }
    }

    /// <summary>A harness error as words: model servers often return {"error":{"message":…}} as the text.</summary>
    private static string? Readable(string? result)
    {
        if (string.IsNullOrWhiteSpace(result)) return null;
        try
        {
            using var document = JsonDocument.Parse(result);
            var root = document.RootElement;
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("error", out var error))
                return Str(error, "message") ?? (error.ValueKind == JsonValueKind.String ? error.GetString() : null) ?? result.Trim();
        }
        catch (JsonException) { }
        return result.Trim();
    }

    /// <summary>Refreshes the Private computer's screen while a harness works, so the person can watch.</summary>
    private async Task WatchScreen(PrivateComputer computer, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(2500, token);
                var png = await computer.Capture(token);
                if (IsPrivate) await ShowPrivateScreen(png);
            }
            catch (OperationCanceledException) { return; }
            catch (Exception) { /* A missed frame is fine; the next one follows. */ }
        }
    }

    private async Task Stop(Conversation owner)
    {
        var cancellation = run;
        if (cancellation is null || !running || selected != owner) return;
        if (owner.Target is ConversationTarget.Private or ConversationTarget.ThisPc)
        {
            stopped = true;
            cancellation.Cancel();
            return;
        }
        if (client is null || currentSession is null) return;
        try
        {
            await client.Abort(currentSession);
            if (run != cancellation) return;
            stopped = true; cancellation.Cancel();
        }
        catch (Exception error) when (run == cancellation) { stopRequested = false; Error(error); Controls(); }
        catch (Exception) { /* The matching turn already ended. */ }
    }
    private async void StopTurn(object sender, RoutedEventArgs e)
    {
        if (!running || selected is null) return;
        stopRequested = true; Controls();
        if (!IsPrivate && !IsThisPc && currentSession is null) Show("Waiting for the session ID before sending the stop request…", false);
        else await Stop(selected);
    }
    private async void OpenComputer(object sender, RoutedEventArgs e)
    {
        if (IsThisPc) return;
        if (IsPrivate)
        {
            if (!busy) await StartPrivateScreen();
            return;
        }
        if (busy || CurrentComputer is not { } computer || client is null) return;
        busy = true; Controls();
        try
        {
            if (!liveLinks.TryGetValue(computer.Id, out var cached) || cached.Until < DateTime.UtcNow)
            { cached = (await client.LiveLink(computer.Id), DateTime.UtcNow.AddMinutes(23)); liveLinks[computer.Id] = cached; }
            if (!await Launcher.LaunchUriAsync(cached.Uri)) Show("Could not open the computer in your browser.");
        }
        catch (Exception error) { Error(error); }
        finally { busy = false; Controls(); }
    }
    private async void WakeComputer(object sender, RoutedEventArgs e)
    {
        if (IsPrivate)
        {
            await UsePrivateComputer(async computer =>
            {
                await computer.Stop();
                PrivateScreen.Source = null; privateScreenShown = false;
            });
            return;
        }
        if (busy || CurrentComputer is not { CanWake: true } computer || client is null) return;
        busy = true; Controls();
        try { await client.Wake(computer.Id); Show("Sai is waking the computer. Choose ⋯ › Refresh computers in a moment.", false); }
        catch (Exception error) { Error(error); }
        finally { busy = false; Controls(); }
    }
    private async Task UsePrivateComputer(Func<PrivateComputer, Task> use)
    {
        if (busy) return;
        busy = true; Controls();
        try { await use(new PrivateComputer()); Notice.IsOpen = false; }
        catch (Exception error) { Error(error); }
        finally { busy = false; Controls(); }
    }

    private async Task ShowPrivateScreen(byte[] png, bool fromPc = false)
    {
        screenFromPc = fromPc;
        using var stream = new InMemoryRandomAccessStream();
        using (var writer = new DataWriter(stream))
        {
            writer.WriteBytes(png);
            await writer.StoreAsync();
            writer.DetachStream();
        }
        stream.Seek(0);
        var image = new BitmapImage();
        await image.SetSourceAsync(stream);
        PrivateScreen.Source = image;
        if (!fromPc) privateScreenShown = true;
        Controls();
    }

    private async void Closing(AppWindow sender, AppWindowClosingEventArgs args)
    {
        if (allowClose || !running || selected is null) return;
        args.Cancel = true;
        if (closing) return;
        closing = true;
        bool local = IsPrivate || IsThisPc;
        var hasSession = currentSession is not null;
        var pending = run;
        var dialog = local
            ? Dialog("A Private task is running", "Stop the task before closing? The guest stays running.", "Stop and close")
            : Dialog("A remote task is running", hasSession ? "Stop it before closing?" : "The session ID has not arrived. Closing disconnects this app; the remote task may continue. Check it in Sai.",
                hasSession ? "Stop and close" : "Close anyway");
        try
        {
            if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
            if ((local || hasSession) && pending == run)
            {
                if (selected is null) return;
                await Stop(selected);
                if (!stopped && pending == run) return;
            }
            await Persist(); allowClose = true; Close();
        }
        finally { closing = false; }
    }
}
