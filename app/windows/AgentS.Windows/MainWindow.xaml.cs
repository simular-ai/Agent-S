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
    /// <summary>Whose screen the panel's image shows: the Private computer, This PC (a pc screenshot) or a Box, by key.</summary>
    private string screenKind = "private";
    private bool IsBox => selected?.Target == ConversationTarget.Box;
    /// <summary>The selected Box's state as shown ("Running", "Asleep"), or null before it's known.</summary>
    private string? boxState;
    /// <summary>The key the panel's image needs for the selected agent; a different agent's screen is never shown.</summary>
    private string? ScreenKindFor(Conversation? owner) => owner?.Target switch
    {
        ConversationTarget.Private => "private",
        ConversationTarget.ThisPc => "pc",
        ConversationTarget.Box => "box:" + (owner.BoxName ?? ""),
        _ => null,
    };
    /// <summary>This PC: approve the rest of the current turn's actions without asking (reset every turn).</summary>
    private bool approveRestOfTurn;
    private Uri? liveViewUri;

    public MainWindow()
    {
        InitializeComponent();
        Title = AppIdentity.Name;
        AppTitle.Text = AppIdentity.Name;
        conversations.CollectionChanged += (_, _) => RebuildAgents();
        var logo = Path.Combine(AppContext.BaseDirectory, "Assets", AppIdentity.LogoFile);
        if (File.Exists(logo)) EmptyChatLogo.Source = new BitmapImage(new Uri(logo));
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        var width = Math.Min(1360, area.Width);
        var height = Math.Min(860, area.Height);
        AppWindow.Resize(new global::Windows.Graphics.SizeInt32(width, height));
        AppWindow.Move(new global::Windows.Graphics.PointInt32(area.X + (area.Width - width) / 2,
            area.Y + (area.Height - height) / 2));
        var icon = Path.Combine(AppContext.BaseDirectory, "Assets", AppIdentity.IconFile);
        if (File.Exists(icon)) AppWindow.SetIcon(icon);
        AppWindow.Closing += Closing;
        InitializeComputerView();
        InitializeChatExtras();
        ModelBrowser.Agents = () => conversations;
        Closed += (_, _) => { updateTimer?.Stop(); run?.Cancel(); client?.Dispose(); };
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            foreach (var conversation in await store.Read()) { UpgradeSteps(conversation); conversations.Add(conversation); }
            storageReady = true;
            var key = Secrets.Read();
            if (key is not null) client = new SaiClient(key);
            Select(conversations.FirstOrDefault(item => item.Target == ConversationTarget.Private) ?? conversations.FirstOrDefault());
            // Existing installs skip first-launch setup; a fresh one starts there, as on macOS.
            if (ShowsOnboarding) ShowOnboarding();
            else if (!AppSettings.Current.FinishedFirstAgentSetup) { AppSettings.Current.FinishedFirstAgentSetup = true; AppSettings.Current.Save(); }
        }
        catch (Exception error) { Error(error); }
        Controls();
        StartUpdateChecks();
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
        // All three sections always show, as on macOS; an empty one links to creating an agent there.
        var local = conversations.Where(c => c.Target == ConversationTarget.Private).Select(Row).ToArray();
        var cloud = conversations.Where(c => c.Target is ConversationTarget.Sai or ConversationTarget.Box).Select(Row).ToArray();
        var pc = conversations.Where(c => c.Target == ConversationTarget.ThisPc).Select(Row).ToArray();
        var groups = new List<AgentGroup>
        {
            new("Private", "", "private", "New private agent…", local),
            new("This PC", "", "pc", "New agent for this PC…", pc),
            new("Cloud", "", "cloud", "New cloud agent…", cloud),
        };
        rebuilding = true;
        try
        {
            agentGroups.Source = groups;
            Agents.ItemsSource = agentGroups.View;
            Agents.SelectedItem = selected is null ? null : Row(selected);
        }
        finally { rebuilding = false; }
        EmptyAgents.Visibility = Visibility.Collapsed;
        UpdateBadges();
    }
    private void Select(Conversation? conversation)
    {
        if (conversation is null) { Agents.SelectedItem = null; SelectAgent(Agents, null!); return; }
        Agents.SelectedItem = Row(conversation);
    }
    private async void DeleteAgentFromMenu(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.Tag is not AgentRow row || busy || row.Conversation == runOwner) return;
        if (await Dialog($"Delete {row.Name}?", "This deletes the local conversation. It does not delete the computer or remote task history.", "Delete").ShowAsync() != ContentDialogResult.Primary) return;
        bool wasSelected = selected == row.Conversation;
        conversations.Remove(row.Conversation);
        ForgetMemory(row.Conversation);
        if (activityOwner == row.Conversation) { HideActivity(); activityOwner = null; }
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
        ComputerColumn.Width = panelOpen ? new GridLength(panelWidth) : new GridLength(0);
        ComputerPanel.Visibility = panelOpen ? Visibility.Visible : Visibility.Collapsed;
        if (panelOpen) _ = UpdateScreen();
    }

    private async void OpenModelMenu(object sender, RoutedEventArgs e)
    {
        if (selected is not { } owner || busy) return;
        var menu = new MenuFlyout { Placement = Microsoft.UI.Xaml.Controls.Primitives.FlyoutPlacementMode.Bottom };
        // The full lists (sizes, ZDR, which agents use each), first in the menu, for every agent that picks its own model.
        void AddBrowse()
        {
            if (owner.Target == ConversationTarget.Sai) return;
            var browse = new MenuFlyoutItem { Text = "Browse all models…", Icon = new FontIcon { Glyph = "" } };
            browse.Click += (_, _) => ModelBrowser.ShowAt(ModelPill, [ModelProvider.Ollama, ModelProvider.OpenRouter], async (provider, id) =>
            {
                owner.Provider = provider; owner.ModelId = id;
                Row(owner).Refresh(); UpdateBadges(); Controls(); await Persist();
            }, () => owner.Provider, () => owner.ModelId);
            menu.Items.Insert(0, browse);
            menu.Items.Insert(1, new MenuFlyoutSeparator());
        }
        IEnumerable<(string Id, string Label)> choices;
        if (owner.Target is ConversationTarget.ThisPc or ConversationTarget.Box)
        {
            // This PC and Box can use Ollama, or OpenRouter with the saved key.
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
            if (owner.Target == ConversationTarget.ThisPc)
            {
                var harnessMenu = new MenuFlyoutSubItem { Text = "Harness", IsEnabled = owner != runOwner };
                foreach (var harness in Harnesses.ThisPc)
                {
                    var item = new ToggleMenuFlyoutItem { Text = harness.DisplayName(), IsChecked = owner.ThisPcHarness == harness };
                    item.Click += async (_, _) =>
                    {
                        if (owner == runOwner) return;
                        owner.Harness = harness;
                        Row(owner).Refresh(); Controls(); await Persist();
                    };
                    harnessMenu.Items.Add(item);
                }
                menu.Items.Add(new MenuFlyoutSeparator());
                menu.Items.Add(harnessMenu);
            }
            AddBrowse(); menu.ShowAt(ModelPill);
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
        bool isPrivate = owner.Target == ConversationTarget.Private;
        if (isPrivate) menu.Items.Insert(0, new MenuFlyoutItem { Text = "Ollama on this PC", IsEnabled = false });
        foreach (var (id, label) in choices)
        {
            var item = new ToggleMenuFlyoutItem { Text = label, IsChecked = (owner.ModelId ?? "auto") == id && (!isPrivate || owner.Provider == ModelProvider.Ollama) };
            item.Click += async (_, _) => { owner.ModelId = id; if (isPrivate) owner.Provider = ModelProvider.Ollama; Controls(); await Persist(); };
            menu.Items.Add(item);
        }
        // Every Private harness, the built-in loop included, can use OpenRouter through the app's key-holding proxy.
        if (isPrivate)
        {
            menu.Items.Add(new MenuFlyoutSeparator());
            menu.Items.Add(new MenuFlyoutItem { Text = Secrets.ReadOpenRouter() is null ? "OpenRouter (add a key in Accounts)" : "OpenRouter", IsEnabled = false });
            var openRouterChoices = ThisPcAgent.SuggestedOpenRouterModels.ToList();
            if (owner.Provider == ModelProvider.OpenRouter && owner.ModelId is { } current && !openRouterChoices.Contains(current)) openRouterChoices.Insert(0, current);
            if (Secrets.ReadOpenRouter() is not null)
                foreach (var id in openRouterChoices)
                {
                    var item = new ToggleMenuFlyoutItem { Text = id, IsChecked = owner.Provider == ModelProvider.OpenRouter && owner.ModelId == id };
                    item.Click += async (_, _) => { owner.Provider = ModelProvider.OpenRouter; owner.ModelId = id; Controls(); await Persist(); };
                    menu.Items.Add(item);
                }
        }
        if (isPrivate && HarnessRunner.Supported)
        {
            menu.Items.Add(new MenuFlyoutSeparator());
            menu.Items.Add(new MenuFlyoutItem { Text = "Harness", IsEnabled = false });
            foreach (var harness in Harnesses.All)
            {
                var item = new ToggleMenuFlyoutItem { Text = harness.DisplayName(), IsChecked = (owner.Harness ?? PrivateHarness.AgentS) == harness };
                item.Click += async (_, _) =>
                {
                    owner.Harness = harness;
                    Controls(); await Persist();
                };
                menu.Items.Add(item);
            }
        }
        AddBrowse(); menu.ShowAt(ModelPill);
    }

    private void Controls()
    {
        // As on macOS, agents stay selectable and the composer stays usable while a turn runs.
        bool mine = running && runOwner is not null && runOwner == selected;
        Agents.IsEnabled = AccountsButton.IsEnabled = !busy;
        NewAgentButton.IsEnabled = !busy && storageReady;
        Composer.IsEnabled = !busy && selected is not null;
        bool hasDraft = !string.IsNullOrWhiteSpace(Composer.Text);
        SendButton.IsEnabled = !busy && storageReady && hasDraft && selected is not null;
        // One button: Stop while this agent works and the draft is empty, otherwise Send (a follow-up while running).
        bool showStop = mine && !hasDraft;
        StopButton.IsEnabled = showStop && !stopRequested;
        SendButton.Visibility = showStop ? Visibility.Collapsed : Visibility.Visible;
        StopButton.Visibility = showStop ? Visibility.Visible : Visibility.Collapsed;
        ToolTipService.SetToolTip(SendButton, running ? "Queue this follow-up (Enter)" : "Send (Enter)");
        WorkingChip.Visibility = mine ? Visibility.Visible : Visibility.Collapsed;
        FloatingChatButton.Visibility = IsThisPc ? Visibility.Visible : Visibility.Collapsed;
        if (activityVisible) RefreshActivity();
        foreach (var conversation in conversations) Row(conversation).Working = running && runOwner == conversation;

        // Model pill: "{harness} · {model}" with where it runs, as on macOS. Switch after the answer.
        ModelPill.IsEnabled = selected is not null && !busy && !mine;
        ToolTipService.SetToolTip(ModelPill, mine ? "Switch after this answer" : "Switch the harness or the model");
        ModelPillIcon.Glyph = selected?.Target switch
        {
            ConversationTarget.Private => "",   // lock
            ConversationTarget.ThisPc => "",    // computer
            _ => "",                            // cloud
        };
        ModelPillText.Text = selected is null ? "No agent selected" : PillText(selected);
        Composer.PlaceholderText = selected is null ? "Create an agent to start…"
            : mine && stopRequested ? "Stopping the current turn…"
            : mine ? "Queue a follow-up…"
            : running && runOwner is { } other ? $"Queue a message for when {other.Name} finishes…"
            : selected.Target switch
            {
                ConversationTarget.Sai => $"Ask {selected.Name} to do something on {CurrentComputer?.Name ?? "its computer"}…",
                ConversationTarget.Box => $"Ask {selected.Name} to do something on your Simular Box…",
                ConversationTarget.ThisPc => $"Ask {selected.Name} to do something on this PC…",
                _ => $"Ask {selected.Name} to do something on its computer…",
            };
        UpdateEmptyChat();
        RenderQueue();
        UpdateStatusBanner();

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
        if (IsBox && selected is not null)
        {
            // ⋯ menu: open the live view, wake the box, refresh the screen.
            bool hasBox = Secrets.ReadBox() is not null;
            ScreenButton.IsEnabled = !busy && hasBox;
            ToolTipService.SetToolTip(ScreenButton, "Open the live view in your browser");
            WakeButton.Text = "Wake box"; WakeButton.IsEnabled = !busy && hasBox;
            RefreshButton.Text = "Refresh screen"; RefreshButton.IsEnabled = !busy && hasBox;
            ComputerName.Text = selected.BoxName ?? "Simular Box";
            ComputerState.Text = !hasBox ? "Add your Box key in Accounts" : running ? "Working…" : boxState ?? "Simular Box";
            StatusDot.Fill = new Microsoft.UI.Xaml.Media.SolidColorBrush(running || boxState is "Running"
                ? global::Windows.UI.Color.FromArgb(255, 52, 199, 89) : global::Windows.UI.Color.FromArgb(255, 154, 154, 154));
        }
        UpdatePlaceholder();
        UpdateComputerView();
        UpdateContext();
    }

    // ---- Right sidebar: computer screen and context -----------------------------------------------

    private void UpdatePlaceholder()
    {
        bool showImage = ScreenKindFor(selected) is { } kind && kind == screenKind && PrivateScreen.Source is not null
            && (!IsBox || (boxState == "Running" && Secrets.ReadBox() is not null));
        bool showLive = !IsPrivate && !IsThisPc && !IsBox && liveViewUri is not null && CurrentComputer?.Online == true;
        PrivateScreen.Visibility = showImage ? Visibility.Visible : Visibility.Collapsed;
        LiveView.Visibility = showLive ? Visibility.Visible : Visibility.Collapsed;
        ScreenPlaceholder.Visibility = showImage || showLive ? Visibility.Collapsed : Visibility.Visible;
        PlaceholderAction.Visibility = Visibility.Collapsed;
        if (selected is null)
        { PlaceholderTitle.Text = "No computer yet"; PlaceholderText.Text = "Create an agent to see its computer here."; return; }
        if (IsThisPc)
        { PlaceholderTitle.Text = "This PC"; PlaceholderText.Text = "The agent works on this desktop. Screenshots it takes appear here."; return; }
        if (IsBox)
        {
            if (Secrets.ReadBox() is null)
            { PlaceholderTitle.Text = "Connect Simular Box"; PlaceholderText.Text = "Add your Box key to see this computer."; PlaceholderAction.Content = "Accounts"; }
            else if (boxState is "Asleep")
            { PlaceholderTitle.Text = "Your Box is asleep"; PlaceholderText.Text = "It wakes with the next task, or wake it now."; PlaceholderAction.Content = "Wake box"; }
            else
            { PlaceholderTitle.Text = "Simular Box"; PlaceholderText.Text = "Its screen appears here."; PlaceholderAction.Content = "Refresh screen"; }
            PlaceholderAction.Visibility = Visibility.Visible;
            return;
        }
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
            case "Wake computer": case "Wake box": WakeComputer(sender, e); break;
            case "Refresh screen": await RefreshBoxScreen(); break;
        }
    }

    /// <summary>Shows the selected Box's state and screen (asking doesn't wake it; a sleeping box shows no screen).</summary>
    private async Task RefreshBoxScreen()
    {
        if (selected is not { Target: ConversationTarget.Box } owner || !panelOpen || BoxRunner.Client() is not { } box) { Controls(); return; }
        try
        {
            var (state, _) = await box.Status(owner.BoxName);
            if (selected != owner) return;
            boxState = Box.IsAsleep(state) ? "Asleep" : "Running";
            if (boxState == "Running" && await box.Screenshot(owner.BoxName) is { } png && selected == owner)
                await ShowPrivateScreen(png, ScreenKindFor(owner)!);
        }
        catch (Exception error) when (error is BoxException or TaskCanceledException or HttpRequestException)
        {
            if (selected == owner && !running) { boxState = null; Show(Describe(error)); }
        }
        Controls();
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
        if (IsBox) { await RefreshBoxScreen(); return; }
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

    private Computer? CurrentComputer => selected?.Target == ConversationTarget.Sai
        ? computers.FirstOrDefault(c => c.Id == selected.ComputerId)
        : null;
    private void Show(string text, bool error = true)
    { Notice.Message = text; Notice.Severity = error ? InfoBarSeverity.Error : InfoBarSeverity.Informational; Notice.IsOpen = true; }
    private void Error(Exception error) => Show(Describe(error));
    private static string Describe(Exception error) => error is SaiException or PrivateComputerException or LocalAgentException or BoxException ? error.Message :
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
        if (IsBox) { await RefreshBoxScreen(); return; }
        if (!IsPrivate) { await Refresh(); await UpdateScreen(); return; }
        if (busy || running) { if (running) Show("Wait for the current turn to finish, or stop it first."); return; }
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


    private async void NewAgent(object sender, RoutedEventArgs e)
    {
        if (busy || !storageReady) return;
        bool privateReady = PrivateComputer.IsConfigured, loadingCloud = false, setupRunning = false;
        var wslState = WslComputer.Readiness.Ready;
        var secondary = Application.Current.Resources.TryGetValue("TextFillColorSecondaryBrush", out var brush)
            ? brush as Microsoft.UI.Xaml.Media.Brush : null;
        // Modes in the order of the "Where should it work?" options.
        const int PrivateMode = 0, PcMode = 1, CloudMode = 2, BoxMode = 3;
        static string Suggested(int mode) => mode switch { CloudMode => "Cloud agent", PcMode => "PC agent", BoxMode => "Box agent", _ => "Private agent" };
        static void Id(UIElement element, string id) => Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(element, id);
        TextBlock Note(string text = "")
        {
            var note = new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, FontSize = 12 };
            if (secondary is not null) note.Foreground = secondary;
            return note;
        }

        // Prefer whatever works on this PC right now: a set-up Private computer, else a connected Sai account.
        bool startCloud = preferCloud || (!preferPrivate && !privateReady && client is not null);
        preferPrivate = preferCloud = false;
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
            "Works in a private Linux computer on this PC, with a model on this PC or from OpenRouter.");
        var pcOption = Option("TargetThisPc", "This PC · your apps and files",
            "Works on this PC itself. It asks you before every action, and you can stop it any time.");
        var cloudOption = Option("TargetCloud", "Cloud · Sai computer",
            "Works on a computer in your Sai account. Nothing to install.");
        var boxOption = Option("TargetBox", "Cloud · Simular Box",
            "Works on your Simular Box, a cloud Ubuntu desktop. Claude Code drives it from this PC.");
        var target = new RadioButtons { Header = "Where should it work?" };
        target.Items.Add(privateOption); target.Items.Add(pcOption); target.Items.Add(cloudOption); target.Items.Add(boxOption);
        target.SelectedIndex = preferPc ? PcMode : startCloud ? CloudMode : PrivateMode;
        preferPc = false;

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
        // Private harnesses can also use OpenRouter (through the app's key-holding proxy); the built-in loop is Ollama only.
        var privateProvider = new ComboBox
        {
            Header = "Model provider", HorizontalAlignment = HorizontalAlignment.Stretch, Visibility = Visibility.Collapsed,
            ItemsSource = new[] { ModelProvider.Ollama.DisplayName(), ModelProvider.OpenRouter.DisplayName() }, SelectedIndex = 0,
        };
        Id(privateProvider, "PrivateProvider");
        var privateOpenRouterModel = new ComboBox
        {
            Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch, Visibility = Visibility.Collapsed, IsEditable = true,
            ItemsSource = ThisPcAgent.SuggestedOpenRouterModels, SelectedIndex = 0,
            Description = "Any id from openrouter.ai/models. Prompts go to OpenRouter.",
        };
        var privateOpenRouterField = WithBrowser(privateOpenRouterModel, [ModelProvider.OpenRouter]);
        var privateKey = new PasswordBox { Header = "OpenRouter API key", PlaceholderText = "Paste your key", Description = "Saved in Windows Credential Locker. The Private computer never sees it." };
        var privateSaveKey = new Button { Content = "Save key" };
        var privateKeyPanel = new StackPanel { Spacing = 8, Visibility = Visibility.Collapsed };
        privateKeyPanel.Children.Add(privateKey); privateKeyPanel.Children.Add(privateSaveKey);
        // Every Private harness, the built-in loop included, can use OpenRouter; then Ollama and a local model aren't needed.
        bool PrivateOpenRouter() => privateProvider.SelectedIndex == 1;
        bool PrivateNeedsOllama() => !PrivateOpenRouter();
        string PrivateOpenRouterModel() => (privateOpenRouterModel.SelectedItem as string ?? privateOpenRouterModel.Text ?? "").Trim();
        // As on macOS: the suggested models are always offered; one that isn't downloaded yet shows Download with progress.
        string[] installedLocal = [];
        var localPanel = new OllamaModelPanel(OllamaSetup.DefaultModel, () => PrivateComputer.IsConfigured ? 0 : PrivateComputerMemory);
        var localBrowse = OllamaModelPanel.BrowseButton("Browse Ollama models…", chosen =>
        {
            var items = (localModel.ItemsSource as string[] ?? []).ToList();
            if (!items.Any(m => OllamaModels.Same(m, chosen))) { items.Add(chosen); localModel.ItemsSource = items.ToArray(); }
            localModel.SelectedItem = items.First(m => OllamaModels.Same(m, chosen));
        });
        var localModelPart = new StackPanel { Spacing = 8, Visibility = Visibility.Collapsed };
        localModelPart.Children.Add(localModel); localModelPart.Children.Add(localBrowse); localModelPart.Children.Add(localPanel);
        localModel.Visibility = Visibility.Visible;
        var local = new StackPanel { Spacing = 12 };
        local.Children.Add(localHarness); local.Children.Add(privateProvider); local.Children.Add(setupCard);
        local.Children.Add(localModelPart); local.Children.Add(privateOpenRouterField); local.Children.Add(privateKeyPanel);

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

        // ---- This PC harness selection and the model providers shared with Box ----
        var claudePath = ThisPcRunner.FindClaude();
        var pcHarness = new ComboBox
        {
            Header = "Harness", HorizontalAlignment = HorizontalAlignment.Stretch,
            ItemsSource = Harnesses.ThisPc.Select(h => h.DisplayName()).ToArray(), SelectedIndex = 0,
        };
        Id(pcHarness, "PcHarness");
        PrivateHarness SelectedPcHarness() => Harnesses.ThisPc[Math.Max(0, pcHarness.SelectedIndex)];
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
        // Either list: choosing an OpenRouter model switches the provider too.
        var pcModelField = WithBrowser(pcModel, [ModelProvider.Ollama, ModelProvider.OpenRouter],
            provider => pcProvider.SelectedIndex = provider == ModelProvider.OpenRouter ? 1 : 0,
            () => pcProvider.SelectedIndex == 1 ? ModelProvider.OpenRouter : ModelProvider.Ollama);
        var openRouterKey = new PasswordBox { Header = "OpenRouter API key", PlaceholderText = "Paste your key", Description = "Saved in Windows Credential Locker. Agents never see it." };
        var saveOpenRouter = new Button { Content = "Save key" };
        var getOpenRouter = new HyperlinkButton { Content = "Get a key", NavigateUri = ThisPcRunner.OpenRouterKeys };
        var openRouterActions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        openRouterActions.Children.Add(saveOpenRouter); openRouterActions.Children.Add(getOpenRouter);
        var openRouterPanel = new StackPanel { Spacing = 8 };
        openRouterPanel.Children.Add(openRouterKey); openRouterPanel.Children.Add(openRouterActions);
        var desktop = new StackPanel { Spacing = 12 };
        const string pcNote = "Harness: Claude Code, installed on this PC. Agent S uses its own Claude Code profile, and asks you before every command, click or file change. Looking (a screenshot or a window's contents) runs without asking.";
        const string boxNote = "Harness: Claude Code, installed on this PC, with no tools on this PC: it works only through your Box's tools (browser, desktop, shell and files on the Box).";
        var desktopNote = Note(pcNote);
        desktop.Children.Add(desktopNote);
        desktop.Children.Add(pcHarness); desktop.Children.Add(pcNotice); desktop.Children.Add(pcProvider); desktop.Children.Add(openRouterPanel); desktop.Children.Add(pcModelField);
        bool UseOpenRouter() => pcProvider.SelectedIndex == 1;
        string PcModelText() => (pcModel.SelectedItem as string ?? pcModel.Text ?? "").Trim();
        void ShowPcModels()
        {
            var keep = PcModelText();
            var items = UseOpenRouter() ? openRouterModels : installedLocal;
            pcModel.ItemsSource = items;
            pcModel.SelectedItem = items.Contains(keep) ? keep : items.FirstOrDefault();
            if (pcModel.SelectedItem is null) pcModel.Text = "";
            pcModel.Description = UseOpenRouter() ? "Any id from openrouter.ai/models. Prompts go to OpenRouter." : "Runs on this PC through Ollama.";
        }

        // ---- Simular Box: the key (if not saved) and which box ----
        var boxKey = new PasswordBox { Header = "Simular Box key", PlaceholderText = "Paste your key", Description = "Saved in Windows Credential Locker." };
        Id(boxKey, "BoxKey");
        var saveBoxKey = new Button { Content = "Save key" };
        var getBoxKey = new HyperlinkButton { Content = "Open Simular Box", NavigateUri = Box.Portal };
        var boxKeyActions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        boxKeyActions.Children.Add(saveBoxKey); boxKeyActions.Children.Add(getBoxKey);
        var boxKeyPanel = new StackPanel { Spacing = 8 };
        boxKeyPanel.Children.Add(boxKey); boxKeyPanel.Children.Add(boxKeyActions);
        var boxName = new TextBox { Header = "Box", PlaceholderText = "Your default box", Description = "Leave empty for the account's default box, or type a box's name." };
        Id(boxName, "BoxName");
        var boxPanel = new StackPanel { Spacing = 12 };
        boxPanel.Children.Add(boxKeyPanel); boxPanel.Children.Add(boxName);

        var hint = Note();
        var form = new StackPanel { Spacing = 16, MinWidth = 420 };
        form.Children.Add(name); form.Children.Add(target); form.Children.Add(local); form.Children.Add(boxPanel); form.Children.Add(desktop); form.Children.Add(cloud); form.Children.Add(hint);
        var dialog = Dialog("New agent", new ScrollViewer { Content = form, VerticalScrollBarVisibility = ScrollBarVisibility.Auto }, "Create");
        dialog.DefaultButton = ContentDialogButton.Primary;

        int Mode() => Math.Max(0, target.SelectedIndex);
        bool UseCloud() => Mode() == CloudMode;
        void SetMode()
        {
            int mode = Mode();
            bool useCloud = mode == CloudMode;
            local.Visibility = mode == PrivateMode ? Visibility.Visible : Visibility.Collapsed;
            desktop.Visibility = mode is PcMode or BoxMode ? Visibility.Visible : Visibility.Collapsed;
            desktopNote.Text = mode == BoxMode ? boxNote : pcNote;
            pcHarness.Visibility = mode == PcMode ? Visibility.Visible : Visibility.Collapsed;
            var chosenHarness = mode == PcMode ? SelectedPcHarness() : PrivateHarness.ClaudeCode;
            string? missing = mode == PcMode ? ThisPcRunner.Missing(chosenHarness)
                : mode == BoxMode && claudePath is null ? "Install Claude Code on this PC." : null;
            pcNotice.Message = missing ?? "";
            pcNotice.IsOpen = missing is not null;
            pcInstall.Content = chosenHarness == PrivateHarness.DeepSeek ? "DeepSeek setup instructions" : $"Install {chosenHarness.DisplayName()}";
            pcInstall.NavigateUri = chosenHarness switch
            {
                PrivateHarness.Codex => ThisPcRunner.CodexSetup,
                PrivateHarness.DeepSeek => ThisPcRunner.DeepSeekSetup,
                _ => ThisPcRunner.ClaudeSetup,
            };
            pcNotice.ActionButton = missing is null || missing.StartsWith("Install Git", StringComparison.Ordinal) ? null : pcInstall;
            boxPanel.Visibility = mode == BoxMode ? Visibility.Visible : Visibility.Collapsed;
            bool hasBoxKey = Secrets.ReadBox() is not null;
            boxKeyPanel.Visibility = hasBoxKey ? Visibility.Collapsed : Visibility.Visible;
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
                PcMode or BoxMode => missing is not null ? missing
                    : mode == BoxMode && !hasBoxKey ? "Add your Simular Box key to continue."
                    : UseOpenRouter() && !hasOpenRouterKey ? "Add your OpenRouter key to continue."
                    : !UseOpenRouter() && !checkedPc ? "Checking Ollama…"
                    : !UseOpenRouter() && !ollamaReady ? "Start Ollama on this PC, or choose OpenRouter."
                    : PcModelText().Length == 0 ? "Choose a model to continue." : null,
                _ => setupRunning ? "Setting up this PC…"
                    : !checkedPc ? "Checking this PC…"
                    : !(privateReady && (!PrivateNeedsOllama() || (ollamaReady && modelReady))) ? "Choose Set up this PC, or pick Cloud."
                    : PrivateOpenRouter() && !hasOpenRouterKey ? "Add your OpenRouter key to continue."
                    : PrivateOpenRouter() && PrivateOpenRouterModel().Length == 0 ? "Choose a model to continue."
                    : !PrivateOpenRouter() && localModel.SelectedItem is not string ? "Choose a model to continue."
                    : !PrivateOpenRouter() && !localPanel.IsReady ? $"Download {OllamaModels.DisplayName(localPanel.Model)} to continue." : null,
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
            installedLocal = installed.Where(m => !OllamaModels.IsCloudReference(m)).Select(OllamaModels.NormalizedName).ToArray();
            var items = installedLocal.Concat(OllamaModels.Suggested.Where(s => !installedLocal.Any(m => OllamaModels.Same(m, s)))).ToArray();
            localModel.ItemsSource = items;
            localModel.SelectedItem = items.FirstOrDefault(m => OllamaModels.Same(m, OllamaSetup.DefaultModel)) ?? items.FirstOrDefault();
        }
        void ShowReadiness()
        {
            bool allReady = privateReady && (!PrivateNeedsOllama() || (ollamaReady && modelReady));
            // With OpenRouter, setup only needs the Private computer.
            ((FrameworkElement)ollamaStep.Icon.Parent).Visibility = ((FrameworkElement)modelStep.Icon.Parent).Visibility = PrivateNeedsOllama() ? Visibility.Visible : Visibility.Collapsed;
            setupCard.Visibility = allReady ? Visibility.Collapsed : Visibility.Visible;
            localHarness.Visibility = Visibility.Visible;
            privateProvider.Visibility = Visibility.Visible;
            localModelPart.Visibility = allReady && !PrivateOpenRouter() ? Visibility.Visible : Visibility.Collapsed;
            privateOpenRouterModel.Visibility = PrivateOpenRouter() ? Visibility.Visible : Visibility.Collapsed;
            privateKeyPanel.Visibility = PrivateOpenRouter() && Secrets.ReadOpenRouter() is null ? Visibility.Visible : Visibility.Collapsed;
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
            Mark(modelStep, modelReady ? "done" : "todo", modelReady ? $"{OllamaModels.DisplayName(OllamaSetup.DefaultModel)} is ready."
                : $"Downloads {OllamaModels.DisplayName(OllamaSetup.DefaultModel)} (about {OllamaModels.PresetSize(OllamaSetup.DefaultModel)}). You can pick another model next.");
            if (!privateReady) wslState = await WslComputer.CheckWsl();
            Mark(linuxStep, privateReady ? "done" : "todo", privateReady ? "Ready." : wslState switch
            {
                WslComputer.Readiness.NeedsRestart => "Windows needs a restart to finish turning on its virtual machine feature.",
                WslComputer.Readiness.NeedsWsl => "Turns on Windows' virtual machine feature, then creates a private desktop. Needs one restart.",
                _ => "Creates a private Linux desktop for the agent (about 400 MB).",
            });
            // When only the restart is left, the one button becomes "Restart now".
            restartPending = !privateReady && wslState == WslComputer.Readiness.NeedsRestart && (!PrivateNeedsOllama() || (ollamaReady && modelReady));
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
                if (PrivateNeedsOllama() && !ollamaReady)
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
                if (PrivateNeedsOllama() && !modelReady)
                {
                    Mark(current, "working", $"Downloading {OllamaModels.DisplayName(OllamaSetup.DefaultModel)}…");
                    await OllamaSetup.Pull(OllamaSetup.DefaultModel, ProgressFor(modelStep), CancellationToken.None);
                    modelReady = true;
                    Mark(current, "done", $"{OllamaModels.DisplayName(OllamaSetup.DefaultModel)} is ready.");
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
            if (name.Text.Trim() is "" or "Private agent" or "PC agent" or "Cloud agent" or "Box agent") name.Text = Suggested(Mode());
            if (Mode() is PcMode or BoxMode) ShowPcModels();
            SetMode();
            if (UseCloud() && client is not null && computers.Length == 0 && !loadingCloud) await LoadCloud(Refresh);
        };
        computer.SelectionChanged += (_, _) => SetMode();
        localModel.SelectionChanged += async (_, _) => { SetMode(); if (localModel.SelectedItem is string chosenModel) await localPanel.Use(chosenModel); };
        localPanel.Changed += () => SetMode();
        name.TextChanged += (_, _) => SetMode();
        pcHarness.SelectionChanged += (_, _) => SetMode();
        pcProvider.SelectionChanged += async (_, _) =>
        {
            ShowPcModels(); SetMode(); RefreshModelField(pcModelField);
            // The full list loads in the background; the suggestions show at once.
            if (UseOpenRouter() && openRouterModels == ThisPcAgent.SuggestedOpenRouterModels)
            {
                var all = await OpenRouterProxy.Models();
                if (all.Length > 0) { openRouterModels = ThisPcAgent.SuggestedOpenRouterModels.Concat(all.Except(ThisPcAgent.SuggestedOpenRouterModels)).ToArray(); if (UseOpenRouter()) ShowPcModels(); }
            }
        };
        localHarness.SelectionChanged += (_, _) => ShowReadiness();
        privateProvider.SelectionChanged += (_, _) => ShowReadiness();
        privateOpenRouterModel.SelectionChanged += (_, _) => SetMode();
        privateOpenRouterModel.TextSubmitted += (_, _) => SetMode();
        privateOpenRouterModel.LostFocus += (_, _) => SetMode();
        privateSaveKey.Click += (_, _) =>
        {
            var value = privateKey.Password.Trim();
            if (value.Length == 0) return;
            try { Secrets.SaveOpenRouter(value); } catch (Exception failure) { Error(failure); }
            privateKey.Password = ""; ShowReadiness();
        };
        pcModel.SelectionChanged += (_, _) => SetMode();
        pcModel.TextSubmitted += (_, _) => SetMode();
        pcModel.LostFocus += (_, _) => SetMode();
        saveBoxKey.Click += (_, _) =>
        {
            var value = boxKey.Password.Trim();
            if (value.Length == 0) return;
            try { Secrets.SaveBox(value); } catch (Exception failure) { Error(failure); }
            boxKey.Password = ""; SetMode();
        };
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
            ShowPcModels(); SetMode(); RefreshModelField(pcModelField);
        }
        if (startCloud && computers.Length == 0) _ = LoadCloud(Refresh);

        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        var conversation = new Conversation { Name = name.Text.Trim() };
        if (Mode() is PcMode or BoxMode)
        {
            if (PcModelText() is not { Length: > 0 } pcChoice) return;
            conversation.Target = Mode() == BoxMode ? ConversationTarget.Box : ConversationTarget.ThisPc;
            if (Mode() == BoxMode && boxName.Text.Trim() is { Length: > 0 } chosenBox) conversation.BoxName = chosenBox;
            conversation.Harness = Mode() == PcMode ? SelectedPcHarness() : PrivateHarness.ClaudeCode;
            conversation.Provider = UseOpenRouter() ? ModelProvider.OpenRouter : ModelProvider.Ollama;
            conversation.ModelId = pcChoice;
        }
        else if (!UseCloud())
        {
            if (!PrivateOpenRouter() && localModel.SelectedItem is not string) return;
            var localChoice = localModel.SelectedItem as string ?? OllamaSetup.DefaultModel;
            conversation.Target = ConversationTarget.Private;
            conversation.ModelId = localChoice;
            var harness = harnessChoices[Math.Max(0, localHarness.SelectedIndex)];
            conversation.Harness = harness.RunsInComputer() && !HarnessRunner.Supported ? PrivateHarness.AgentS : harness;
            if (PrivateOpenRouter())
            {
                conversation.Provider = ModelProvider.OpenRouter;
                conversation.ModelId = PrivateOpenRouterModel();
            }
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
        // Each agent keeps its own draft, as on macOS.
        if (selected is not null) drafts[selected] = Composer.Text;
        selected = next; usage = null; boxState = null;
        Composer.Text = selected is not null && drafts.TryGetValue(selected, out var draft) ? draft : "";
        Messages.Children.Clear(); entryText.Clear(); entryMarkdown.Clear(); approvalControls.Clear();
        stepGroups.Clear(); openGroup = null;
        if (selected is not null) foreach (var entry in selected.Entries) Draw(entry);
        openGroup?.Refresh();
        Controls();
        // After the rows (and folded step groups) have laid out, show the newest message.
        DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () => { Transcript.UpdateLayout(); Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true); });
        if (!busy && selected?.Target == ConversationTarget.Sai && client is not null && computers.Length == 0)
            await Refresh();
        await UpdateScreen();
    }

    private static void SetButtonsEnabled(StackPanel panel, bool enabled)
    {
        foreach (var control in panel.Children.OfType<Control>()) control.IsEnabled = enabled;
    }

    /// <summary>A palette brush for the current theme (App.xaml keeps the palette in a dictionary merged after WinUI's).</summary>
    private Microsoft.UI.Xaml.Media.Brush Theme(string key)
    {
        var theme = Root.ActualTheme == ElementTheme.Dark ? "Dark" : "Light";
        foreach (var dictionary in Application.Current.Resources.MergedDictionaries.Reverse())
            if (dictionary.ThemeDictionaries.TryGetValue(theme, out var found) && found is ResourceDictionary palette && palette.TryGetValue(key, out var brush))
                return (Microsoft.UI.Xaml.Media.Brush)brush;
        return (Microsoft.UI.Xaml.Media.Brush)Application.Current.Resources[key];
    }

    /// <summary>A group shows a spinner only for the selected agent's running turn, and only the newest group.</summary>
    private bool GroupRunning(ChatSurface surface, ActivityGroup group) => running && runOwner == surface.Owner() && group == surface.OpenGroup;

    private void Draw(ChatEntry entry) => Draw(entry, Chat);

    /// <summary>Draws one row into <paramref name="surface"/>: the main chat, or the floating window (smaller).</summary>
    private void Draw(ChatEntry entry, ChatSurface surface)
    {
        bool compact = surface.Compact;
        var text = new TextBlock { Text = entry.Text, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true, FontSize = compact ? 14 : 16, LineHeight = compact ? 20 : 24 };
        surface.Text[entry.Id] = text;
        FrameworkElement element;
        StackPanel? panel = null;
        // Tool steps fold into one row per stretch, as on macOS.
        if (entry.Role == "Activity" && entry.Tool is not null)
        {
            surface.Text.Remove(entry.Id);
            var near = surface.Following;
            if (surface.OpenGroup is null)
            {
                surface.OpenGroup = new ActivityGroup(Theme, group => GroupRunning(surface, group));
                surface.Panel.Children.Add(surface.OpenGroup);
            }
            surface.OpenGroup.Add(entry);
            surface.Groups[entry.Id] = surface.OpenGroup;
            if (near) DispatcherQueue.TryEnqueue(() => surface.Scroll.ChangeView(null, surface.Scroll.ScrollableHeight, null, true));
            return;
        }
        var ended = surface.OpenGroup;
        surface.OpenGroup = null;
        ended?.Refresh();
        switch (entry.Role)
        {
            case "You":
                // Black bubble on the right, as on macOS.
                text.Foreground = Theme("BubbleForeground");
                element = PromptRow(text, new Border
                {
                    Child = text, Background = Theme("BubbleBackground"), CornerRadius = new CornerRadius(compact ? 16 : 20),
                    Padding = compact ? new Thickness(14, 8, 14, 8) : new Thickness(18, 10, 18, 10), HorizontalAlignment = HorizontalAlignment.Right, MaxWidth = compact ? 300 : 560,
                });
                break;
            case "Activity":
            {
                // A status line (which model, "Starting the Private computer…"), not a tool step.
                text.FontSize = 13; text.Foreground = Theme("MutedText"); text.MaxLines = 2; text.TextTrimming = TextTrimming.CharacterEllipsis;
                element = text;
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
                    Child = panel, BorderBrush = Theme("CardBorder"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(compact ? 14 : 18),
                    Background = Theme("CardBackground"), Padding = compact ? new Thickness(12, 10, 12, 10) : new Thickness(18, 14, 18, 14), HorizontalAlignment = HorizontalAlignment.Left, MaxWidth = compact ? 360 : 600,
                };
                break;
            }
            default:
            {
                // Agent replies: Markdown, left aligned. Streamed text re-renders as it grows (see Change).
                var markdown = MarkdownView.Create(MemoryWire.VisibleText(entry.Text), Theme);
                surface.Text.Remove(entry.Id);
                surface.Markdown[entry.Id] = markdown;
                element = markdown;
                break;
            }
        }
        if (panel is not null && entry.LocalAnswer is { } localAnswer && !entry.Answered)
        {
            // This PC: the exact step is shown above; nothing runs until the person chooses.
            var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            // The same answer as the floating Agent Activity window's buttons; either one updates both.
            void Answer(bool allowed, string note, bool rest = false)
            {
                AnswerLocal(entry, allowed, note, rest);
                SetButtonsEnabled(buttons, false);
                text.Text = entry.Text;
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
            surface.Approvals[entry.Id] = buttons;
        }
        else if (panel is not null && entry.ApprovalId is not null && !entry.Answered)
        {
            var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            if (entry.LinkOnly)
            {
                if (entry.ApprovalUrl is { } url)
                {
                    var open = new Button { Content = "Open to continue" };
                    open.Click += async (_, _) => { if (!await Launcher.LaunchUriAsync(url)) Show("Could not open the approval in your browser."); };
                    buttons.Children.Add(open);
                    buttons.Children.Add(new TextBlock { Text = "Sai continues once it's done.", Opacity = 0.7, FontSize = 13, VerticalAlignment = VerticalAlignment.Center });
                }
                else panel.Children.Add(new TextBlock { Text = "No valid approval link was supplied.", TextWrapping = TextWrapping.Wrap });
            }
            else
            {
                // As on macOS: Approve, Approve for this task, Deny.
                foreach (var (label, answer, done) in new[] { ("Approve", "approve", "Approved."), ("Approve for this task", "approve_for_task", "Approved for this task."), ("Deny", "deny", "Denied.") })
                {
                    var button = new Button { Content = label };
                    if (answer == "approve") Accent(button);
                    button.Click += async (_, _) =>
                    {
                        SetButtonsEnabled(buttons, false);
                        try
                        {
                            if (client is null) throw new SaiException("Connect your Sai account first.");
                            await client.Approve(entry.ApprovalId, answer);
                            entry.Answered = true; text.Text = entry.Text += "\n" + done;
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
            surface.Approvals[entry.Id] = buttons;
        }
        var following = surface.Following;
        surface.Panel.Children.Add(element);
        if (following) DispatcherQueue.TryEnqueue(() => surface.Scroll.ChangeView(null, surface.Scroll.ScrollableHeight, null, true));
    }
    private void Add(Conversation owner, ChatEntry entry)
    {
        owner.Entries.Add(entry);
        Row(owner).Refresh();
        if (selected == owner) { Draw(entry); EmptyChat.Visibility = Visibility.Collapsed; UpdateContext(); }
        // The floating window shows the same conversation, row for row.
        if (activityChat is not null && activityOwner == owner) { Draw(entry, activityChat); RefreshActivity(); }
    }
    private void Change(ChatEntry entry)
    {
        if (selected is not null) Row(selected).Refresh();
        foreach (var surface in Surfaces)
        {
            var following = surface.Following;
            if (surface.Groups.TryGetValue(entry.Id, out var group)) group.Refresh();
            else if (surface.Text.TryGetValue(entry.Id, out var text)) text.Text = entry.Text;
            else if (surface.Markdown.TryGetValue(entry.Id, out var markdown)) MarkdownView.Fill(markdown, entry.Text, Theme);
            else continue;
            if (following) DispatcherQueue.TryEnqueue(() => surface.Scroll.ChangeView(null, surface.Scroll.ScrollableHeight, null, true));
        }
        if (activityVisible && activityOwner?.Entries.Contains(entry) == true) RefreshActivity();
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
                var hasInput = value.TryGetProperty("input", out var input);
                if (tool is null) { tool = new ChatEntry { Id = toolId, Role = "Activity", Tool = Str(value, "toolName") ?? "tool" }; Add(owner, tool); }
                tool.Text = (hasInput ? Str(input, "title") : null) ?? Str(value, "toolName") ?? "Working";
                if (hasInput) tool.Input = ToolActivity.Detail(input) ?? HarnessStream.Clip(input.GetRawText());
                Change(tool); break;
            case "data-progress":
                var progress = owner.Entries.LastOrDefault(entry => entry.Role == "Activity" && entry.Tool is not null);
                if (progress is not null) { progress.Output = HarnessStream.Clip((progress.Output is null ? "" : progress.Output + "\n") + Str(data, "text")); Change(progress); }
                break;
            case "tool-output-available":
                if (owner.Entries.FirstOrDefault(entry => entry.Id == turn + "/tool/" + Str(value, "toolCallId")) is { } finished)
                {
                    finished.Done = true;
                    if (value.TryGetProperty("output", out var output))
                        finished.Output = HarnessStream.Clip(output.ValueKind == JsonValueKind.String ? output.GetString()! : output.GetRawText());
                    Change(finished);
                }
                break;
            case "tool-output-error":
                if (owner.Entries.FirstOrDefault(entry => entry.Id == turn + "/tool/" + Str(value, "toolCallId")) is { } failedStep)
                { failedStep.Failed = true; failedStep.Output = Str(value, "errorText") ?? "The action failed."; Change(failedStep); }
                else Add(owner, new ChatEntry { Role = "Error", Text = Str(value, "errorText") ?? "An action failed." });
                break;
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

    private void Send(object sender, RoutedEventArgs e)
    {
        if (busy || selected is null || string.IsNullOrWhiteSpace(Composer.Text)) return;
        var owner = selected;
        var message = Composer.Text.Trim();
        // While a turn runs, a message becomes a follow-up, as on macOS ("Queue a follow-up…").
        if (running)
        {
            Composer.Text = "";
            Enqueue(owner, message);
            return;
        }
        if (!CanRun(owner)) return;
        Composer.Text = "";
        _ = RunTurn(owner, message);
    }

    /// <summary>Whether a turn can start for this agent now; shows why not.</summary>
    private bool CanRun(Conversation owner)
    {
        bool local = owner.Target == ConversationTarget.Private;
        bool pc = owner.Target == ConversationTarget.ThisPc;
        bool onBox = owner.Target == ConversationTarget.Box;
        if (!local && !pc && !onBox && (owner.Target != ConversationTarget.Sai || client is null)) { Show("Connect your Sai account in Settings first."); return false; }
        if (!local && !pc && !onBox && computers.All(c => c.Id != owner.ComputerId)) { Show("The saved computer is not available on this account. Refresh computers or create a new agent."); return false; }
        if (pc && ThisPcRunner.Missing(owner.ThisPcHarness) is { } missing) { Show(missing); return false; }
        if (local && !PrivateComputer.IsConfigured) { Show("This PC's Private computer isn't set up yet. Choose New agent › Private › Set up this PC."); return false; }
        return true;
    }

    /// <summary>The agent whose turn is running (one at a time); null when idle.</summary>
    private Conversation? runOwner;
    private readonly Dictionary<Conversation, string> drafts = [];
    /// <summary>The running turn's own message, recorded as the source of notes it changes.</summary>
    private string currentUserEntryId = "";

    /// <returns>True when the turn finished on its own; false when it failed or was stopped.</returns>
    private async Task<bool> RunTurn(Conversation owner, string message)
    {
        bool local = owner.Target == ConversationTarget.Private;
        bool pc = owner.Target == ConversationTarget.ThisPc;
        bool onBox = owner.Target == ConversationTarget.Box;
        bool succeeded = false;
        var history = owner.Entries.Where(entry => entry.Role is "You" or "Agent").TakeLast(12)
            .Select(entry => new PrivateMessage(entry.Role == "You" ? "user" : "assistant", entry.Text)).ToArray();
        running = true; runOwner = owner; stopped = stopRequested = false; Notice.IsOpen = false;
        currentSession = null;
        var cancellation = new CancellationTokenSource();
        turn = Guid.NewGuid().ToString("N"); run = cancellation; Controls();
        var userEntry = new ChatEntry { Role = "You", Text = message };
        currentUserEntryId = userEntry.Id;
        Add(owner, userEntry);
        DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
        try
        {
            await Persist();
            if (pc)
            {
                await RunThisPc(owner, message, cancellation.Token);
            }
            else if (onBox)
            {
                if (!panelOpen) TogglePanel(this, new RoutedEventArgs());
                await RunBox(owner, message, cancellation.Token);
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
                // OpenRouter goes through the app's key-holding proxy (ZDR applies); Ollama stays on this PC.
                IAsyncEnumerable<PrivateEvent> turn;
                if (owner.Provider == ModelProvider.OpenRouter)
                {
                    if (Secrets.ReadOpenRouter() is null) throw new PrivateComputerException("Add your OpenRouter key in Accounts, then send again.");
                    var proxy = ThisPcRunner.OpenRouterProxy();
                    turn = agent.RunOpenRouter(message, owner.ModelId ?? ThisPcAgent.SuggestedOpenRouterModels[0], history, computer,
                        (new Uri(proxy.BaseUrl + "/"), proxy.Token), cancellation.Token);
                }
                else turn = agent.Run(message, owner.ModelId ?? OllamaSetup.DefaultModel, history, computer, cancellation.Token);
                await foreach (var progress in turn)
                {
                    if (progress.Screenshot is { } screenshot) await ShowPrivateScreen(screenshot);
                    if (string.IsNullOrWhiteSpace(progress.Text)) continue;
                    // Desktop actions are tool steps; "Private desktop ready." is a status line.
                    bool step = progress.Role == "Activity" && progress.Text != "Private desktop ready.";
                    Add(owner, new ChatEntry { Role = progress.Role, Text = progress.Text, Tool = step ? "desktop" : null, Done = step });
                }
            }
            else
            {
                await foreach (var value in client!.Run(message, owner.ComputerId, owner.SessionId, owner.ModelId, cancellation.Token))
                    await Apply(owner, value);
                if (stopRequested && !stopped) Show("The stream ended without confirming that the remote task stopped.");
            }
            succeeded = !stopped && !stopRequested;
        }
        catch (OperationCanceledException) when (stopped)
        { Show(local ? "Private turn stopped." : pc ? "Stopped. Nothing else runs on this PC." : onBox ? "Stopped. Claude Code was closed; a Box action already sent may still finish." : "Sai confirmed the stop request.", false); }
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
            foreach (var controls in Surfaces.SelectMany(s => s.Approvals.Values)) SetButtonsEnabled(controls, false);
            // Steps that never finished stay "unfinished"; the group's spinner stops.
            run = null; running = false; runOwner = null; cancellation.Dispose();
            foreach (var group in Surfaces.SelectMany(s => s.Groups.Values).Distinct()) group.Refresh();
            Row(owner).Working = false;
            Controls();
            if (pc) ActivityTurnEnded(owner);
            await Persist();
        }
        await ContinueQueue(owner, succeeded);
        return succeeded;
    }

    /// <summary>One Claude Code turn on this PC itself, every action approved by the person, like "This Mac".</summary>
    private async Task RunThisPc(Conversation owner, string message, CancellationToken token)
    {
        var harness = owner.ThisPcHarness;
        pcTurnOwner = owner; approveRestOfTurn = false;
        var model = owner.ModelId ?? "gemma4";
        Add(owner, new ChatEntry { Role = "Activity", Text = $"{harness.DisplayName()} · {(owner.Provider == ModelProvider.OpenRouter ? "OpenRouter · " : "")}{model}" });
        var calls = new Dictionary<string, (ChatEntry Entry, bool Screenshot)>();
        var resume = owner.HarnessSessions.GetValueOrDefault(harness);
        bool started = false, reportedError = false, finished = false;
        var memory = BeginMemory(owner, message, currentUserEntryId);
        try
        {
            await foreach (var raw in ThisPcRunner.Run(owner, memory.Prompt, resume, model, Decide, token))
            foreach (var item in memory.Consume(raw))
            {
                if (item is HarnessFinished { IsError: false }) finished = true;
                switch (item)
                {
                    case HarnessStarted begun:
                        started = true; owner.HarnessSessions[harness] = begun.SessionId; UpdateContext(); break;
                    case HarnessText or HarnessTextDelta or HarnessTextSnapshot:
                        RenderText(owner, item); break;
                    case HarnessToolCall call:
                        // "pc apps" reads better than the tool's full path.
                        var activity = new ChatEntry { Role = "Activity", Tool = call.Name, Text = ShortPc(call.Summary), Input = call.Detail is { } d ? ShortPc(d) : null };
                        calls[call.Id] = (activity, call.Summary.EndsWith(" screenshot", StringComparison.Ordinal));
                        Add(owner, activity); break;
                    case HarnessToolResult result when calls.TryGetValue(result.Id, out var done):
                        done.Entry.Output = result.Output;
                        if (result.IsError) done.Entry.Failed = true; else done.Entry.Done = true;
                        Change(done.Entry);
                        if (!result.IsError && done.Screenshot && File.Exists(ThisPcRunner.ScreenshotPath))
                        {
                            try { await ShowPrivateScreen(await File.ReadAllBytesAsync(ThisPcRunner.ScreenshotPath, token), "pc"); } catch (IOException) { }
                        }
                        break;
                    case HarnessUsage used:
                        usage = used; UpdateContext(); break;
                    case HarnessFinished { IsError: true } end when !reportedError:
                        reportedError = true;
                        Add(owner, new ChatEntry { Role = "Error", Text = Readable(end.Result) ?? $"{harness.DisplayName()} reported an error." }); break;
                }
            }
        }
        catch (PrivateComputerException) when (!started && resume is not null)
        {
            owner.HarnessSessions.Remove(harness);
            throw new PrivateComputerException($"{harness.DisplayName()} could not continue its earlier session. Send the message again to start a new one.");
        }
        finally { FinishMemory(owner, memory, finished && !reportedError && !token.IsCancellationRequested); }
    }

    /// <summary>One Claude Code turn on a Simular Box, through Box's tools only, as on macOS.</summary>
    private async Task RunBox(Conversation owner, string message, CancellationToken token)
    {
        var model = owner.ModelId ?? "gemma4";
        Add(owner, new ChatEntry { Role = "Activity", Text = $"Claude Code · {(owner.Provider == ModelProvider.OpenRouter ? "OpenRouter · " : "")}{model} · {owner.BoxName ?? "Simular Box"}" });
        using var screens = CancellationTokenSource.CreateLinkedTokenSource(token);
        var watching = WatchBoxScreen(owner, screens.Token);
        var calls = new Dictionary<string, ChatEntry>();
        var resume = owner.HarnessSessions.GetValueOrDefault(PrivateHarness.ClaudeCode);
        bool started = false, reportedError = false, finished = false;
        var memory = BeginMemory(owner, message, currentUserEntryId);
        try
        {
            await foreach (var raw in BoxRunner.Run(owner, memory.Prompt, resume, model, token))
            foreach (var item in memory.Consume(raw))
            {
                if (item is HarnessFinished { IsError: false }) finished = true;
                switch (item)
                {
                    case HarnessStarted begun:
                        started = true; owner.HarnessSessions[PrivateHarness.ClaudeCode] = begun.SessionId; UpdateContext(); break;
                    case HarnessText or HarnessTextDelta or HarnessTextSnapshot:
                        RenderText(owner, item); break;
                    case HarnessToolCall call:
                        // Titled like macOS ("Open a webpage"); the details keep the Box tool and its input.
                        var activity = new ChatEntry { Role = "Activity", Tool = call.Name, Text = call.Summary == call.Name ? "" : call.Summary, Input = call.Detail };
                        calls[call.Id] = activity; Add(owner, activity); break;
                    case HarnessToolResult result when calls.TryGetValue(result.Id, out var step):
                        step.Output = result.Output;
                        if (result.IsError) step.Failed = true; else step.Done = true;
                        Change(step); break;
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
        finally
        {
            FinishMemory(owner, memory, finished && !reportedError && !token.IsCancellationRequested);
            screens.Cancel();
            try { await watching; } catch (OperationCanceledException) { }
        }
        if (selected == owner) await RefreshBoxScreen();
    }

    /// <summary>Refreshes the Box's screen while its agent works, so the person can watch.</summary>
    private async Task WatchBoxScreen(Conversation owner, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(4000, token);
                if (selected != owner || !panelOpen || BoxRunner.Client() is not { } box) continue;
                if (await box.Screenshot(owner.BoxName, token) is { } png && selected == owner)
                {
                    boxState = "Running";
                    await ShowPrivateScreen(png, ScreenKindFor(owner)!);
                }
            }
            catch (OperationCanceledException) { return; }
            catch (Exception) { /* A missed frame is fine; the next one follows. */ }
        }
    }

    private static string ShortPc(string text) => text.Replace(ThisPcAgent.ToolCommand(ThisPcRunner.PcToolPath) + " ", "pc ", StringComparison.Ordinal);

    /// <summary>The This PC agent whose turn is running; approval requests for any other agent are denied.</summary>
    private Conversation? pcTurnOwner;

    /// <summary>Called by the approval hook (off the UI thread) for each This PC tool use.</summary>
    private Task<bool> Decide(ApprovalRequest request, CancellationToken cancellation)
    {
        var answer = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var registration = cancellation.Register(() => answer.TrySetResult(false));
        _ = answer.Task.ContinueWith(_ => registration.Dispose(), TaskScheduler.Default);
        if (!DispatcherQueue.TryEnqueue(async () =>
        {
            var owner = pcTurnOwner;
            if (owner is null || owner.Id != request.Agent || !running) { answer.TrySetResult(false); return; }
            // Wait before approving the next action while the person takes over or edits in the floating chat.
            await WaitWhilePaused(cancellation);
            if (pcTurnOwner != owner || !running || stopRequested || cancellation.IsCancellationRequested) { answer.TrySetResult(false); return; }
            bool uiWork = ThisPcAgent.IsUICommand(request.Tool, request.Input, ThisPcRunner.PcToolPath);
            // Looking runs without asking; so does the rest of a turn the person approved as a whole.
            if (ThisPcAgent.IsLookOnly(request.Tool, request.Input, ThisPcRunner.PcToolPath) || approveRestOfTurn)
            {
                // Working in the PC's apps keeps the Agent Activity window on top, as on macOS.
                if (uiWork) ShowActivity(owner, manual: false);
                answer.TrySetResult(true);
                return;
            }
            var entry = new ChatEntry { Role = "Needs you", Text = ShortPc(ThisPcAgent.Describe(request.Tool, request.Input)) };
            entry.LocalAnswer = allowed => answer.TrySetResult(allowed);
            Add(owner, entry);
            // An approval for app work opens the window too, so the person can answer next to the app.
            if (uiWork) ShowActivity(owner, manual: false);
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
        var provider = owner.Provider;
        var runModel = provider == ModelProvider.OpenRouter ? model : await HarnessRunner.ContextModel(model, token);
        status.Text = $"{harness.DisplayName()} · {(provider == ModelProvider.OpenRouter ? "OpenRouter · " : "")}{model}"; Change(status);

        using var screens = CancellationTokenSource.CreateLinkedTokenSource(token);
        var watching = WatchScreen(computer, screens.Token);
        var calls = new Dictionary<string, ChatEntry>();
        var resume = owner.HarnessSessions.GetValueOrDefault(harness);
        bool started = false, reportedError = false, finished = false;
        var memory = BeginMemory(owner, message, currentUserEntryId);
        try
        {
            await foreach (var raw in HarnessRunner.Run(harness, memory.Prompt, resume, runModel, provider, token))
            foreach (var item in memory.Consume(raw))
            {
                if (item is HarnessFinished { IsError: false }) finished = true;
                switch (item)
                {
                    case HarnessStarted begun:
                        started = true; owner.HarnessSessions[harness] = begun.SessionId; UpdateContext(); break;
                    case HarnessText or HarnessTextDelta or HarnessTextSnapshot:
                        RenderText(owner, item); break;
                    case HarnessToolCall call:
                        var activity = new ChatEntry { Role = "Activity", Tool = call.Name, Text = call.Summary, Input = call.Detail };
                        calls[call.Id] = activity; Add(owner, activity); break;
                    case HarnessToolResult result when calls.TryGetValue(result.Id, out var step):
                        step.Output = result.Output;
                        if (result.IsError) step.Failed = true; else step.Done = true;
                        Change(step); break;
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
            FinishMemory(owner, memory, finished && !reportedError && !token.IsCancellationRequested);
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
        if (cancellation is null || !running || runOwner != owner) return;
        if (owner.Target is ConversationTarget.Private or ConversationTarget.ThisPc or ConversationTarget.Box)
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
        if (!running || runOwner is not { } owner) return;
        stopRequested = true; Controls();
        // A paused This PC turn stops too; its waiting step is denied.
        SetPcPaused(false);
        if (owner.Target == ConversationTarget.Sai && currentSession is null) Show("Waiting for the session ID before sending the stop request…", false);
        else await Stop(owner);
    }
    private async void OpenComputer(object sender, RoutedEventArgs e)
    {
        if (IsThisPc) return;
        if (IsBox)
        {
            // The live view opens in the browser, where the person is signed in to box.simular.ai.
            if (busy || selected is not { } owner || BoxRunner.Client() is not { } box) return;
            try
            {
                var (_, view) = await box.Status(owner.BoxName);
                if (view is null) Show("Simular Box didn't return a live view link. Open box.simular.ai instead.");
                else if (!await Launcher.LaunchUriAsync(view)) Show("Could not open the live view in your browser.");
            }
            catch (Exception error) when (error is BoxException or TaskCanceledException or HttpRequestException) { Error(error); }
            return;
        }
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
        if (IsBox)
        {
            if (busy || selected is not { } owner || BoxRunner.Client() is not { } box) return;
            busy = true; Controls();
            try { await box.Wake(owner.BoxName); boxState = "Running"; Show("Simular Box is waking up.", false); }
            catch (Exception error) when (error is BoxException or TaskCanceledException or HttpRequestException) { Error(error); }
            finally { busy = false; Controls(); }
            await RefreshBoxScreen();
            return;
        }
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
        if (busy || (running && runOwner?.Target == ConversationTarget.Private)) { if (!busy) Show("Wait for the current turn to finish, or stop it first."); return; }
        busy = true; Controls();
        try { await use(new PrivateComputer()); Notice.IsOpen = false; }
        catch (Exception error) { Error(error); }
        finally { busy = false; Controls(); }
    }

    private async Task ShowPrivateScreen(byte[] png, string kind = "private")
    {
        bool fromPc = kind != "private";
        screenKind = kind;
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
        if (restartingForUpdate && !allowClose) { args.Cancel = true; return; }
        if (allowClose || !running || runOwner is not { } owner) return;
        args.Cancel = true;
        if (closing) return;
        closing = true;
        bool local = owner.Target != ConversationTarget.Sai;
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
                if (runOwner is null) return;
                await Stop(runOwner);
                if (!stopped && pending == run) return;
            }
            await Persist(); allowClose = true; Close();
        }
        finally { closing = false; }
    }
}
