using System.Collections.ObjectModel;
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

    public MainWindow()
    {
        InitializeComponent();
        Agents.ItemsSource = conversations;
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        var width = Math.Min(1120, area.Width);
        var height = Math.Min(820, area.Height);
        AppWindow.Resize(new global::Windows.Graphics.SizeInt32(width, height));
        AppWindow.Move(new global::Windows.Graphics.PointInt32(area.X + (area.Width - width) / 2,
            area.Y + (area.Height - height) / 2));
        AppWindow.Closing += Closing;
        Closed += (_, _) => { run?.Cancel(); client?.Dispose(); };
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            foreach (var conversation in await store.Read()) conversations.Add(conversation);
            storageReady = true;
            Agents.SelectedItem = conversations.FirstOrDefault(item => item.Target == ConversationTarget.Private)
                ?? conversations.FirstOrDefault();
            var key = Secrets.Read();
            if (key is not null)
            {
                client = new SaiClient(key);
                if (selected?.Target == ConversationTarget.Sai) await Refresh();
            }
            if (selected is null && !PrivateComputer.IsConfigured)
                Show("Set up the Private computer to start using Agent S locally. See the Windows README.", false);
        }
        catch (Exception error) { Error(error); }
        Controls();
    }

    private bool IsPrivate => selected?.Target == ConversationTarget.Private;

    private void Controls()
    {
        Agents.IsEnabled = AccountsButton.IsEnabled = !busy;
        DeleteButton.IsEnabled = !busy && selected is not null;
        NewAgentButton.IsEnabled = !busy && storageReady;
        Composer.IsEnabled = !busy && selected is not null;
        SendButton.IsEnabled = !busy && storageReady && (selected?.Target switch
        {
            ConversationTarget.Private => true,
            ConversationTarget.Sai => client is not null,
            _ => false,
        });
        StopButton.IsEnabled = running && !stopRequested;
        ScreenButton.Content = IsPrivate
            ? PrivatePreview.Visibility == Visibility.Visible ? "Hide computer" : "Show computer"
            : "Open computer";
        ScreenButton.IsEnabled = IsPrivate ? !busy || running : !busy && CurrentComputer is not null;
        WakeButton.Content = IsPrivate ? "Stop computer" : "Wake computer";
        WakeButton.IsEnabled = !busy && (IsPrivate ? PrivateComputer.IsConfigured : CurrentComputer?.CanWake == true);
        RefreshButton.Content = IsPrivate ? "Reset computer" : "Refresh computers";
        RefreshButton.IsEnabled = !busy && (IsPrivate ? PrivateComputer.IsConfigured : client is not null);
        Heading.Text = selected?.Name ?? "Choose an agent or create one";
        Status.Text = IsPrivate
            ? running ? "Private agent is working…" : $"Private · local Ollama · {selected?.ModelId}"
            : running ? "Sai is working…" : CurrentComputer is { } c ? $"{c.Name} · {c.Availability} · Sai cloud" : "Private computer · Sai cloud available in Accounts";
    }
    private Computer? CurrentComputer => selected?.Target == ConversationTarget.Sai
        ? computers.FirstOrDefault(c => c.Id == selected.ComputerId)
        : null;
    private void Show(string text, bool error = true)
    { Notice.Message = text; Notice.Severity = error ? InfoBarSeverity.Error : InfoBarSeverity.Informational; Notice.IsOpen = true; }
    private void Error(Exception error) => Show(error is SaiException or PrivateComputerException or LocalAgentException ? error.Message :
        error is OperationCanceledException ? "The request timed out or was interrupted. Its remote result is not confirmed." :
        "The operation failed. Check your connection, credential access and local storage, then retry.");
    private async Task Persist()
    {
        if (!storageReady) return;
        try { await store.Save(conversations); }
        catch { Show("Could not save conversation history. Keep this window open to avoid losing the current conversation."); }
    }
    private ContentDialog Dialog(string title, object content, string primary) => new()
    { Title = title, Content = content, PrimaryButtonText = primary, CloseButtonText = "Cancel", XamlRoot = Root.XamlRoot };

    private async Task Refresh()
    {
        if (client is null) return;
        busy = true; Controls();
        try
        {
            var nextComputers = await client.Computers();
            var nextModels = await client.Models();
            computers = nextComputers; models = nextModels.Where(m => m.Allowed).ToArray();
            Notice.IsOpen = false;
        }
        catch (Exception error) { Error(error); }
        finally { busy = false; Controls(); }
    }
    private async void RefreshComputers(object sender, RoutedEventArgs e)
    {
        if (!IsPrivate) { await Refresh(); return; }
        if (busy) return;
        var dialog = Dialog("Reset the Private computer?",
            "This erases guest changes since setup. Conversation history on Windows is kept.", "Reset computer");
        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        await UsePrivateComputer(async computer =>
        {
            await computer.Reset();
            PrivateScreen.Source = null;
            PrivatePreview.Visibility = Visibility.Collapsed;
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
        var dialog = Dialog("Accounts · Sai", content, "Save key");
        dialog.SecondaryButtonText = "Remove key";
        var result = await dialog.ShowAsync();
        try
        {
            if (result == ContentDialogResult.Primary)
            {
                if (string.IsNullOrWhiteSpace(password.Password)) { Show("Enter a Sai API key."); return; }
                Secrets.Save(password.Password.Trim());
                client?.Dispose(); client = new SaiClient(password.Password.Trim());
            }
            else if (result == ContentDialogResult.Secondary) { Secrets.Delete(); client?.Dispose(); client = null; }
            else return;
            // A different credential must never reuse the prior account's sessions or live links.
            foreach (var conversation in conversations.Where(item => item.Target == ConversationTarget.Sai))
                conversation.SessionId = null;
            liveLinks.Clear(); computers = []; models = [];
            await Persist();
            if (client is not null) await Refresh();
        }
        catch (Exception error) { Error(error); }
        finally { password.Password = ""; Controls(); }
    }

    private async void NewAgent(object sender, RoutedEventArgs e)
    {
        if (busy || !storageReady) return;
        var name = new TextBox { Header = "Name", Text = "Private agent", MaxLength = 100 };
        var target = new ComboBox
        {
            Header = "Computer",
            ItemsSource = new[] { "Private · local Linux VM", "Cloud · Sai computer" },
            SelectedIndex = 0, HorizontalAlignment = HorizontalAlignment.Stretch
        };
        var localModel = new TextBox { Header = "Local Ollama model tag", Text = "gemma4", MaxLength = 128 };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(localModel, "PrivateModel");
        var computer = new ComboBox { Header = "Sai computer", HorizontalAlignment = HorizontalAlignment.Stretch };
        var model = new ComboBox { Header = "Sai model", HorizontalAlignment = HorizontalAlignment.Stretch };
        var cloudStatus = new TextBlock { TextWrapping = TextWrapping.Wrap };
        var cloud = new StackPanel { Spacing = 12, Visibility = Visibility.Collapsed };
        cloud.Children.Add(computer); cloud.Children.Add(model); cloud.Children.Add(cloudStatus);
        var form = new StackPanel { Spacing = 12 };
        form.Children.Add(name); form.Children.Add(target); form.Children.Add(localModel); form.Children.Add(cloud);
        var dialog = Dialog("New agent", form, "Create");
        bool loadingCloud = false;

        void SetMode()
        {
            bool useCloud = target.SelectedIndex == 1;
            localModel.Visibility = useCloud ? Visibility.Collapsed : Visibility.Visible;
            cloud.Visibility = useCloud ? Visibility.Visible : Visibility.Collapsed;
            dialog.IsPrimaryButtonEnabled = !useCloud || (!loadingCloud && computer.SelectedItem is Computer);
        }
        void SetCloudChoices()
        {
            computer.ItemsSource = computers;
            computer.SelectedIndex = computers.Length > 0 ? 0 : -1;
            model.ItemsSource = new[] { new SaiModel("auto", "Automatic", true) }
                .Concat(models.Where(item => item.Id != "auto")).ToArray();
            model.SelectedIndex = 0;
            cloudStatus.Text = client is null ? "Add a Sai API key in Accounts to use cloud computers."
                : computers.Length == 0 ? "Could not load Sai computers. Check Accounts and try again." : "";
        }
        computer.SelectionChanged += (_, _) => SetMode();
        target.SelectionChanged += async (_, _) =>
        {
            SetMode();
            if (target.SelectedIndex != 1 || client is null || computers.Length != 0 || loadingCloud) return;
            loadingCloud = true;
            SetMode();
            cloudStatus.Text = "Loading Sai computers…";
            try { await Refresh(); }
            finally { loadingCloud = false; SetCloudChoices(); SetMode(); }
        };
        SetCloudChoices();
        SetMode();

        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        if (string.IsNullOrWhiteSpace(name.Text)) { Show("Give the agent a name."); return; }
        var conversation = new Conversation { Name = name.Text.Trim() };
        if (target.SelectedIndex == 0)
        {
            if (string.IsNullOrWhiteSpace(localModel.Text)) { Show("Enter a local Ollama model tag."); return; }
            conversation.Target = ConversationTarget.Private;
            conversation.ModelId = localModel.Text.Trim();
        }
        else
        {
            if (computer.SelectedItem is not Computer choice) return;
            conversation.Target = ConversationTarget.Sai;
            conversation.ComputerId = choice.Id;
            conversation.ModelId = (model.SelectedItem as SaiModel)?.Id;
        }
        conversations.Add(conversation); Agents.SelectedItem = conversation; await Persist();
    }
    private async void SelectAgent(object sender, SelectionChangedEventArgs e)
    {
        selected = Agents.SelectedItem as Conversation;
        Composer.Text = ""; Messages.Children.Clear(); entryText.Clear(); approvalControls.Clear();
        PrivatePreview.Visibility = Visibility.Collapsed; PrivateScreen.Source = null;
        if (selected is not null) foreach (var entry in selected.Entries) Draw(entry);
        Controls();
        if (!busy && selected?.Target == ConversationTarget.Sai && client is not null && computers.Length == 0)
            await Refresh();
    }
    private async void DeleteAgent(object sender, RoutedEventArgs e)
    {
        if (busy || selected is null) return;
        if (await Dialog("Delete agent?", "This deletes the local conversation. It does not delete the Sai computer or remote task history.", "Delete").ShowAsync() != ContentDialogResult.Primary) return;
        conversations.Remove(selected); Agents.SelectedItem = conversations.FirstOrDefault(); await Persist();
    }

    private static void SetButtonsEnabled(StackPanel panel, bool enabled)
    {
        foreach (var control in panel.Children.OfType<Control>()) control.IsEnabled = enabled;
    }

    private void Draw(ChatEntry entry)
    {
        var panel = new StackPanel { Spacing = 6 };
        panel.Children.Add(new TextBlock { Text = entry.Role, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
        var text = new TextBlock { Text = entry.Text, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true };
        panel.Children.Add(text); entryText[entry.Id] = text;
        if (entry.ApprovalId is not null && !entry.Answered)
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
        Messages.Children.Add(panel);
        if (following) DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
    }
    private void Add(Conversation owner, ChatEntry entry)
    {
        owner.Entries.Add(entry);
        if (selected == owner) Draw(entry);
    }
    private void Change(ChatEntry entry)
    {
        var following = Transcript.ScrollableHeight - Transcript.VerticalOffset < 64;
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
        if (!local && (owner.Target != ConversationTarget.Sai || client is null)) return;
        if (!local && CurrentComputer is null) { Show("The saved computer is not available on this account. Refresh computers or create a new agent."); return; }
        if (local && !PrivateComputer.IsConfigured)
        {
            Show("Set up the Private computer with scripts/setup-private-vm.ps1. See the Windows README.");
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
            if (local)
            {
                var computer = new PrivateComputer();
                using var agent = new LocalAgent();
                PrivatePreview.Visibility = Visibility.Visible;
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
        { Show(local ? "Private turn stopped." : "Sai confirmed the stop request.", false); }
        catch (Exception error) { Error(error); }
        finally
        {
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

    private async Task Stop(Conversation owner)
    {
        var cancellation = run;
        if (cancellation is null || !running || selected != owner) return;
        if (owner.Target == ConversationTarget.Private)
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
        if (!IsPrivate && currentSession is null) Show("Waiting for the session ID before sending the stop request…", false);
        else await Stop(selected);
    }
    private async void OpenComputer(object sender, RoutedEventArgs e)
    {
        if (IsPrivate)
        {
            bool show = PrivatePreview.Visibility == Visibility.Collapsed;
            PrivatePreview.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
            Controls();
            if (show && !busy)
                await UsePrivateComputer(async computer =>
                {
                    await computer.Start(CancellationToken.None);
                    await ShowPrivateScreen(await computer.Capture(CancellationToken.None));
                });
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
                PrivateScreen.Source = null;
                PrivatePreview.Visibility = Visibility.Collapsed;
            });
            return;
        }
        if (busy || CurrentComputer is not { CanWake: true } computer || client is null) return;
        busy = true; Controls();
        try { await client.Wake(computer.Id); Show("Sai confirmed the wake request. Refresh to check startup progress.", false); }
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

    private async Task ShowPrivateScreen(byte[] png)
    {
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
    }

    private async void Closing(AppWindow sender, AppWindowClosingEventArgs args)
    {
        if (allowClose || !running || selected is null) return;
        args.Cancel = true;
        if (closing) return;
        closing = true;
        bool local = IsPrivate;
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
