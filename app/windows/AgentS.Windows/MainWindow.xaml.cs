using System.Collections.ObjectModel;
using System.Text.Json;
using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Windowing;
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
        AppWindow.Resize(new global::Windows.Graphics.SizeInt32(1120, 820));
        AppWindow.Closing += Closing;
        Closed += (_, _) => { run?.Cancel(); client?.Dispose(); };
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        try
        {
            foreach (var conversation in await store.Read()) conversations.Add(conversation);
            storageReady = true;
            Agents.SelectedItem = conversations.FirstOrDefault();
            var key = Secrets.Read();
            if (key is not null) { client = new SaiClient(key); await Refresh(); }
            else Show("Add your Sai API key in Accounts to get started.", false);
        }
        catch (Exception error) { Error(error); }
        Controls();
    }

    private void Controls()
    {
        Agents.IsEnabled = AccountsButton.IsEnabled = DeleteButton.IsEnabled = !busy;
        NewAgentButton.IsEnabled = !busy && client is not null && storageReady;
        Composer.IsEnabled = !busy && selected is not null;
        SendButton.IsEnabled = !busy && selected is not null && client is not null && storageReady;
        StopButton.IsEnabled = running && !stopRequested;
        RefreshButton.IsEnabled = !busy && client is not null;
        ScreenButton.IsEnabled = !busy && CurrentComputer is not null;
        WakeButton.IsEnabled = !busy && CurrentComputer?.CanWake == true;
        Heading.Text = selected?.Name ?? "Choose an agent or create one";
        Status.Text = running ? "Sai is working…" : CurrentComputer is { } c ? $"{c.Name} · {c.Availability} · Sai cloud" : "Cloud mode · Sai API";
    }
    private Computer? CurrentComputer => computers.FirstOrDefault(c => c.Id == selected?.ComputerId);
    private void Show(string text, bool error = true)
    { Notice.Message = text; Notice.Severity = error ? InfoBarSeverity.Error : InfoBarSeverity.Informational; Notice.IsOpen = true; }
    private void Error(Exception error) => Show(error is SaiException ? error.Message :
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
    private async void RefreshComputers(object sender, RoutedEventArgs e) => await Refresh();

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
            foreach (var conversation in conversations) conversation.SessionId = null;
            liveLinks.Clear(); computers = []; models = [];
            await Persist();
            if (client is not null) await Refresh();
        }
        catch (Exception error) { Error(error); }
        finally { password.Password = ""; Controls(); }
    }

    private async void NewAgent(object sender, RoutedEventArgs e)
    {
        if (busy || client is null) return;
        if (!computers.Any()) { Show("No computers loaded. Refresh computers, or add one on your Sai account.", false); return; }
        var name = new TextBox { Header = "Name", Text = "Sai agent", MaxLength = 100 };
        var computer = new ComboBox { Header = "Sai computer", ItemsSource = computers, SelectedIndex = 0, HorizontalAlignment = HorizontalAlignment.Stretch };
        var modelChoices = new[] { new SaiModel("auto", "Automatic", true) }.Concat(models.Where(m => m.Id != "auto")).ToArray();
        var model = new ComboBox { Header = "Model", ItemsSource = modelChoices, SelectedIndex = 0, HorizontalAlignment = HorizontalAlignment.Stretch };
        var form = new StackPanel { Spacing = 12 }; form.Children.Add(name); form.Children.Add(computer); form.Children.Add(model);
        form.Children.Add(new TextBlock { Text = "This agent runs through Sai on the selected computer. Local Windows and private Linux modes are not included yet.", TextWrapping = TextWrapping.Wrap });
        if (await Dialog("New agent", form, "Create").ShowAsync() != ContentDialogResult.Primary) return;
        if (string.IsNullOrWhiteSpace(name.Text)) { Show("Give the agent a name."); return; }
        var conversation = new Conversation { Name = name.Text.Trim(), ComputerId = ((Computer)computer.SelectedItem).Id, ModelId = ((SaiModel)model.SelectedItem).Id };
        conversations.Add(conversation); Agents.SelectedItem = conversation; await Persist();
    }
    private void SelectAgent(object sender, SelectionChangedEventArgs e)
    {
        selected = Agents.SelectedItem as Conversation;
        Composer.Text = ""; Messages.Children.Clear(); entryText.Clear(); approvalControls.Clear();
        if (selected is not null) foreach (var entry in selected.Entries) Draw(entry);
        Controls();
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
                        catch (Exception error) { Error(error); SetButtonsEnabled(buttons, true); }
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
        if (busy || selected is null || client is null || string.IsNullOrWhiteSpace(Composer.Text)) return;
        if (CurrentComputer is null) { Show("The saved computer is not available on this account. Refresh computers or create a new agent."); return; }
        var owner = selected;
        var message = Composer.Text.Trim(); Composer.Text = "";
        busy = running = true; stopped = stopRequested = false; Notice.IsOpen = false;
        currentSession = null;
        turn = Guid.NewGuid().ToString("N"); run = new CancellationTokenSource(); Controls();
        Add(owner, new ChatEntry { Role = "You", Text = message });
        DispatcherQueue.TryEnqueue(() => Transcript.ChangeView(null, Transcript.ScrollableHeight, null, true));
        await Persist();
        try
        {
            await foreach (var value in client.Run(message, owner.ComputerId, owner.SessionId, owner.ModelId, run.Token)) await Apply(owner, value);
            if (stopRequested && !stopped) Show("The stream ended without confirming that the remote task stopped.");
        }
        catch (OperationCanceledException) when (stopped) { Show("Sai confirmed the stop request.", false); }
        catch (Exception error) { Error(error); }
        finally
        {
            foreach (var entry in owner.Entries.Where(entry => entry.ApprovalId is not null && !entry.Answered))
            { entry.Answered = true; entry.Text += "\nTurn disconnected. Check pending approvals in Sai."; Change(entry); }
            // Disable remaining controls without silently approving or denying anything.
            running = busy = false; run.Dispose(); run = null; Controls(); await Persist();
            foreach (var controls in approvalControls.Values) SetButtonsEnabled(controls, false);
        }
    }
    private async Task Stop(Conversation owner)
    {
        if (client is null || currentSession is null) return;
        try { await client.Abort(currentSession); stopped = true; run?.Cancel(); }
        catch (Exception error) { stopRequested = false; Error(error); Controls(); }
    }
    private async void StopTurn(object sender, RoutedEventArgs e)
    {
        if (!running || selected is null) return;
        stopRequested = true; Controls();
        if (currentSession is null) Show("Waiting for the session ID before sending the stop request…", false);
        else await Stop(selected);
    }
    private async void OpenComputer(object sender, RoutedEventArgs e)
    {
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
        if (busy || CurrentComputer is not { CanWake: true } computer || client is null) return;
        busy = true; Controls();
        try { await client.Wake(computer.Id); Show("Sai confirmed the wake request. Refresh to check startup progress.", false); }
        catch (Exception error) { Error(error); }
        finally { busy = false; Controls(); }
    }
    private async void Closing(AppWindow sender, AppWindowClosingEventArgs args)
    {
        if (allowClose || !running || selected is null) return;
        args.Cancel = true;
        if (closing) return;
        closing = true;
        var hasSession = currentSession is not null;
        var dialog = Dialog("A remote task is running", hasSession ? "Stop it before closing?" : "The session ID has not arrived. Closing disconnects this app; the remote task may continue. Check it in Sai.", hasSession ? "Stop and close" : "Close anyway");
        try
        {
            if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
            if (hasSession) { await Stop(selected); if (!stopped) return; }
            await Persist(); allowClose = true; Close();
        }
        finally { closing = false; }
    }
}
