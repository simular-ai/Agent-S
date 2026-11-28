using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using global::Windows.ApplicationModel.DataTransfer;

namespace AgentS.Windows;

/// <summary>
/// The macOS sidebar's agent actions: Agent Settings…, Rename…, Color, New conversation, drag to reorder
/// (with Move up / Move down), the inference badge, and "New … agent…" links in empty sections.
/// </summary>
public sealed partial class MainWindow
{
    private bool preferPc;
    private InstalledModel[]? ollamaInstalled;
    private DateTime ollamaCheckedAt;

    private static Conversation? RowOf(object sender) => ((sender as FrameworkElement)?.Tag as AgentRow)?.Conversation;

    private void NewAgentForGroup(object sender, RoutedEventArgs e)
    {
        switch ((sender as FrameworkElement)?.Tag as string)
        {
            case "private": preferPrivate = true; break;
            case "pc": preferPc = true; break;
            case "cloud": preferCloud = true; break;
        }
        NewAgent(sender, e);
    }

    // ---- Rename, color, new conversation ----------------------------------------------------------

    private async void RenameFromMenu(object sender, RoutedEventArgs e)
    {
        if (RowOf(sender) is not { } agent || busy) return;
        var name = new TextBox { Text = agent.Name, MaxLength = 100, SelectionStart = 0, SelectionLength = agent.Name.Length };
        var dialog = Dialog("Rename agent", name, "Rename");
        dialog.DefaultButton = ContentDialogButton.Primary;
        if (await dialog.ShowAsync() != ContentDialogResult.Primary || name.Text.Trim() is not { Length: > 0 } chosen) return;
        agent.Name = chosen;
        Row(agent).Refresh(); Controls();
        await Persist();
    }

    private async void ColorFromMenu(object sender, RoutedEventArgs e)
    {
        if (RowOf(sender) is not { } agent || sender is not MenuFlyoutItem item) return;
        agent.Color = AgentRow.Colors.Any(c => c.Name == item.Text) ? item.Text : null;
        Row(agent).Refresh();
        await Persist();
    }

    /// <summary>Clears the chat and starts a new harness session, keeping the agent's settings (macOS "New conversation").</summary>
    private async void NewConversationFromMenu(object sender, RoutedEventArgs e)
    {
        if (RowOf(sender) is { } agent) await NewConversation(agent);
    }

    private async Task NewConversation(Conversation agent)
    {
        if (agent == runOwner) { Show("Wait for the current turn to finish, or stop it first."); return; }
        agent.Entries.Clear(); agent.SessionId = null; agent.HarnessSessions.Clear();
        if (followUps.TryGetValue(agent, out var queue)) { queue.Items.Clear(); queue.Paused = false; }
        Row(agent).Refresh();
        if (selected == agent)
        {
            Messages.Children.Clear(); entryText.Clear(); entryMarkdown.Clear(); approvalControls.Clear(); stepGroups.Clear(); openGroup = null;
            usage = null;
        }
        if (activityOwner == agent && activityChat is not null) { activityChat.Clear(); RefreshActivity(); }
        Controls();
        await Persist();
    }

    // ---- Reorder ----------------------------------------------------------------------------------

    private Conversation? dragged;

    private void AgentDragStarting(object sender, DragItemsStartingEventArgs e)
    {
        dragged = (e.Items.FirstOrDefault() as AgentRow)?.Conversation;
        e.Data.RequestedOperation = DataPackageOperation.Move;
        e.Data.SetText(dragged?.Name ?? "");
    }

    /// <summary>Rows move only within their own section.</summary>
    private static int Section(Conversation c) => c.Target switch
    {
        ConversationTarget.Private => 0,
        ConversationTarget.ThisPc => 1,
        _ => 2,
    };

    private void RowDragOver(object sender, DragEventArgs e)
    {
        if (dragged is null || RowOf(sender) is not { } target || target == dragged || Section(target) != Section(dragged)) { e.AcceptedOperation = DataPackageOperation.None; return; }
        e.AcceptedOperation = DataPackageOperation.Move;
        e.DragUIOverride.Caption = "Move here";
    }

    private async void RowDrop(object sender, DragEventArgs e)
    {
        if (dragged is not { } moving || RowOf(sender) is not { } target || target == moving || Section(target) != Section(moving)) return;
        dragged = null;
        await MoveAgent(moving, conversations.IndexOf(target));
    }

    private async void MoveFromMenu(object sender, RoutedEventArgs e)
    {
        if (RowOf(sender) is not { } agent || sender is not MenuFlyoutItem item) return;
        var section = conversations.Where(c => Section(c) == Section(agent)).ToList();
        var at = section.IndexOf(agent) + (item.Text == "Move up" ? -1 : 1);
        if (at < 0 || at >= section.Count) return;
        await MoveAgent(agent, conversations.IndexOf(section[at]));
    }

    private async Task MoveAgent(Conversation agent, int index)
    {
        var from = conversations.IndexOf(agent);
        if (from < 0 || index < 0 || from == index) return;
        conversations.Move(from, index);
        Select(selected);
        await Persist();
    }

    // ---- Inference badge --------------------------------------------------------------------------

    /// <summary>
    /// As on macOS: "Hosted inference" (Sai, OpenRouter, Ollama cloud models), "Local inference" (an Ollama
    /// model installed on this PC), or "Inference location unconfirmed" (Ollama isn't answering, or the model isn't there).
    /// </summary>
    private async void UpdateBadges()
    {
        if (DateTime.UtcNow - ollamaCheckedAt > TimeSpan.FromSeconds(30))
        {
            ollamaCheckedAt = DateTime.UtcNow;
            ollamaInstalled = await OllamaSetup.Installed(TimeSpan.FromSeconds(2));
        }
        foreach (var agent in conversations)
        {
            string detail = agent.Target == ConversationTarget.Sai ? $"Sai · {(agent.ModelId is null or "auto" ? "auto" : agent.ModelId)}"
                : agent.Provider == ModelProvider.OpenRouter ? $"OpenRouter · {agent.ModelId}" : $"Ollama · {agent.ModelId ?? OllamaModels.Default}";
            (string Glyph, string Label) badge;
            if (agent.Target == ConversationTarget.Sai || agent.Provider == ModelProvider.OpenRouter) badge = ("", "Hosted inference");
            else
            {
                var model = agent.ModelId ?? OllamaModels.Default;
                var installed = ollamaInstalled?.FirstOrDefault(m => OllamaModels.Same(m.Name, model));
                badge = OllamaModels.IsCloudReference(model) || installed is { Remote: true } ? ("", "Hosted inference")
                    : installed is not null ? ("", "Local inference")
                    : ("", "Inference location unconfirmed");
            }
            Row(agent).SetBadge(badge.Glyph, $"{badge.Label} · {detail}");
        }
    }

    // ---- Agent Settings… --------------------------------------------------------------------------

    private void AgentSettingsFromMenu(object sender, RoutedEventArgs e)
    {
        if (RowOf(sender) is { } agent) _ = AgentSettings(agent);
    }

    /// <summary>The macOS Agent Settings sheet: name, where it works, harness, provider and model, then New conversation or Done.</summary>
    private async Task AgentSettings(Conversation agent)
    {
        if (busy) return;
        bool isRunning = agent == runOwner;
        var form = new StackPanel { Spacing = 14, MinWidth = 440 };
        var head = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 12 };
        head.Children.Add(new Border
        {
            Width = 44, Height = 44, CornerRadius = new CornerRadius(22), Background = Row(agent).Avatar,
            Child = new TextBlock { Text = Row(agent).Initial, Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.White), FontSize = 18, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center },
        });
        var name = new TextBox { Header = "Name", Text = agent.Name, MaxLength = 100, MinWidth = 340 };
        head.Children.Add(name);
        form.Children.Add(head);
        form.Children.Add(new TextBlock { Text = "Works on", Style = (Style)Application.Current.Resources["SectionLabel"] });
        form.Children.Add(new TextBlock
        {
            Text = agent.Target switch
            {
                ConversationTarget.Private => "Private · Linux computer on this PC",
                ConversationTarget.ThisPc => "This PC · your apps and files",
                ConversationTarget.Box => $"Cloud · Simular Box · {agent.BoxName ?? "default box"}",
                _ => $"Cloud · Sai computer · {computers.FirstOrDefault(c => c.Id == agent.ComputerId)?.Name ?? "unknown"}",
            },
            TextWrapping = TextWrapping.Wrap,
        });
        if (isRunning) form.Children.Add(Muted("Changes to the model apply after the current answer."));

        form.Children.Add(new TextBlock { Text = "Model", Style = (Style)Application.Current.Resources["SectionLabel"] });
        ComboBox? harness = null, provider = null, ollamaModel = null, saiComputer = null, saiModel = null;
        ComboBox? openRouterModel = null;
        TextBox? boxName = null;
        OllamaModelPanel? panel = null;
        if (agent.Target == ConversationTarget.Sai)
        {
            if (client is not null && computers.Length == 0) await Refresh();
            saiComputer = new ComboBox { Header = "Computer", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = computers, SelectedItem = computers.FirstOrDefault(c => c.Id == agent.ComputerId) };
            var choices = new[] { new SaiModel("auto", "Automatic (recommended)") }.Concat(models.Where(m => m.Id != "auto")).ToArray();
            saiModel = new ComboBox { Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = choices, SelectedItem = choices.FirstOrDefault(m => m.Id == (agent.ModelId ?? "auto")) ?? choices[0] };
            form.Children.Add(saiComputer); form.Children.Add(saiModel);
        }
        else
        {
            var harnesses = agent.Target switch
            {
                ConversationTarget.Private => Harnesses.All,
                ConversationTarget.ThisPc => Harnesses.ThisPc,
                _ => [PrivateHarness.ClaudeCode],
            };
            harness = new ComboBox { Header = "Harness", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = harnesses.Select(h => h.DisplayName()).ToArray(),
                SelectedIndex = Math.Max(0, Array.IndexOf(harnesses, agent.Harness ?? (agent.Target == ConversationTarget.Private ? PrivateHarness.AgentS : PrivateHarness.ClaudeCode))),
                IsEnabled = harnesses.Length > 1 };
            provider = new ComboBox { Header = "Provider", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = new[] { "Ollama", "OpenRouter" }, SelectedIndex = agent.Provider == ModelProvider.OpenRouter ? 1 : 0 };
            var installed = (await OllamaSetup.Installed(TimeSpan.FromSeconds(2)) ?? []).Where(m => !m.Remote).Select(m => OllamaModels.NormalizedName(m.Name)).ToList();
            foreach (var suggested in OllamaModels.Suggested) if (!installed.Any(m => OllamaModels.Same(m, suggested))) installed.Add(suggested);
            var currentOllama = agent.Provider == ModelProvider.Ollama ? agent.ModelId ?? OllamaModels.Default : OllamaModels.Default;
            if (!installed.Any(m => OllamaModels.Same(m, currentOllama))) installed.Insert(0, currentOllama);
            ollamaModel = new ComboBox { Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = installed.ToArray(), SelectedItem = installed.First(m => OllamaModels.Same(m, currentOllama)) };
            panel = new OllamaModelPanel(currentOllama, () => agent.Target == ConversationTarget.Private && !privateScreenShown ? PrivateComputerMemory : 0);
            var models = ollamaModel;
            var browse = OllamaModelPanel.BrowseButton("Browse Ollama models…", chosen =>
            {
                var list = (models.ItemsSource as string[] ?? []).ToList();
                if (!list.Any(m => OllamaModels.Same(m, chosen))) { list.Add(chosen); models.ItemsSource = list.ToArray(); }
                models.SelectedItem = list.First(m => OllamaModels.Same(m, chosen));
            });
            var ollamaPart = new StackPanel { Spacing = 8 };
            ollamaPart.Children.Add(ollamaModel); ollamaPart.Children.Add(browse); ollamaPart.Children.Add(panel);
            openRouterModel = OpenRouterModelBox(agent.Provider == ModelProvider.OpenRouter ? agent.ModelId ?? ThisPcAgent.SuggestedOpenRouterModels[0] : ThisPcAgent.SuggestedOpenRouterModels[0], out var openRouterField);
            var zdrNote = Muted(AppSettings.Current.OpenRouterRequiresZdr ? "Zero Data Retention: requested (Settings)." : "Zero Data Retention: off (Settings).");
            var openRouterPart = new StackPanel { Spacing = 8 };
            openRouterPart.Children.Add(openRouterField); openRouterPart.Children.Add(zdrNote);
            if (Secrets.ReadOpenRouter() is null) openRouterPart.Children.Add(Muted("Add your OpenRouter key in Settings."));
            void ShowProvider()
            {
                bool hosted = provider!.SelectedIndex == 1;
                ollamaPart.Visibility = hosted ? Visibility.Collapsed : Visibility.Visible;
                openRouterPart.Visibility = hosted ? Visibility.Visible : Visibility.Collapsed;
            }
            provider.SelectionChanged += (_, _) => ShowProvider();
            ollamaModel.SelectionChanged += async (_, _) => { if (models.SelectedItem is string m) await panel.Use(m); };
            ShowProvider();
            form.Children.Add(harness); form.Children.Add(provider); form.Children.Add(ollamaPart); form.Children.Add(openRouterPart);
            if (agent.Target == ConversationTarget.Box)
            {
                boxName = new TextBox { Header = "Box", Text = agent.BoxName ?? "", PlaceholderText = "Your default box" };
                form.Children.Add(new TextBlock { Text = "Box", Style = (Style)Application.Current.Resources["SectionLabel"] });
                form.Children.Add(boxName);
            }
        }

        var dialog = Dialog("Agent Settings", new ScrollViewer { Content = form, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, MaxHeight = 560 }, "Done");
        dialog.SecondaryButtonText = "New conversation";
        dialog.DefaultButton = ContentDialogButton.Primary;
        var result = await dialog.ShowAsync();
        if (result == ContentDialogResult.None) return;
        if (name.Text.Trim() is { Length: > 0 } chosenName) agent.Name = chosenName;
        if (saiComputer?.SelectedItem is Computer computer && computer.Id != agent.ComputerId) { agent.ComputerId = computer.Id; agent.SessionId = null; }
        if (saiModel?.SelectedItem is SaiModel model) agent.ModelId = model.Id;
        if (harness is not null && provider is not null && !isRunning)
        {
            var harnesses = agent.Target switch
            {
                ConversationTarget.Private => Harnesses.All,
                ConversationTarget.ThisPc => Harnesses.ThisPc,
                _ => [PrivateHarness.ClaudeCode],
            };
            agent.Harness = harnesses[Math.Max(0, harness.SelectedIndex)];
            if (provider.SelectedIndex == 1 && (openRouterModel is null ? "" : ModelText(openRouterModel)) is { Length: > 0 } hostedModel)
            { agent.Provider = ModelProvider.OpenRouter; agent.ModelId = hostedModel; }
            else if (provider.SelectedIndex == 0 && ollamaModel?.SelectedItem is string localModel)
            { agent.Provider = ModelProvider.Ollama; agent.ModelId = localModel; }
        }
        if (boxName is not null) agent.BoxName = boxName.Text.Trim() is { Length: > 0 } box ? box : null;
        Row(agent).Refresh();
        if (result == ContentDialogResult.Secondary) await NewConversation(agent);
        UpdateBadges(); Controls();
        await Persist();
    }
}
