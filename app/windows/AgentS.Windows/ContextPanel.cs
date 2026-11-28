using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace AgentS.Windows;

/// <summary>
/// The macOS context panel: "Action items (N)" (approvals to review and Open work to run or delete),
/// "Memory" (notes the agent keeps, which you can add, edit, pin, resolve, share or forget), and
/// "Session and instructions". Memory is stored per agent, beside the conversations.
/// </summary>
public sealed partial class MainWindow
{
    private bool showActionItems = true, showMemory, showSession;
    private readonly HashSet<Guid> checkedWork = [];
    private readonly Dictionary<string, AgentMemoryDocument?> memoryDocs = [];

    private static string MemoryFolder => Path.Combine(HistoryStore.DataFolder, "memory");

    // ---- Memory storage and routes ----------------------------------------------------------------

    /// <summary>The agent's memory, or null when its file can't be read (memory is then off for it).</summary>
    private AgentMemoryDocument? MemoryOf(Conversation agent)
    {
        if (memoryDocs.TryGetValue(agent.Id, out var cached)) return cached;
        AgentMemoryDocument? document;
        try { document = new AgentMemoryStore(MemoryFolder, agent.Id).Load(); }
        catch (Exception error) when (error is IOException or System.Text.Json.JsonException or UnauthorizedAccessException or ArgumentException) { document = null; }
        memoryDocs[agent.Id] = document;
        return document;
    }

    private bool SaveMemory(Conversation agent, AgentMemoryDocument document)
    {
        try { new AgentMemoryStore(MemoryFolder, agent.Id).Save(document); memoryDocs[agent.Id] = document; return true; }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException)
        { Show("Couldn't save this agent's memory."); return false; }
    }

    private void ForgetMemory(Conversation agent)
    {
        memoryDocs.Remove(agent.Id);
        try { new AgentMemoryStore(MemoryFolder, agent.Id).Delete(); } catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException) { }
    }

    /// <summary>Where the agent's notes apply; Ollama recall needs the model confirmed local or hosted.</summary>
    private AgentMemoryRoute RouteOf(Conversation agent)
    {
        OllamaInference? inference = null;
        if (agent.Target != ConversationTarget.Sai && agent.Provider == ModelProvider.Ollama)
        {
            var model = agent.ModelId ?? OllamaModels.Default;
            var installed = ollamaInstalled?.FirstOrDefault(m => OllamaModels.Same(m.Name, model));
            inference = OllamaModels.IsCloudReference(model) || installed is { Remote: true } ? OllamaInference.Hosted
                : installed is not null ? OllamaInference.Local : OllamaInference.Unconfirmed;
        }
        return AgentMemoryRoute.For(agent, inference);
    }

    /// <summary>Starts a memory turn: the prompt gets the instructions and recalled notes; the reply's trailer is hidden.</summary>
    private MemoryTurn BeginMemory(Conversation agent, string message, string sourceId) =>
        new(MemoryOf(agent)?.Notes, RouteOf(agent), message, sourceId);

    /// <summary>Ends a memory turn: shows the last held-back text, then saves the notes it changed (only after success).</summary>
    private void FinishMemory(Conversation owner, MemoryTurn memory, bool success)
    {
        var document = MemoryOf(owner);
        var (events, updated, error) = memory.Finish(document?.Notes, success);
        foreach (var item in events) RenderText(owner, item);
        if (updated is not null && document is not null) SaveMemory(owner, new AgentMemoryDocument { Notes = updated, Sessions = document.Sessions });
        if (error is not null) System.Diagnostics.Debug.WriteLine($"Memory proposal rejected: {error.Message}");
        if (selected == owner) UpdateContext();
    }

    /// <summary>Shows a harness reply: a whole block, or (through memory) streamed deltas and snapshots. Returns false for other events.</summary>
    private bool RenderText(Conversation owner, HarnessEvent item)
    {
        switch (item)
        {
            case HarnessText said:
                if (!string.IsNullOrWhiteSpace(said.Text)) Add(owner, new ChatEntry { Role = "Agent", Text = said.Text });
                return true;
            case HarnessTextDelta delta:
            {
                var id = turn + "/m/" + delta.Id;
                var entry = owner.Entries.FirstOrDefault(e => e.Id == id);
                if (entry is null) { if (delta.Text.Length == 0) return true; entry = new ChatEntry { Id = id, Role = "Agent" }; Add(owner, entry); }
                entry.Text += delta.Text; Change(entry);
                return true;
            }
            case HarnessTextSnapshot snapshot:
            {
                var id = turn + "/m/" + snapshot.Id;
                var entry = owner.Entries.FirstOrDefault(e => e.Id == id);
                if (entry is null) { if (string.IsNullOrWhiteSpace(snapshot.Text)) return true; entry = new ChatEntry { Id = id, Role = "Agent" }; Add(owner, entry); }
                entry.Text = snapshot.Text; Change(entry);
                return true;
            }
            default: return false;
        }
    }

    // ---- Panel ------------------------------------------------------------------------------------

    /// <summary>What the panel last showed; it's rebuilt only when this changes, not on every screen frame.</summary>
    private string? contextShown;

    private void UpdateContext()
    {
        ContextPanel.Visibility = selected is null ? Visibility.Collapsed : Visibility.Visible;
        var owner = selected;
        var shown = owner is null ? "" : string.Join('|',
            owner.Id, owner.Name, owner.Target, owner.Harness, owner.Provider, owner.ModelId, owner.ComputerId, owner.BoxName, owner.SessionId,
            string.Join(',', owner.HarnessSessions.Select(s => s.Value)),
            string.Join(',', owner.Entries.Where(e => e.Role == "Needs you" && !e.Answered && (e.LocalAnswer is not null || e.ApprovalId is not null)).Select(e => e.Id)),
            MemoryOf(owner)?.Notes.Revision, usage, computers.Length, ollamaInstalled?.Length, PrivateComputer.IsConfigured);
        if (shown == contextShown && ContextSections.Children.Count > 0) return;
        contextShown = shown;
        ContextSections.Children.Clear();
        if (owner is null) return;
        ContextSections.Children.Add(ActionItems(owner));
        ContextSections.Children.Add(MemorySection(owner));
        ContextSections.Children.Add(SessionSection(owner));
    }

    private Expander Section(string header, UIElement content, bool expanded, Action<bool> remember)
    {
        var expander = new Expander
        {
            Header = new TextBlock { Text = header, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold },
            Content = content, IsExpanded = expanded,
            HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch,
        };
        expander.Expanding += (_, _) => remember(true);
        expander.Collapsed += (_, _) => remember(false);
        return expander;
    }

    private static string FirstSentence(string text)
    {
        var line = text.Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault()?.Trim() ?? text.Trim();
        var end = line.IndexOfAny(['.', '!', '?']);
        var sentence = end > 0 && end < line.Length - 1 ? line[..(end + 1)] : line;
        return sentence.Length > 90 ? sentence[..89] + "…" : sentence;
    }

    private UIElement ActionItems(Conversation owner)
    {
        var body = new StackPanel { Spacing = 8 };
        var requests = owner.Entries.Where(e => e.Role == "Needs you" && !e.Answered && (e.LocalAnswer is not null || e.ApprovalId is not null)).ToArray();
        var work = MemoryOf(owner)?.Notes.OpenWork(RouteOf(owner)) ?? [];
        if (requests.Length == 0 && work.Count == 0) body.Children.Add(Muted("Nothing needs your attention."));
        foreach (var request in requests)
        {
            var row = new Grid { ColumnSpacing = 8 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.Children.Add(new FontIcon { Glyph = "", FontSize = 13, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 3, 0, 0) });
            var title = new TextBlock { Text = FirstSentence(request.Text) is { Length: > 0 } t ? t : "Approval request", TextWrapping = TextWrapping.Wrap, MaxLines = 3 };
            Grid.SetColumn(title, 1); row.Children.Add(title);
            var review = new Button { Content = "Review", Padding = new Thickness(10, 3, 10, 3) };
            review.Click += (_, _) => Reveal(request);
            Grid.SetColumn(review, 2); row.Children.Add(review);
            body.Children.Add(new Border { Child = row, Padding = new Thickness(10), CornerRadius = new CornerRadius(10), Background = Theme("CardBackground"), BorderBrush = Theme("CardBorder"), BorderThickness = new Thickness(1) });
        }
        if (work.Count > 0)
        {
            body.Children.Add(new TextBlock { Text = "OPEN WORK", Style = (Style)Application.Current.Resources["SectionLabel"], Margin = new Thickness(0, 6, 0, 0) });
            var actions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, HorizontalAlignment = HorizontalAlignment.Right };
            var delete = new Button { Content = "Delete", Padding = new Thickness(10, 3, 10, 3) };
            var runChecked = Accent(new Button { Content = "Run", Padding = new Thickness(12, 3, 12, 3) });
            actions.Children.Add(delete); actions.Children.Add(runChecked);
            void ShowActions() => actions.Visibility = work.Any(n => checkedWork.Contains(n.Id)) ? Visibility.Visible : Visibility.Collapsed;
            foreach (var note in work)
            {
                var check = new CheckBox { Content = new TextBlock { Text = FirstSentence(note.Text), TextWrapping = TextWrapping.Wrap }, IsChecked = checkedWork.Contains(note.Id) };
                ToolTipService.SetToolTip(check, note.Text);
                check.Checked += (_, _) => { checkedWork.Add(note.Id); ShowActions(); };
                check.Unchecked += (_, _) => { checkedWork.Remove(note.Id); ShowActions(); };
                body.Children.Add(check);
            }
            delete.Click += async (_, _) =>
            {
                var ids = work.Where(n => checkedWork.Contains(n.Id)).Select(n => n.Id).ToArray();
                if (ids.Length == 0 || await Dialog("Delete checked work?", "The agent forgets these items.", "Delete").ShowAsync() != ContentDialogResult.Primary) return;
                EditMemory(owner, m => m.Forget(ids), ids);
                foreach (var id in ids) checkedWork.Remove(id);
            };
            runChecked.Click += (_, _) =>
            {
                var items = work.Where(n => checkedWork.Contains(n.Id)).ToArray();
                if (items.Length == 0) return;
                // The macOS wording: one turn for all checked items.
                var message = "Work on the checked items below.\n\n" + string.Join("\n", items.Select(n =>
                {
                    var lines = n.Text.Trim().Split('\n');
                    return "- " + lines[0] + string.Concat(lines.Skip(1).Select(l => "\n  " + l));
                }));
                foreach (var item in items) checkedWork.Remove(item.Id);
                if (running) Enqueue(owner, message);
                else if (CanRun(owner)) _ = RunTurn(owner, message);
                UpdateContext();
            };
            ShowActions();
            body.Children.Add(actions);
        }
        return Section($"Action items ({requests.Length + work.Count})", body, showActionItems, open => showActionItems = open);
    }

    /// <summary>Scrolls the chat to a pending approval ("Review").</summary>
    private void Reveal(ChatEntry entry)
    {
        if (approvalControls.TryGetValue(entry.Id, out var controls))
            controls.StartBringIntoView(new BringIntoViewOptions { VerticalAlignmentRatio = 0.5, AnimationDesired = true });
    }

    private UIElement MemorySection(Conversation owner)
    {
        var body = new StackPanel { Spacing = 8 };
        var document = MemoryOf(owner);
        var route = RouteOf(owner);
        if (document is null) body.Children.Add(Muted("This agent's memory couldn't be read, so it's off for now."));
        else if (!route.AllowsMemory) body.Children.Add(Muted("Memory isn't available for this agent's model."));
        else
        {
            var add = new Button { Content = "Add note", Padding = new Thickness(10, 3, 10, 3) };
            add.Click += async (_, _) => await EditNote(owner, null);
            body.Children.Add(add);
            if (document.Notes.Notes.Count == 0) body.Children.Add(Muted("No memory yet. The agent saves notes as it works, or add your own."));
            foreach (var kind in AgentMemoryKinds.All)
            {
                var notes = document.Notes.NotesIn(kind);
                if (notes.Count == 0) continue;
                body.Children.Add(new TextBlock { Text = kind.DisplayName().ToUpperInvariant(), Style = (Style)Application.Current.Resources["SectionLabel"], Margin = new Thickness(0, 6, 0, 0) });
                foreach (var note in notes) body.Children.Add(NoteRow(owner, note, route));
            }
        }
        return Section("Memory", body, showMemory, open => showMemory = open);
    }

    private UIElement NoteRow(Conversation owner, AgentMemoryNote note, AgentMemoryRoute route)
    {
        var row = new Grid { ColumnSpacing = 6 };
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var text = new StackPanel { Spacing = 2 };
        var line = new TextBlock { Text = note.Text, TextWrapping = TextWrapping.Wrap, MaxLines = 4, TextTrimming = TextTrimming.CharacterEllipsis };
        if (note.Status == AgentMemoryStatus.Resolved) line.Opacity = 0.55;
        text.Children.Add(line);
        var facts = new List<string>();
        if (note.IsPinned) facts.Add("Pinned");
        if (note.Status == AgentMemoryStatus.Resolved) facts.Add("Resolved");
        facts.Add(note.Scope.IsPrivate ? "Private" : "Shared with " + note.Scope.DisplayName);
        text.Children.Add(new TextBlock { Text = string.Join(" · ", facts), FontSize = 11, Foreground = Theme("MutedText") });
        row.Children.Add(text);
        var more = new Button { Content = new FontIcon { Glyph = "", FontSize = 12 }, Padding = new Thickness(6), BorderThickness = new Thickness(0), Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.Transparent), VerticalAlignment = VerticalAlignment.Top };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(more, "Note actions");
        var menu = new MenuFlyout();
        void Item(string label, Action act) { var item = new MenuFlyoutItem { Text = label }; item.Click += (_, _) => act(); menu.Items.Add(item); }
        Item("Edit…", () => _ = EditNote(owner, note));
        Item(note.IsPinned ? "Unpin" : "Pin", () => EditMemory(owner, m => m.SetPinned(!note.IsPinned, note.Id), [note.Id]));
        if (note.Kind == AgentMemoryKind.ActiveWork)
            Item(note.Status == AgentMemoryStatus.Open ? "Mark resolved" : "Reopen",
                () => EditMemory(owner, m => m.SetStatus(note.Status == AgentMemoryStatus.Open ? AgentMemoryStatus.Resolved : AgentMemoryStatus.Open, note.Id), [note.Id]));
        if (note.SourceChatItemId is { } source && owner.Entries.FirstOrDefault(e => e.Id == source) is { } sourceEntry)
            Item("Source message", () => { if (entryText.TryGetValue(sourceEntry.Id, out var t)) t.StartBringIntoView(); });
        menu.Items.Add(new MenuFlyoutSeparator());
        if (note.Scope.IsPrivate)
            Item("Share with current route…", async () =>
            {
                if (await Dialog("Share memory?", $"Agents on {route.DisplayName} can read this note.", "Share").ShowAsync() == ContentDialogResult.Primary)
                    EditMemory(owner, m => m.SetScope(AgentMemoryScope.Of(route), note.Id), [note.Id]);
            });
        else Item("Keep private", () => EditMemory(owner, m => m.SetScope(AgentMemoryScope.Private, note.Id), [note.Id]));
        Item("Forget", () => EditMemory(owner, m => m.Forget(note.Id), [note.Id]));
        more.Flyout = menu;
        Grid.SetColumn(more, 1); row.Children.Add(more);
        return new Border { Child = row, Padding = new Thickness(10, 8, 4, 8), CornerRadius = new CornerRadius(10), Background = Theme("CardBackground"), BorderBrush = Theme("CardBorder"), BorderThickness = new Thickness(1) };
    }

    private void EditMemory(Conversation owner, Func<AgentMemory, bool> change, IEnumerable<Guid> ids)
    {
        if (MemoryOf(owner) is not { } document) return;
        try
        {
            if (document.Edited(change, ids) is { } next) SaveMemory(owner, next);
        }
        catch (AgentMemoryException error) { Show(error.Message); }
        UpdateContext();
    }

    /// <summary>The macOS "Add memory note" / edit sheet: section, text, and sharing.</summary>
    private async Task EditNote(Conversation owner, AgentMemoryNote? note)
    {
        var route = RouteOf(owner);
        var kind = new ComboBox { Header = "Section", ItemsSource = AgentMemoryKinds.All.Select(k => k.DisplayName()).ToArray(), HorizontalAlignment = HorizontalAlignment.Stretch,
            SelectedIndex = Array.IndexOf(AgentMemoryKinds.All, note?.Kind ?? AgentMemoryKind.ActiveWork) };
        var text = new TextBox { Header = "Note", Text = note?.Text ?? "", AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, MinHeight = 90, MaxLength = AgentMemory.MaxNoteBytes };
        var sharing = new ComboBox { Header = "Sharing", HorizontalAlignment = HorizontalAlignment.Stretch,
            ItemsSource = note is null ? new[] { "Private", "Share with current route" } : new[] { "Keep existing", "Private", "Share with current route" }, SelectedIndex = 0 };
        var form = new StackPanel { Spacing = 12, MinWidth = 380 };
        form.Children.Add(kind); form.Children.Add(text); form.Children.Add(sharing);
        var dialog = Dialog(note is null ? "Add memory note" : "Edit memory note", form, "Save");
        dialog.DefaultButton = ContentDialogButton.Primary;
        if (await dialog.ShowAsync() != ContentDialogResult.Primary || text.Text.Trim().Length == 0) return;
        var chosenKind = AgentMemoryKinds.All[Math.Max(0, kind.SelectedIndex)];
        var share = (string)sharing.SelectedItem;
        var id = note?.Id;
        EditMemory(owner, m =>
        {
            bool changed = m.SaveUserNote(id, chosenKind, text.Text.Trim(), note is null && share == "Share with current route" ? route : null);
            var savedId = id ?? m.Notes.OrderByDescending(n => n.UpdatedAt).First().Id;
            if (note is not null && share != "Keep existing")
                changed |= m.SetScope(share == "Private" ? AgentMemoryScope.Private : AgentMemoryScope.Of(route), savedId);
            return changed;
        }, id is { } existing ? [existing] : []);
    }

    private UIElement SessionSection(Conversation owner)
    {
        var grid = new Grid { ColumnSpacing = 16, RowSpacing = 8 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        void Pair(string label, string value, bool mono = false)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var key = new TextBlock { Text = label, Opacity = 0.6 };
            var text = new TextBlock { Text = value, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true };
            if (mono) text.FontFamily = new Microsoft.UI.Xaml.Media.FontFamily("Cascadia Mono, Consolas");
            Grid.SetRow(key, grid.RowDefinitions.Count - 1);
            Grid.SetRow(text, grid.RowDefinitions.Count - 1); Grid.SetColumn(text, 1);
            grid.Children.Add(key); grid.Children.Add(text);
        }
        PrivateHarness? harness = owner.Target switch
        {
            ConversationTarget.Sai => null,
            ConversationTarget.Private => owner.Harness ?? PrivateHarness.AgentS,
            ConversationTarget.ThisPc => owner.ThisPcHarness,
            _ => PrivateHarness.ClaudeCode,
        };
        Pair("Harness", harness is null ? "Sai" : harness.Value == PrivateHarness.AgentS ? "Agent S (built-in)" : harness.Value.DisplayName());
        Pair("Model", owner.Target == ConversationTarget.Sai ? $"{(owner.ModelId is null or "auto" ? "auto" : owner.ModelId)} · Sai API"
            : $"{ModelName(owner)} · {(owner.Provider == ModelProvider.OpenRouter ? "OpenRouter" : "Ollama on this PC")}");
        Pair("Computer", owner.Target switch
        {
            ConversationTarget.Sai => computers.FirstOrDefault(c => c.Id == owner.ComputerId)?.Name ?? "Unknown",
            ConversationTarget.Box => owner.BoxName is null ? "Simular Box (default)" : $"Simular Box · {owner.BoxName}",
            ConversationTarget.ThisPc => "This PC",
            _ => PrivateComputer.UsesWsl || !PrivateComputer.IsConfigured ? "Private Linux (WSL)" : "Private Linux (Hyper-V)",
        });
        var session = owner.Target == ConversationTarget.Sai ? owner.SessionId
            : harness is { } h && h.RunsInComputer() ? owner.HarnessSessions.GetValueOrDefault(h) : null;
        if (owner.Target == ConversationTarget.Sai || harness is { } running && running.RunsInComputer())
            Pair("Session", session ?? "New (starts with the next message)", mono: true);
        if (owner == selected && usage is { } used) Pair("Last turn", $"{used.Input:N0} tokens in · {used.Output:N0} out");

        var body = new StackPanel { Spacing = 10 };
        body.Children.Add(new TextBlock { Text = "SESSION", Style = (Style)Application.Current.Resources["SectionLabel"] });
        body.Children.Add(grid);
        var (label, instructions) = Instructions(owner, harness);
        if (instructions is not null)
        {
            body.Children.Add(new TextBlock { Text = "INSTRUCTIONS", Style = (Style)Application.Current.Resources["SectionLabel"], Margin = new Thickness(0, 6, 0, 0) });
            body.Children.Add(new Expander
            {
                Header = new TextBlock { Text = label, Opacity = 0.75 },
                Content = new TextBlock { Text = instructions, FontFamily = new Microsoft.UI.Xaml.Media.FontFamily("Cascadia Mono, Consolas"), FontSize = 12, TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true },
                HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch,
            });
        }
        return Section("Session and instructions", body, showSession, open => showSession = open);
    }

    /// <summary>What the harness is told, as macOS shows it: the system prompt, AGENTS.md, or Box's tools.</summary>
    private static (string Label, string? Text) Instructions(Conversation owner, PrivateHarness? harness) => owner.Target switch
    {
        ConversationTarget.Box => ("Simular Box tools", Box.Instructions(owner.BoxName)),
        ConversationTarget.ThisPc => ("Agent S tools and approvals",
            harness == PrivateHarness.ClaudeCode ? ThisPcAgent.Instructions(ThisPcRunner.PcToolPath) : ThisPcCommand.Instructions),
        ConversationTarget.Private when harness is PrivateHarness.Codex or PrivateHarness.DeepSeek => ("AGENTS.md, written before each turn", AgentInstructions.Text(harness.Value)),
        ConversationTarget.Private when harness is PrivateHarness.ClaudeCode => ("System prompt (replaces Claude Code's default)", AgentInstructions.Text(PrivateHarness.ClaudeCode)),
        _ => ("", (string?)null),
    };
}
