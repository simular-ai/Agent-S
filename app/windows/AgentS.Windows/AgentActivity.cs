using AgentS.Core;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>
/// The floating "Agent Activity" window for This PC agents (macOS's MacTurnPanel): a small copy of the chat that
/// stays on top of the PC's apps while the agent works in them. It shows the same conversation as the main chat,
/// drawn by the same code (see ChatSurface): messages, folded tool steps, approvals with Allow / Allow for this
/// task / Deny, and "Working…" under the newest row. The header has the agent (opens the full chat), its status,
/// Take over (pause before its next action) and Stop; the composer sends a message, or a follow-up while it works.
/// It opens on its own when a This PC turn works in an app or needs approval, and closes when that turn ends
/// (bringing the chat back), unless the person opened it with the button or is typing in it.
/// </summary>
public sealed partial class MainWindow
{
    private Window? activityWindow;
    private Conversation? activityOwner;
    /// <summary>Opened with the button (or the person typed in it): stays after the turn ends.</summary>
    private bool activityManual;
    private bool activityVisible, activityFocused, activityComposerFocused, activityClosesWithApp;
    private TextBlock? activityName, activityStatus;
    private Microsoft.UI.Xaml.Shapes.Ellipse? activityDot;
    private Button? activityTakeOver, activityStop;
    private Border? activityWorking;
    private TextBox? activityComposer;

    /// <summary>Take over: the agent's next action on this PC waits until the person resumes.</summary>
    private bool pcPaused;
    private bool PcComposerEditing => activityVisible && activityFocused && activityComposerFocused
        && running && !stopRequested && runOwner == activityOwner;
    private bool PcActionsPaused => pcPaused || PcComposerEditing;
    private TaskCompletionSource pcPauseChanged = NewResume();

    private void ActivityPauseChanged()
    {
        var changed = pcPauseChanged;
        pcPauseChanged = NewResume();
        changed.TrySetResult();
        RefreshActivity();
    }
    private static TaskCompletionSource NewResume() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private void SetPcPaused(bool paused)
    {
        if (pcPaused == paused) return;
        pcPaused = paused;
        ActivityPauseChanged();
    }

    /// <summary>Waits before the next action while the person takes over or edits in the floating chat.</summary>
    private async Task WaitWhilePaused(CancellationToken cancellation)
    {
        var canceled = Task.Delay(Timeout.Infinite, cancellation);
        while (PcActionsPaused && running && !stopRequested && !cancellation.IsCancellationRequested)
            await Task.WhenAny(pcPauseChanged.Task, canceled);
    }

    /// <summary>The chat header's "Show floating chat" button (This PC agents).</summary>
    private void ShowFloatingChat(object sender, RoutedEventArgs e)
    {
        if (selected is not { Target: ConversationTarget.ThisPc } owner) return;
        ShowActivity(owner, manual: true);
    }

    /// <summary>Opens the window for <paramref name="owner"/>. Automatic openings come from app work or approvals.</summary>
    private void ShowActivity(Conversation owner, bool manual)
    {
        if (owner.Target != ConversationTarget.ThisPc) return;
        activityWindow ??= MakeActivityWindow();
        if (activityOwner != owner)
        {
            activityOwner = owner;
            if (activityComposer is not null) activityComposer.Text = "";
            Redraw(activityChat!);
        }
        activityManual |= manual;
        RefreshActivity();
        if (!activityVisible)
        {
            // Shown without taking the keyboard from the app the agent (or the person) is using.
            activityWindow.AppWindow.Show(activateWindow: manual);
            activityVisible = true;
            ActivityPauseChanged();
            ScrollActivityToEnd();
        }
        else if (manual) activityWindow.Activate();
    }

    /// <summary>A This PC turn ended: close an automatic window and bring the chat back, as on macOS.</summary>
    private void ActivityTurnEnded(Conversation owner)
    {
        SetPcPaused(false);
        if (activityOwner != owner || activityWindow is null) return;
        bool typing = activityFocused || !string.IsNullOrWhiteSpace(activityComposer?.Text);
        if (activityManual || typing) { activityManual = true; RefreshActivity(); return; }
        bool wasVisible = activityVisible;
        HideActivity();
        if (wasVisible) OpenChat(owner);
    }

    private void HideActivity()
    {
        if (activityWindow is null || !activityVisible) return;
        activityWindow.AppWindow.Hide();
        activityVisible = false; activityManual = false; activityFocused = false;
        ActivityPauseChanged();
    }

    private void OpenChat(Conversation owner)
    {
        if (!conversations.Contains(owner)) return;
        Select(owner);
        if (AppWindow.Presenter is OverlappedPresenter { State: OverlappedPresenterState.Minimized } presenter) presenter.Restore();
        Activate();
    }

    private void ScrollActivityToEnd()
    {
        if (activityChat is not { } surface) return;
        DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () =>
        { surface.Scroll.UpdateLayout(); surface.Scroll.ChangeView(null, surface.Scroll.ScrollableHeight, null, true); });
    }

    private Window MakeActivityWindow()
    {
        var window = new Window { Title = "Agent Activity" };
        var icon = Path.Combine(AppContext.BaseDirectory, "Assets", AppIdentity.IconFile);
        if (File.Exists(icon)) window.AppWindow.SetIcon(icon);
        if (window.AppWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.IsAlwaysOnTop = true;
            presenter.IsMaximizable = false;
            presenter.IsMinimizable = false;
        }
        window.AppWindow.IsShownInSwitchers = false;
        // A little larger than macOS's 388 x 354 panel, since it holds the whole chat; bottom-right of the main window's screen.
        double scale = Content.XamlRoot?.RasterizationScale ?? 1.0;
        int width = (int)(420 * scale), height = (int)(520 * scale);
        var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
        window.AppWindow.Resize(new global::Windows.Graphics.SizeInt32(width, height));
        window.AppWindow.Move(new global::Windows.Graphics.PointInt32(
            area.X + Math.Max(0, area.Width - width - (int)(18 * scale)), area.Y + Math.Max(0, area.Height - height - (int)(18 * scale))));
        // Closing hides it; it comes back with the next app work or the button. It closes for good with the main window.
        window.AppWindow.Closing += (_, args) => { if (!activityClosesWithApp) { args.Cancel = true; HideActivity(); } };
        Closed += (_, _) => { activityClosesWithApp = true; window.Close(); };
        window.Activated += (_, args) =>
        {
            activityFocused = args.WindowActivationState != WindowActivationState.Deactivated;
            ActivityPauseChanged();
        };
        window.Content = BuildActivityContent();
        return window;
    }

    private FrameworkElement BuildActivityContent()
    {
        var root = new Grid { Background = Theme("ChatBackground"), RequestedTheme = (Content as FrameworkElement)?.ActualTheme ?? ElementTheme.Default };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        // ---- Header: the agent (opens the chat), its status, Take over and Stop ----
        var header = new Grid { Padding = new Thickness(12), ColumnSpacing = 8, BorderBrush = Theme("DividerBrush"), BorderThickness = new Thickness(0, 0, 0, 1) };
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var who = new StackPanel { Spacing = 3 };
        var nameLine = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4 };
        activityName = new TextBlock { FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, FontSize = 15, TextTrimming = TextTrimming.CharacterEllipsis };
        nameLine.Children.Add(activityName);
        nameLine.Children.Add(new FontIcon { Glyph = "", FontSize = 10, Opacity = 0.5, VerticalAlignment = VerticalAlignment.Center });
        var statusLine = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
        activityDot = new Microsoft.UI.Xaml.Shapes.Ellipse { Width = 6, Height = 6, VerticalAlignment = VerticalAlignment.Center };
        activityStatus = new TextBlock { FontSize = 12, Opacity = 0.7 };
        statusLine.Children.Add(activityDot); statusLine.Children.Add(activityStatus);
        who.Children.Add(nameLine); who.Children.Add(statusLine);
        var open = new Button { Content = who, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), Padding = new Thickness(0), HorizontalContentAlignment = HorizontalAlignment.Left };
        ToolTipService.SetToolTip(open, "Open the full conversation");
        open.Click += (_, _) => { if (activityOwner is { } owner) OpenChat(owner); };
        activityTakeOver = Accent(new Button { Content = "Take over" });
        activityTakeOver.Click += (_, _) => SetPcPaused(!pcPaused);
        activityStop = new Button { Content = "Stop" };
        activityStop.Click += (_, _) => { if (activityOwner is { } owner && running && runOwner == owner) StopTurn(this, new RoutedEventArgs()); };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(activityTakeOver, "ActivityTakeOver");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(activityStop, "ActivityStop");
        Grid.SetColumn(activityTakeOver, 1); Grid.SetColumn(activityStop, 2);
        header.Children.Add(open); header.Children.Add(activityTakeOver); header.Children.Add(activityStop);

        // ---- The conversation, as in the main chat, with "Working…" under the newest row ----
        var messages = new StackPanel { Spacing = 14 };
        activityWorking = new Border
        {
            Visibility = Visibility.Collapsed, HorizontalAlignment = HorizontalAlignment.Left, CornerRadius = new CornerRadius(16),
            Padding = new Thickness(12, 7, 12, 7), Background = Theme("ChipBackground"),
        };
        var working = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        working.Children.Add(new ProgressRing { Width = 14, Height = 14, IsActive = true });
        working.Children.Add(new TextBlock { Text = "Working…", FontSize = 13, Foreground = Theme("MutedText") });
        activityWorking.Child = working;
        var transcript = new StackPanel { Spacing = 14, Padding = new Thickness(14, 12, 14, 12) };
        transcript.Children.Add(messages); transcript.Children.Add(activityWorking);
        var scroll = new ScrollViewer { Content = transcript, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
        Grid.SetRow(scroll, 1);
        activityChat = new ChatSurface(messages, scroll, () => activityOwner, compact: true);

        // ---- Composer: a message, or a follow-up while the agent works ----
        var composer = new Grid { Padding = new Thickness(12, 8, 12, 12), ColumnSpacing = 8, BorderBrush = Theme("DividerBrush"), BorderThickness = new Thickness(0, 1, 0, 0) };
        composer.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        composer.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        activityComposer = new TextBox { AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, MaxHeight = 96 };
        activityComposer.GotFocus += (_, _) => { activityComposerFocused = true; ActivityPauseChanged(); };
        activityComposer.LostFocus += (_, _) => { activityComposerFocused = false; ActivityPauseChanged(); };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(activityComposer, "Message");
        var send = new Button { Content = new FontIcon { Glyph = "", FontSize = 14 } };
        ToolTipService.SetToolTip(send, "Send (Enter)");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(send, "Send");
        void Submit()
        {
            if (activityOwner is not { } owner || activityComposer.Text.Trim() is not { Length: > 0 } text) return;
            if (!running && !CanRun(owner)) return;
            activityComposer.Text = "";
            send.Focus(FocusState.Programmatic);
            if (running) Enqueue(owner, text);
            else _ = RunTurn(owner, text);
            RefreshActivity();
        }
        send.Click += (_, _) => Submit();
        activityComposer.PreviewKeyDown += (_, args) =>
        {
            bool shift = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(global::Windows.System.VirtualKey.Shift).HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
            if (args.Key == global::Windows.System.VirtualKey.Enter && !shift) { args.Handled = true; Submit(); }
        };
        Grid.SetColumn(send, 1);
        composer.Children.Add(activityComposer); composer.Children.Add(send);
        Grid.SetRow(composer, 2);

        root.Children.Add(header); root.Children.Add(scroll); root.Children.Add(composer);
        return root;
    }

    /// <summary>Updates the header, "Working…" and the composer; the rows themselves update with the main chat.</summary>
    private void RefreshActivity()
    {
        if (activityWindow is null || activityOwner is not { } owner || activityName is null) return;
        bool working = running && runOwner == owner;
        bool needsYou = owner.Entries.Any(e => e.LocalAnswer is not null && !e.Answered);
        (string Title, global::Windows.UI.Color Color) status =
            working && stopRequested ? ("Stopping…", global::Windows.UI.Color.FromArgb(255, 128, 128, 128))
            : needsYou ? ("Approval needed", global::Windows.UI.Color.FromArgb(255, 220, 120, 20))
            : working && pcPaused ? ("New actions paused", global::Windows.UI.Color.FromArgb(255, 36, 150, 84))
            : PcComposerEditing ? ("PC actions pause while you type", global::Windows.UI.Color.FromArgb(255, 36, 150, 84))
            : working ? ("Controlling this PC", global::Windows.UI.Color.FromArgb(255, 220, 120, 20))
            : ("Ready", global::Windows.UI.Color.FromArgb(255, 128, 128, 128));
        activityName.Text = owner.Name;
        activityStatus!.Text = status.Title;
        activityDot!.Fill = new SolidColorBrush(status.Color);
        activityTakeOver!.Content = pcPaused ? "Resume" : "Take over";
        ToolTipService.SetToolTip(activityTakeOver, pcPaused ? "Resume this agent's actions on this PC" : "Pause before this agent's next action on this PC");
        activityTakeOver.IsEnabled = activityStop!.IsEnabled = working && !stopRequested;
        // As the main chat: "Working…" under the newest row while this agent's turn runs, not while it waits for you.
        var showWorking = working && !needsYou && !PcActionsPaused ? Visibility.Visible : Visibility.Collapsed;
        if (activityWorking!.Visibility != showWorking)
        {
            bool following = activityChat?.Following ?? true;
            activityWorking.Visibility = showWorking;
            if (following) ScrollActivityToEnd();
        }
        activityComposer!.PlaceholderText = working ? "Queue a follow-up…" : $"Ask {owner.Name} to do something on this PC…";
    }

    /// <summary>Answers a This PC approval from either view; both show the result.</summary>
    private void AnswerLocal(ChatEntry entry, bool allowed, string note, bool rest = false)
    {
        if (entry.Answered || entry.LocalAnswer is not { } answer) return;
        entry.Answered = true;
        if (rest) approveRestOfTurn = true;
        answer(allowed);
        entry.Text += "\n" + note;
        foreach (var surface in Surfaces)
            if (surface.Approvals.TryGetValue(entry.Id, out var controls)) SetButtonsEnabled(controls, false);
        Change(entry);
        RefreshActivity();
    }
}
