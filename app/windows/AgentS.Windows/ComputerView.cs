using AgentS.Core;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using global::Windows.System;
using global::Windows.UI.Core;

namespace AgentS.Windows;

/// <summary>
/// The computer panel's macOS controls: fill the window with the computer (sidebar, chat and context
/// hidden), and "Take over", which sends your mouse and keyboard to the Private computer.
/// </summary>
public sealed partial class MainWindow
{
    private bool maximized;
    private TakeOverSession? takeOver;
    private bool renderingFrame;
    private DateTime lastMove;

    /// <summary>Take over needs a running WSL Private computer (the Hyper-V prototype allows only bounded agent actions).</summary>
    private bool CanTakeOver => IsPrivate && PrivateComputer.UsesWsl && privateScreenShown && !busy;

    private void InitializeComputerView()
    {
        ScreenSurface.PointerPressed += ScreenPressed;
        ScreenSurface.PointerMoved += ScreenMoved;
        ScreenSurface.PointerReleased += ScreenReleased;
        ScreenSurface.PointerWheelChanged += ScreenWheel;
        ScreenSurface.PreviewKeyDown += ScreenKeyDown;
        ScreenSurface.CharacterReceived += ScreenCharacter;
        Root.SizeChanged += (_, _) => { FitMaximizedScreen(); ClampPanel(); };
        Closed += (_, _) => EndTakeOver();
        // A resizable split, as on macOS (ideal 640, at least 420): drag the panel's left edge.
        panelWidth = Math.Max(MinPanelWidth, AppSettings.Current.ComputerPanelWidth);
        if (panelOpen) ComputerColumn.Width = new GridLength(panelWidth);
        var handle = new ResizeHandle { Width = 8, HorizontalAlignment = HorizontalAlignment.Left, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), Margin = new Thickness(-4, 0, 0, 0) };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(handle, "Resize the computer panel");
        double startX = 0, startWidth = 0;
        handle.PointerPressed += (_, e) => { if (maximized) return; handle.CapturePointer(e.Pointer); startX = e.GetCurrentPoint(Root).Position.X; startWidth = panelWidth; e.Handled = true; };
        handle.PointerMoved += (_, e) =>
        {
            if (handle.PointerCaptures?.Count is not > 0) return;
            panelWidth = Math.Clamp(startWidth - (e.GetCurrentPoint(Root).Position.X - startX), MinPanelWidth, MaxPanelWidth);
            ComputerColumn.Width = new GridLength(panelWidth);
        };
        handle.PointerReleased += (_, e) =>
        {
            handle.ReleasePointerCapture(e.Pointer);
            AppSettings.Current.ComputerPanelWidth = panelWidth; AppSettings.Current.Save();
        };
        ComputerPanel.Children.Add(handle);
    }

    private const double MinPanelWidth = 420;
    private double panelWidth = 520;
    /// <summary>Leaves the sidebar and at least 360 px of chat.</summary>
    private double MaxPanelWidth => Math.Max(MinPanelWidth, Root.ActualWidth - SidebarColumn.ActualWidth - 360);

    private void ClampPanel()
    {
        if (maximized || !panelOpen || Root.ActualWidth <= 0) return;
        var width = Math.Clamp(panelWidth, MinPanelWidth, MaxPanelWidth);
        if (Math.Abs(ComputerColumn.Width.Value - width) > 0.5) ComputerColumn.Width = new GridLength(width);
    }

    /// <summary>The split's handle, with the left-right resize cursor.</summary>
    private sealed class ResizeHandle : Grid
    {
        public ResizeHandle() => ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.SizeWestEast);
    }

    // ---- Maximize ---------------------------------------------------------------------------------

    private void ToggleMaximized(object sender, RoutedEventArgs e) => SetMaximized(!maximized);

    private void SetMaximized(bool on)
    {
        if (on && !panelOpen) TogglePanel(this, new RoutedEventArgs());
        maximized = on;
        var hidden = on ? Visibility.Collapsed : Visibility.Visible;
        SidebarPane.Visibility = ChatPane.Visibility = ContextPanel.Visibility = ContextDivider.Visibility = hidden;
        SidebarColumn.Width = on ? new GridLength(0) : new GridLength(260);
        ChatColumn.MinWidth = on ? 0 : 360;
        ChatColumn.Width = on ? new GridLength(0) : new GridLength(1, GridUnitType.Star);
        ComputerColumn.Width = on ? new GridLength(1, GridUnitType.Star) : new GridLength(panelOpen ? panelWidth : 0);
        PanelScroll.VerticalScrollBarVisibility = on ? ScrollBarVisibility.Disabled : ScrollBarVisibility.Auto;
        PanelScroll.VerticalScrollMode = on ? ScrollMode.Disabled : ScrollMode.Auto;
        MaximizeIcon.Glyph = on ? "" : "";
        var tip = on ? "Back to the chat" : "Fill the window with the computer";
        ToolTipService.SetToolTip(MaximizeButton, tip);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(MaximizeButton, tip);
        FitMaximizedScreen();
    }

    /// <summary>Maximized, the screen takes all the height under the header; otherwise it sizes to its picture.</summary>
    private void FitMaximizedScreen()
    {
        if (!maximized)
        {
            ScreenCard.ClearValue(FrameworkElement.HeightProperty);
            LiveView.Height = 260;
            return;
        }
        var header = ScreenSection.Children[0] is FrameworkElement head ? head.ActualHeight : 40;
        var height = Math.Max(200, PanelScroll.ActualHeight - header - ScreenSection.Padding.Top - ScreenSection.Padding.Bottom - ScreenSection.Spacing);
        ScreenCard.Height = height;
        LiveView.Height = height - 2;
    }

    // ---- Take over --------------------------------------------------------------------------------

    private void ToggleTakeOver(object sender, RoutedEventArgs e)
    {
        if (TakeOverButton.IsChecked == true) StartTakeOver(); else EndTakeOver();
    }

    private void StartTakeOver()
    {
        if (takeOver is not null) return;
        if (!CanTakeOver) { TakeOverButton.IsChecked = false; return; }
        try { takeOver = new TakeOverSession(); }
        catch (Exception error) { TakeOverButton.IsChecked = false; Error(error); return; }
        takeOver.Frame += png =>
        {
            // Drop frames while one is still being drawn, so the view never falls behind.
            if (renderingFrame) return;
            renderingFrame = true;
            if (!DispatcherQueue.TryEnqueue(async () =>
            {
                try { if (takeOver is not null && IsPrivate) await ShowPrivateScreen(png); }
                catch (Exception) { /* A bad frame is skipped. */ }
                finally { renderingFrame = false; }
            })) renderingFrame = false;
        };
        ScreenCard.BorderBrush = (Brush)Application.Current.Resources["AccentFillColorDefaultBrush"];
        ScreenCard.BorderThickness = new Thickness(2);
        ScreenSurface.IsTabStop = true;
        ScreenSurface.Focus(FocusState.Programmatic);
        TakeOverButton.IsChecked = true;
        var tip = "Release control and return to watching";
        ToolTipService.SetToolTip(TakeOverButton, tip);
        Show("You're controlling the private computer. Click the hand again to give control back.", false);
    }

    private void EndTakeOver()
    {
        var session = takeOver;
        takeOver = null;
        session?.Dispose();
        ScreenCard.BorderBrush = Theme("CardBorder");
        ScreenCard.BorderThickness = new Thickness(1);
        ScreenSurface.IsTabStop = false;
        TakeOverButton.IsChecked = false;
        ToolTipService.SetToolTip(TakeOverButton, "Control the private computer with your keyboard and mouse");
    }

    /// <summary>Called from Controls(): shows the right buttons and ends a take-over the computer can no longer take.</summary>
    private void UpdateComputerView()
    {
        // The header keeps Take over, Maximize and ⋯ (as on macOS); the screen action lives in ⋯.
        ScreenMenuItem.Text = IsPrivate ? "Refresh screen" : IsBox ? "Open the live view in your browser" : "Open in browser";
        ScreenMenuItem.IsEnabled = ScreenButton.IsEnabled;
        ScreenMenuItem.Visibility = IsThisPc ? Visibility.Collapsed : Visibility.Visible;
        WatchingBadge.Visibility = PrivateScreen.Visibility == Visibility.Visible && takeOver is null && (IsPrivate || IsBox)
            ? Visibility.Visible : Visibility.Collapsed;
        ToolTipService.SetToolTip(StatusDot, ComputerState.Text);
        MaximizeButton.Visibility = selected is null ? Visibility.Collapsed : Visibility.Visible;
        TakeOverButton.Visibility = IsPrivate && PrivateComputer.UsesWsl ? Visibility.Visible : Visibility.Collapsed;
        TakeOverButton.IsEnabled = takeOver is not null || CanTakeOver;
        if (takeOver is not null && (!IsPrivate || !privateScreenShown || !panelOpen || !takeOver.IsAlive)) EndTakeOver();
        if (maximized && (selected is null || !panelOpen)) SetMaximized(false);
    }

    // ---- Mouse and keyboard -----------------------------------------------------------------------

    private (int X, int Y)? GuestPoint(PointerRoutedEventArgs e)
    {
        var point = e.GetCurrentPoint(PrivateScreen).Position;
        return TakeOverInput.ToScreen(point.X, point.Y, PrivateScreen.ActualWidth, PrivateScreen.ActualHeight);
    }

    private static int ButtonOf(PointerUpdateKind kind) => kind switch
    {
        PointerUpdateKind.LeftButtonPressed or PointerUpdateKind.LeftButtonReleased => 1,
        PointerUpdateKind.MiddleButtonPressed or PointerUpdateKind.MiddleButtonReleased => 2,
        PointerUpdateKind.RightButtonPressed or PointerUpdateKind.RightButtonReleased => 3,
        _ => 0,
    };

    private async void ScreenPressed(object sender, PointerRoutedEventArgs e)
    {
        if (takeOver is not { } session) return;
        ScreenSurface.Focus(FocusState.Pointer);
        var button = ButtonOf(e.GetCurrentPoint(PrivateScreen).Properties.PointerUpdateKind);
        if (button == 0 || GuestPoint(e) is not { } at) return;
        e.Handled = true;
        ScreenSurface.CapturePointer(e.Pointer);
        await session.Send(TakeOverInput.Move(at.X, at.Y));
        await session.Send(TakeOverInput.Button(button, down: true));
    }

    private async void ScreenMoved(object sender, PointerRoutedEventArgs e)
    {
        if (takeOver is not { } session || DateTime.UtcNow - lastMove < TimeSpan.FromMilliseconds(30) || GuestPoint(e) is not { } at) return;
        lastMove = DateTime.UtcNow;
        await session.Send(TakeOverInput.Move(at.X, at.Y));
    }

    private async void ScreenReleased(object sender, PointerRoutedEventArgs e)
    {
        if (takeOver is not { } session) return;
        var button = ButtonOf(e.GetCurrentPoint(PrivateScreen).Properties.PointerUpdateKind);
        ScreenSurface.ReleasePointerCapture(e.Pointer);
        if (button == 0) return;
        e.Handled = true;
        // A drag that ends outside the picture still releases the button where it left off.
        if (GuestPoint(e) is { } at) await session.Send(TakeOverInput.Move(at.X, at.Y));
        await session.Send(TakeOverInput.Button(button, down: false));
    }

    private async void ScreenWheel(object sender, PointerRoutedEventArgs e)
    {
        if (takeOver is not { } session || GuestPoint(e) is not { } at) return;
        e.Handled = true;
        var properties = e.GetCurrentPoint(PrivateScreen).Properties;
        var delta = properties.MouseWheelDelta;
        var notches = Math.Clamp(Math.Abs(delta) / 120, 1, 5);
        await session.Send(TakeOverInput.Move(at.X, at.Y));
        for (var i = 0; i < notches; i++) await session.Send(TakeOverInput.Wheel(properties.IsHorizontalMouseWheel ? -delta : delta, properties.IsHorizontalMouseWheel));
    }

    private static bool IsDown(VirtualKey key) =>
        InputKeyboardSource.GetKeyStateForCurrentThread(key).HasFlag(CoreVirtualKeyStates.Down);

    /// <summary>Keys that type nothing (Enter, arrows, F-keys…) and shortcuts with Ctrl or Alt go as key presses; text goes through CharacterReceived.</summary>
    private async void ScreenKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (takeOver is not { } session) return;
        bool ctrl = IsDown(VirtualKey.Control), alt = IsDown(VirtualKey.Menu), shift = IsDown(VirtualKey.Shift);
        string? name = e.Key switch
        {
            VirtualKey.Enter => "Return", VirtualKey.Tab => "Tab", VirtualKey.Escape => "Escape", VirtualKey.Back => "BackSpace",
            VirtualKey.Delete => "Delete", VirtualKey.Insert => "Insert", VirtualKey.Home => "Home", VirtualKey.End => "End",
            VirtualKey.PageUp => "Page_Up", VirtualKey.PageDown => "Page_Down",
            VirtualKey.Left => "Left", VirtualKey.Right => "Right", VirtualKey.Up => "Up", VirtualKey.Down => "Down",
            >= VirtualKey.F1 and <= VirtualKey.F12 => "F" + (e.Key - VirtualKey.F1 + 1),
            _ => null,
        };
        if (name is null && (ctrl || alt))
        {
            name = e.Key switch
            {
                >= VirtualKey.A and <= VirtualKey.Z => ((char)('a' + (e.Key - VirtualKey.A))).ToString(),
                >= VirtualKey.Number0 and <= VirtualKey.Number9 => ((char)('0' + (e.Key - VirtualKey.Number0))).ToString(),
                VirtualKey.Space => "space",
                _ => null,
            };
        }
        if (name is null) return;
        e.Handled = true;
        var modifiers = new List<string>();
        if (ctrl) modifiers.Add("ctrl");
        if (alt) modifiers.Add("alt");
        if (shift) modifiers.Add("shift");
        await session.Send(TakeOverInput.Key(modifiers, name));
    }

    private async void ScreenCharacter(UIElement sender, CharacterReceivedRoutedEventArgs e)
    {
        if (takeOver is not { } session || e.Character < ' ' || e.Character == '\u007f') return;
        // Ctrl/Alt shortcuts were sent as key presses already (AltGr characters still type).
        if (IsDown(VirtualKey.Control) != IsDown(VirtualKey.Menu)) return;
        e.Handled = true;
        await session.Send(TakeOverInput.Text(e.Character.ToString()));
    }
}
