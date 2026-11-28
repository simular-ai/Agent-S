using System.Diagnostics;
using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;

namespace AgentS.Windows;

/// <summary>
/// First launch, as on macOS: "Set up your first agent" (harness, provider and model for a Private agent),
/// then "Get your agent ready", where the Private computer sets up while the model downloads.
/// </summary>
public sealed partial class MainWindow
{
    /// <summary>Opens New agent on Cloud (from "Set up a cloud agent…").</summary>
    private bool preferCloud;
    /// <summary>The Private computer's memory when it starts, for the model memory check: the chosen limit, else 2 GB as on macOS.</summary>
    private static ulong PrivateComputerMemory => (ulong)(WslComputer.MemoryGB ?? PrivateComputer.MemoryChoices[0]) << 30;

    /// <summary>"Computer memory": 2, 4 or 6 GB, as macOS's "Sandbox memory".</summary>
    private static ComboBox MemoryPicker(int selectedGB) => new()
    {
        Header = "Computer memory", HorizontalAlignment = HorizontalAlignment.Stretch,
        ItemsSource = PrivateComputer.MemoryChoices.Select(gb => $"{gb} GB").ToArray(),
        SelectedIndex = Math.Max(0, Array.IndexOf(PrivateComputer.MemoryChoices, selectedGB)),
    };

    private bool ShowsOnboarding => !AppSettings.Current.FinishedFirstAgentSetup && conversations.Count == 0;

    private static readonly PrivateHarness[] FirstAgentHarnesses = [PrivateHarness.DeepSeek, PrivateHarness.ClaudeCode, PrivateHarness.Codex];

    private void ShowOnboarding()
    {
        Onboarding.Visibility = Visibility.Visible;
        var page = Page("Set up your first agent", "The open-source agent that uses apps and websites on its own private desktop.");

        var form = new StackPanel { Spacing = 14 };
        var where = Card("Private · Linux computer on this PC", "\uE7F4", "Your agent gets a separate desktop that runs on this PC.");
        var memory = MemoryPicker(AppSettings.Current.ComputerMemoryGB ?? PrivateComputer.MemoryChoices[0]);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(memory, "FirstAgentMemory");
        ((StackPanel)where.Child).Children.Add(memory);
        ((StackPanel)where.Child).Children.Add(Muted("The most its desktop and apps can use. Ollama runs the model in this PC's memory separately. You can change it in Settings."));
        form.Children.Add(where);

        var harness = new ComboBox { Header = "Harness", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = FirstAgentHarnesses.Select(h => h.DisplayName()).ToArray(), SelectedIndex = 0 };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(harness, "FirstAgentHarness");
        var provider = new ComboBox { Header = "Provider", HorizontalAlignment = HorizontalAlignment.Stretch, ItemsSource = new[] { "Ollama", "OpenRouter" }, SelectedIndex = 0 };
        var providerNote = Muted("Ollama runs the model on this PC.");
        var models = new ComboBox { Header = "Model", HorizontalAlignment = HorizontalAlignment.Stretch };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(models, "FirstAgentModel");
        string selectedModel = OllamaSetup.DefaultModel;
        var details = Muted("");
        var browse = OllamaModelPanel.BrowseButton("Browse models…", chosen => { selectedModel = chosen; FillModels(); });
        var modelRow = new Grid { ColumnSpacing = 12 };
        modelRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        modelRow.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        modelRow.Children.Add(details); Grid.SetColumn(browse, 1); modelRow.Children.Add(browse);
        var zdr = ZdrToggle(AppSettings.Current.OpenRouterRequiresZdr);
        var openRouterModel = OpenRouterModelBox(ThisPcAgent.SuggestedOpenRouterModels[0], out var openRouterField);
        var ollamaPart = new StackPanel { Spacing = 8 };
        ollamaPart.Children.Add(models); ollamaPart.Children.Add(modelRow);
        var openRouterPart = new StackPanel { Spacing = 8, Visibility = Visibility.Collapsed };
        openRouterPart.Children.Add(zdr); openRouterPart.Children.Add(openRouterField);
        form.Children.Add(harness); form.Children.Add(provider); form.Children.Add(providerNote); form.Children.Add(ollamaPart); form.Children.Add(openRouterPart);

        async void FillModels()
        {
            var installed = (await OllamaSetup.Installed(TimeSpan.FromSeconds(3)) ?? []).Where(m => !m.Remote).Select(m => OllamaModels.NormalizedName(m.Name)).ToList();
            foreach (var name in OllamaModels.Suggested) if (!installed.Any(m => OllamaModels.Same(m, name))) installed.Add(name);
            if (!installed.Any(m => OllamaModels.Same(m, selectedModel))) installed.Add(selectedModel);
            models.ItemsSource = installed.Select(m => new ModelChoice(m, OllamaModels.DisplayName(m))).ToArray();
            models.SelectedIndex = installed.FindIndex(m => OllamaModels.Same(m, selectedModel));
        }
        models.SelectionChanged += (_, _) =>
        {
            if (models.SelectedItem is not ModelChoice choice) return;
            selectedModel = choice.Id;
            var input = OllamaModels.InputLabel(OllamaModels.SuggestedInput(choice.Id));
            var size = OllamaModels.PresetSize(choice.Id);
            details.Text = string.Join(" · ", new[] { input, size is null ? "" : $"Estimated download · {size}" }.Where(s => s.Length > 0));
        };
        provider.SelectionChanged += (_, _) =>
        {
            bool openRouter = provider.SelectedIndex == 1;
            ollamaPart.Visibility = openRouter ? Visibility.Collapsed : Visibility.Visible;
            openRouterPart.Visibility = openRouter ? Visibility.Visible : Visibility.Collapsed;
            providerNote.Text = openRouter ? "OpenRouter and the selected model provider receive your messages and content the agent reads." : "Ollama runs the model on this PC.";
        };
        FillModels();

        var create = Accent(new Button { Content = "Set up agent" });
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(create, "FirstAgentCreate");
        create.Click += async (_, _) =>
        {
            bool openRouter = provider.SelectedIndex == 1;
            var model = openRouter ? ModelText(openRouterModel) : selectedModel;
            if (model.Length == 0) return;
            if (openRouter) AppSettings.Current.OpenRouterRequiresZdr = zdr.IsOn;
            AppSettings.Current.ComputerMemoryGB = PrivateComputer.MemoryChoices[Math.Max(0, memory.SelectedIndex)];
            AppSettings.Current.Save();
            var agent = new Conversation
            {
                Name = "Agent", Target = ConversationTarget.Private, Harness = FirstAgentHarnesses[Math.Max(0, harness.SelectedIndex)],
                Provider = openRouter ? ModelProvider.OpenRouter : ModelProvider.Ollama, ModelId = model,
            };
            conversations.Add(agent); Select(agent); await Persist();
            ShowPreparation(agent);
        };
        page.Children.Add(form);
        page.Children.Add(Footer(Muted("Next, set up the private computer and connect your model."), create));
        Onboarding.Children.Clear();
        Onboarding.Children.Add(Scroll(page));
    }

    /// <summary>"Get your agent ready": the Private computer and the model, side by side, then Open chat.</summary>
    private void ShowPreparation(Conversation agent)
    {
        Onboarding.Visibility = Visibility.Visible;
        bool openRouter = agent.Provider == ModelProvider.OpenRouter;
        var page = Page("Get your agent ready", openRouter ? "" : "You can download the model while the private computer sets up.");

        // ---- The Private computer ----
        var computerState = Muted("");
        var computerProgress = new ProgressBar { IsIndeterminate = true, Visibility = Visibility.Collapsed };
        var computerButton = new Button { Visibility = Visibility.Collapsed };
        var computerCard = Card("Private computer on this PC", "\uE7F4",
            "A separate Linux desktop for your agent. The first setup downloads Debian and installs the tools your agent uses.");
        ((StackPanel)computerCard.Child).Children.Add(computerState);
        ((StackPanel)computerCard.Child).Children.Add(computerProgress);
        ((StackPanel)computerCard.Child).Children.Add(computerButton);
        bool computerReady = PrivateComputer.IsConfigured, settingUp = false, restartPending = false;
        var cloud = new Button { Content = "Set up a cloud agent…" };
        var open = new Button();

        // ---- The model ----
        Border modelCard;
        OllamaModelPanel? panel = null;
        if (openRouter)
        {
            modelCard = Card(agent.ModelId ?? "OpenRouter", "\uE753", "OpenRouter runs this model online.");
            var key = new PasswordBox { Header = "OpenRouter API key", PlaceholderText = "Paste your key" };
            var save = new Button { Content = "Save key" };
            var connected = new TextBlock { Text = "✓ OpenRouter connected", Foreground = Green };
            void ShowKey()
            {
                bool has = Secrets.ReadOpenRouter() is not null;
                key.Visibility = save.Visibility = has ? Visibility.Collapsed : Visibility.Visible;
                connected.Visibility = has ? Visibility.Visible : Visibility.Collapsed;
            }
            save.Click += (_, _) =>
            {
                if (key.Password.Trim() is not { Length: > 0 } value) return;
                try { Secrets.SaveOpenRouter(value); } catch (Exception failure) { Error(failure); }
                key.Password = ""; ShowKey(); Refresh();
            };
            var inner = (StackPanel)modelCard.Child;
            inner.Children.Add(key); inner.Children.Add(save); inner.Children.Add(connected);
            ShowKey();
        }
        else
        {
            modelCard = Card(OllamaModels.DisplayName(agent.ModelId ?? OllamaSetup.DefaultModel), "\uE770", "Ollama downloads and runs this model on this PC.");
            panel = new OllamaModelPanel(agent.ModelId ?? OllamaSetup.DefaultModel, () => computerReady ? 0 : PrivateComputerMemory);
            ((StackPanel)modelCard.Child).Children.Add(panel);
        }

        bool Ready() => computerReady && (openRouter ? Secrets.ReadOpenRouter() is not null : panel?.IsReady == true);
        void Refresh()
        {
            computerProgress.Visibility = settingUp ? Visibility.Visible : Visibility.Collapsed;
            if (computerReady) { computerState.Text = "✓ Private computer ready"; computerState.Foreground = Green; computerState.Opacity = 1; computerButton.Visibility = Visibility.Collapsed; }
            else if (!settingUp)
            {
                computerButton.Visibility = Visibility.Visible;
                computerButton.Content = restartPending ? "Restart now" : computerState.Text.Length > 0 && computerState.Foreground == Red ? "Try again" : "Set up private computer";
            }
            else computerButton.Visibility = Visibility.Collapsed;
            bool ready = Ready();
            open.Content = ready ? "Open chat" : "Finish later";
            if (ready) Accent(open); else open.ClearValue(FrameworkElement.StyleProperty);
            cloud.Visibility = conversations.Any(c => c.Target is ConversationTarget.Sai or ConversationTarget.Box) ? Visibility.Collapsed : Visibility.Visible;
        }
        async Task SetUpComputer()
        {
            if (restartPending)
            {
                Process.Start(new ProcessStartInfo("shutdown.exe", "/r /t 10 /c \"Restarting to finish setting up Agent S\"") { UseShellExecute = false, CreateNoWindow = true });
                return;
            }
            settingUp = true; computerState.ClearValue(TextBlock.ForegroundProperty); computerState.Opacity = 0.75; Refresh();
            try
            {
                computerState.Text = "Checking Windows' virtual machine feature…";
                var wsl = await WslComputer.CheckWsl();
                if (wsl == WslComputer.Readiness.NeedsWsl)
                {
                    computerState.Text = "Turning on Windows' virtual machine feature. Approve the Windows prompt…";
                    await WslComputer.InstallWsl();
                    wsl = await WslComputer.CheckWsl();
                }
                if (wsl != WslComputer.Readiness.Ready)
                {
                    restartPending = true;
                    computerState.Text = "Restart your PC to finish turning on Windows' virtual machine feature. Afterwards, open Agent S to continue.";
                    return;
                }
                await WslComputer.Setup(new Progress<(string Step, double? Percent)>(update =>
                {
                    computerState.Text = update.Step;
                    computerProgress.IsIndeterminate = update.Percent is null;
                    if (update.Percent is { } percent) computerProgress.Value = percent;
                }), CancellationToken.None);
                computerReady = true;
            }
            catch (Exception failure) { computerState.Text = Describe(failure); computerState.Foreground = Red; computerState.Opacity = 1; }
            finally { settingUp = false; Refresh(); Controls(); }
        }
        computerButton.Click += async (_, _) => await SetUpComputer();
        if (panel is not null) panel.Changed += Refresh;
        open.Click += (_, _) => FinishOnboarding();
        cloud.Click += (_, _) => { FinishOnboarding(); preferCloud = true; NewAgent(cloud, new RoutedEventArgs()); };

        page.Children.Add(computerCard);
        page.Children.Add(modelCard);
        var actions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 12, HorizontalAlignment = HorizontalAlignment.Center };
        actions.Children.Add(cloud); actions.Children.Add(open);
        page.Children.Add(actions);
        Onboarding.Children.Clear();
        Onboarding.Children.Add(Scroll(page));
        Refresh();
        // As on macOS, the computer starts setting up right away, while the model downloads.
        if (!computerReady) _ = SetUpComputer();
    }

    private void FinishOnboarding()
    {
        AppSettings.Current.FinishedFirstAgentSetup = true;
        AppSettings.Current.Save();
        Onboarding.Visibility = Visibility.Collapsed;
        Onboarding.Children.Clear();
        Controls();
    }

    // ---- Small builders shared with Agent Settings ------------------------------------------------

    private sealed record ModelChoice(string Id, string Label) { public override string ToString() => Label; }

    private static SolidColorBrush Green => new(global::Windows.UI.Color.FromArgb(255, 36, 150, 84));
    private static SolidColorBrush Red => new(global::Windows.UI.Color.FromArgb(255, 221, 54, 72));

    private static TextBlock Muted(string text) => new() { Text = text, TextWrapping = TextWrapping.Wrap, FontSize = 13, Opacity = 0.75 };

    private static T Accent<T>(T button) where T : Button
    {
        if (Application.Current.Resources.TryGetValue("AccentButtonStyle", out var accent) && accent is Style style) button.Style = style;
        return button;
    }

    /// <summary>"Request Zero Data Retention (ZDR)", as in the macOS OpenRouter settings.</summary>
    private static ToggleSwitch ZdrToggle(bool on) => new()
    {
        Header = "Request Zero Data Retention (ZDR)", IsOn = on,
        OnContent = "On · applies to all OpenRouter agents", OffContent = "Off · applies to all OpenRouter agents",
    };

    /// <summary>An editable OpenRouter model box: suggestions at once, the full list as it loads. <paramref name="field"/> holds it with its browser and details.</summary>
    private static ComboBox OpenRouterModelBox(string current, out FrameworkElement field)
    {
        var box = new ComboBox { Header = "Model ID", IsEditable = true, HorizontalAlignment = HorizontalAlignment.Stretch, Text = current,
            PlaceholderText = ThisPcAgent.SuggestedOpenRouterModels[0] };
        var suggestions = ThisPcAgent.SuggestedOpenRouterModels.Prepend(current).Distinct().ToArray();
        box.ItemsSource = suggestions;
        box.SelectedItem = current;
        box.Loaded += async (_, _) =>
        {
            var (all, _) = await ModelBrowser.LoadOpenRouter();
            if (all.Length == 0) return;
            var keep = box.Text;
            box.ItemsSource = suggestions.Concat(all.Select(m => m.Id).Except(suggestions)).ToArray();
            box.Text = keep;
        };
        field = WithBrowser(box, [ModelProvider.OpenRouter]);
        return box;
    }

    /// <summary>The model box's current text (typed, or the chosen item).</summary>
    private static string ModelText(ComboBox box) => ((box.SelectedItem as string) is { Length: > 0 } item && item == box.Text ? item : box.Text ?? "").Trim();

    /// <summary>
    /// macOS's model field: "Browse … models…" above the box (the full list, with prices, ZDR and which agents use each),
    /// and for an OpenRouter model, its text token price, input and Zero Data Retention below it. The returned field
    /// follows the box's visibility, so callers keep showing and hiding the box.
    /// </summary>
    private static FrameworkElement WithBrowser(ComboBox box, IReadOnlyList<ModelProvider> providers, Action<ModelProvider>? providerChosen = null, Func<ModelProvider>? startOn = null)
    {
        bool openRouterOnly = providers.Count == 1 && providers[0] == ModelProvider.OpenRouter;
        var field = new StackPanel { Spacing = 6, Visibility = box.Visibility };
        var details = new StackPanel { Spacing = 4 };
        var browse = new Button { Content = openRouterOnly ? "Browse OpenRouter models…" : "Browse all models…" };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(browse, openRouterOnly ? "BrowseOpenRouterModels" : "BrowseAllModels");
        ModelBrowser.Attach(browse, providers, (provider, model) =>
        {
            providerChosen?.Invoke(provider);
            if (box.ItemsSource is IEnumerable<string> items && !items.Contains(model)) box.ItemsSource = items.Prepend(model).ToArray();
            box.SelectedItem = model;
            box.Text = model;
            _ = ShowDetails();
        }, startOn, () => ModelText(box));
        field.Children.Add(browse); field.Children.Add(box); field.Children.Add(details);
        // Callers show and hide the box; the field (with its button and details) follows.
        box.RegisterPropertyChangedCallback(UIElement.VisibilityProperty, (_, _) => field.Visibility = box.Visibility);

        Grid Pair(string label, FrameworkElement value)
        {
            var row = new Grid { ColumnSpacing = 12 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.Children.Add(new TextBlock { Text = label, FontSize = 13 });
            Grid.SetColumn(value, 1); row.Children.Add(value);
            return row;
        }
        TextBlock Value(string text) => new() { Text = text, FontSize = 13, Opacity = 0.75, HorizontalAlignment = HorizontalAlignment.Right };
        TextBlock Caption(string text) => new() { Text = text, FontSize = 12, Opacity = 0.7, TextWrapping = TextWrapping.Wrap };

        async Task ShowDetails()
        {
            var id = ModelText(box);
            bool openRouter = openRouterOnly || startOn?.Invoke() == ModelProvider.OpenRouter;
            details.Children.Clear();
            if (!openRouter || id.Length == 0) return;
            details.Children.Add(Caption("Checking price and ZDR…"));
            var (models, zdr) = await ModelBrowser.LoadOpenRouter();
            if (ModelText(box) != id) return;
            details.Children.Clear();
            var found = models.FirstOrDefault(m => m.Id == id);
            details.Children.Add(Pair("Text token price", Value(found?.PriceText ?? OpenRouterModel.PriceLabel(id))));
            details.Children.Add(Pair("Model input", Value(found?.Input is { } input ? OllamaModels.InputLabel(input) : "Unknown")));
            var availability = zdr?.Availability(id, isPublicModel: found is not null) ?? OpenRouterZdrAvailability.Unknown;
            details.Children.Add(Pair("Zero Data Retention", ModelBrowser.ZdrIndicator(availability, 13)));
            if (availability == OpenRouterZdrAvailability.SafetyRetentionOnly)
                details.Children.Add(Caption("Using this model with ZDR requested requires allowing Safety Retention in OpenRouter."));
            else if (availability == OpenRouterZdrAvailability.Unavailable)
                details.Children.Add(Caption("OpenRouter may reject this model when ZDR is requested."));
        }
        box.SelectionChanged += (_, _) => _ = ShowDetails();
        box.TextSubmitted += (_, _) => _ = ShowDetails();
        box.LostFocus += (_, _) => _ = ShowDetails();
        box.Loaded += (_, _) => _ = ShowDetails();
        field.Tag = (Func<Task>)ShowDetails;
        return field;
    }

    /// <summary>Refreshes a model field's details after something outside it changed (its provider).</summary>
    private static void RefreshModelField(FrameworkElement field) { if (field.Tag is Func<Task> show) _ = show(); }

    private StackPanel Page(string title, string subtitle)
    {
        var page = new StackPanel { Spacing = 18, MaxWidth = 520, Padding = new Thickness(28), HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center };
        var head = new StackPanel { Spacing = 8, HorizontalAlignment = HorizontalAlignment.Center };
        var logo = Path.Combine(AppContext.BaseDirectory, "Assets", AppIdentity.LogoFile);
        if (File.Exists(logo)) head.Children.Add(new Image { Width = 44, Height = 44, Source = new BitmapImage(new Uri(logo)) });
        head.Children.Add(new TextBlock { Text = title, FontSize = 22, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Center });
        if (subtitle.Length > 0) head.Children.Add(new TextBlock { Text = subtitle, TextAlignment = TextAlignment.Center, TextWrapping = TextWrapping.Wrap, Opacity = 0.75, HorizontalAlignment = HorizontalAlignment.Center });
        page.Children.Add(head);
        return page;
    }

    private Border Card(string title, string glyph, string text)
    {
        var inner = new StackPanel { Spacing = 9 };
        var head = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        head.Children.Add(new FontIcon { Glyph = glyph, FontSize = 16 });
        head.Children.Add(new TextBlock { Text = title, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
        inner.Children.Add(head);
        if (text.Length > 0) inner.Children.Add(Muted(text));
        return new Border { Child = inner, CornerRadius = new CornerRadius(14), Padding = new Thickness(16), Background = Theme("ChipBackground") };
    }

    private static Grid Footer(UIElement note, Button action)
    {
        var footer = new Grid { ColumnSpacing = 12 };
        footer.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        footer.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        footer.Children.Add(note); Grid.SetColumn(action, 1); footer.Children.Add(action);
        return footer;
    }

    private static ScrollViewer Scroll(UIElement content) => new() { Content = content, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
}
