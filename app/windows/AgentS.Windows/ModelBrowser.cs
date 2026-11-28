using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>
/// The full model lists, as one browser: every Ollama library variant (download size, inputs, installed,
/// whether it may not fit in memory) and every OpenRouter model (context, price, Zero Data Retention), each
/// labeled with the agents that use it. It's a flyout, so it also opens from inside dialogs.
/// </summary>
internal static class ModelBrowser
{
    /// <summary>The app's agents, for "Used by …" labels.</summary>
    public static Func<IEnumerable<Conversation>> Agents { get; set; } = () => [];

    private static readonly OllamaCatalog Library = new();
    private static readonly OpenRouterCatalog OpenRouter = new();
    private static readonly Dictionary<string, OllamaCatalog.Variant[]?> families = [];
    private static string[] familyOrder = [];
    private static Task? libraryLoad;
    private static int familiesLoaded;
    private const int MaxRows = 250;

    private static TextBlock Muted(string text, double size = 12) => new() { Text = text, FontSize = size, Opacity = 0.7, TextWrapping = TextWrapping.Wrap };

    private static string UsedBy(ModelProvider provider, string model)
    {
        var names = Agents().Where(a => a.Target != ConversationTarget.Sai && a.Provider == provider &&
            (provider == ModelProvider.OpenRouter ? a.ModelId == model : OllamaModels.Same(a.ModelId ?? OllamaModels.Default, model)))
            .Select(a => a.Name).Distinct().ToArray();
        return names.Length == 0 ? "" : "Used by " + string.Join(", ", names);
    }

    /// <summary>The OpenRouter list and ZDR catalog, shared with the model fields' details (cached 10 minutes).</summary>
    public static Task<(OpenRouterModel[] Models, OpenRouterZdrCatalog? Zdr)> LoadOpenRouter(bool force = false) => OpenRouter.Load(force: force);

    /// <summary>A button that opens the browser; <paramref name="chosen"/> gets the provider and model id.</summary>
    public static Button Button(string label, IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn = null, Func<string?>? current = null)
    {
        var button = new Button { Content = label };
        Attach(button, providers, chosen, startOn, current);
        return button;
    }

    /// <summary>Opens the browser at <paramref name="anchor"/> (for the model menu above the chat).</summary>
    public static void ShowAt(FrameworkElement anchor, IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn = null, Func<string?>? current = null) =>
        Build(providers, chosen, startOn, current).ShowAt(anchor);

    public static void Attach(Button button, IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn = null, Func<string?>? current = null) =>
        button.Flyout = Build(providers, chosen, startOn, current);

    private static Flyout Build(IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn, Func<string?>? current)
    {
        var title = new TextBlock { FontSize = 18, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold };
        var tabs = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        var search = new TextBox { PlaceholderText = "Name, provider, or model ID" };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(search, "Search models");
        var options = new StackPanel { Spacing = 6 };
        var status = Muted("");
        var list = new ListView { SelectionMode = ListViewSelectionMode.None, IsItemClickEnabled = true, Height = 420 };
        // As macOS's sheet: a progress or "Refresh ZDR availability" on the left, Done on the right.
        var footerLeft = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, VerticalAlignment = VerticalAlignment.Center };
        var done = new Button { Content = "Done", HorizontalAlignment = HorizontalAlignment.Right };
        var footer = new Grid();
        footer.Children.Add(footerLeft); footer.Children.Add(done);
        var panel = new StackPanel { Spacing = 8, Width = 560 };
        panel.Children.Add(title); panel.Children.Add(tabs); panel.Children.Add(search); panel.Children.Add(options); panel.Children.Add(status); panel.Children.Add(list); panel.Children.Add(footer);
        // Flyouts are capped near 456px wide by default; the lists need the room.
        var presenter = new Style(typeof(FlyoutPresenter));
        presenter.Setters.Add(new Setter(FrameworkElement.MaxWidthProperty, 620.0));
        presenter.Setters.Add(new Setter(FrameworkElement.MaxHeightProperty, 760.0));
        presenter.Setters.Add(new Setter(ScrollViewer.HorizontalScrollBarVisibilityProperty, ScrollBarVisibility.Disabled));
        var flyout = new Flyout { Content = panel, Placement = FlyoutPlacementMode.Bottom, FlyoutPresenterStyle = presenter };
        done.Click += (_, _) => flyout.Hide();
        var tab = providers[0];
        InstalledModel[] installed = [];
        OpenRouterModel[] hosted = [];
        OpenRouterZdrCatalog? zdr = null;
        bool openRouterLoading = false;

        FrameworkElement Row(string title, IEnumerable<(string Text, Brush? Color)> facts, object tag)
        {
            var row = new StackPanel { Spacing = 2, Padding = new Thickness(0, 5, 0, 5), Tag = tag };
            row.Children.Add(new TextBlock { Text = title, TextTrimming = TextTrimming.CharacterEllipsis });
            var line = new TextBlock { FontSize = 12, TextWrapping = TextWrapping.Wrap };
            foreach (var (text, color) in facts.Where(f => f.Text.Length > 0))
            {
                if (line.Inlines.Count > 0) line.Inlines.Add(new Microsoft.UI.Xaml.Documents.Run { Text = " · ", Foreground = Gray });
                line.Inlines.Add(new Microsoft.UI.Xaml.Documents.Run { Text = text, Foreground = color ?? Gray });
            }
            if (line.Inlines.Count > 0) row.Children.Add(line);
            return row;
        }
        TextBlock Header(string text) => new() { Text = text, Style = (Style)Application.Current.Resources["SectionLabel"], Margin = new Thickness(0, 8, 0, 2) };

        void RenderOllama()
        {
            list.Items.Clear();
            var query = search.Text.Trim();
            bool Match(string name) => query.Length == 0 || query.Split(' ', StringSplitOptions.RemoveEmptyEntries).All(w => name.Contains(w, StringComparison.OrdinalIgnoreCase));
            var available = OllamaSetup.AvailableMemory();
            (string, Brush?) Fit(ulong bytes) => available is { } free && OllamaMemoryEstimate.FromBytes(bytes) is { } e && e.ExceedsBudget(free, 0)
                ? ("May not fit in memory", Orange) : ("", null);
            var mine = installed.Where(m => !m.Remote && Match(m.Name)).ToArray();
            if (mine.Length > 0)
            {
                list.Items.Add(Header("INSTALLED ON THIS PC"));
                foreach (var m in mine)
                {
                    var name = OllamaModels.NormalizedName(m.Name);
                    list.Items.Add(Row(name, [(OllamaModels.InputLabel(OllamaModels.SuggestedInput(name)), null), ($"Installed · {OllamaModels.FormatBytes(m.Size)}", Green), Fit((ulong)Math.Max(0, m.Size)), (UsedBy(ModelProvider.Ollama, name), Accent)], "ollama:" + name));
                }
            }
            int shown = 0, matching = 0;
            lock (families)
            {
                foreach (var family in familyOrder)
                {
                    if (!families.TryGetValue(family, out var variants) || variants is null) continue;
                    var hits = variants.Where(v => Match(v.Name)).ToArray();
                    matching += hits.Length;
                    if (hits.Length == 0 || shown >= MaxRows) continue;
                    if (shown == 0) list.Items.Add(Header("OLLAMA LIBRARY"));
                    // Each family under its name, as in the macOS browser.
                    list.Items.Add(new TextBlock { Text = family, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Margin = new Thickness(0, 8, 0, 0), IsHitTestVisible = false });
                    foreach (var v in hits.Take(MaxRows - shown))
                    {
                        var have = installed.FirstOrDefault(m => OllamaModels.Same(m.Name, v.Name));
                        var estimate = OllamaMemoryEstimate.FromCatalogSize(v.Size ?? "");
                        list.Items.Add(Row(v.Name,
                        [
                            (OllamaModels.InputLabel(v.Input), null),
                            have is not null ? ($"Installed · {OllamaModels.FormatBytes(have.Size)}", Green) : (v.Size is { } size ? $"{size} download" : "Size unknown", null),
                            estimate is not null && available is { } free && estimate.ExceedsBudget(free, 0) ? ("May not fit in memory", Orange) : ("", null),
                            (UsedBy(ModelProvider.Ollama, v.Name), Accent),
                        ], "ollama:" + v.Name));
                        shown++;
                    }
                }
            }
            bool loading = libraryLoad is { IsCompleted: false };
            status.Text = (loading ? $"Loading the Ollama library… {familiesLoaded} of {familyOrder.Length} families. " : "")
                + (matching > shown ? $"Showing {shown} of {matching} variants; search to narrow. " : $"{matching} local variants. ")
                + "Sizes are downloads; Ollama needs about that much free memory to run one.";
        }

        // macOS's catalog row: name and input, the id, the price on the left; ZDR and context on the right; a check on the current model.
        FrameworkElement OpenRouterRow(OpenRouterModel m, bool selected)
        {
            var availability = zdr?.Availability(m.Id, isPublicModel: true) ?? OpenRouterZdrAvailability.Unknown;
            var row = new Grid { ColumnSpacing = 12, Padding = new Thickness(0, 6, 0, 6), Tag = "openrouter:" + m.Id };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(16) });
            var left = new StackPanel { Spacing = 2 };
            var nameLine = new TextBlock { TextTrimming = TextTrimming.CharacterEllipsis };
            nameLine.Inlines.Add(new Microsoft.UI.Xaml.Documents.Run { Text = m.Name });
            nameLine.Inlines.Add(new Microsoft.UI.Xaml.Documents.Run { Text = "  " + (m.Input is null ? "Input unknown" : OllamaModels.InputLabel(m.Input)), FontSize = 12, Foreground = Gray });
            left.Children.Add(nameLine);
            if (m.Name != m.Id) left.Children.Add(new TextBlock { Text = m.Id, FontSize = 12, Foreground = Gray, TextTrimming = TextTrimming.CharacterEllipsis });
            left.Children.Add(new TextBlock { Text = m.PriceText, FontSize = 12, Foreground = Gray });
            if (UsedBy(ModelProvider.OpenRouter, m.Id) is { Length: > 0 } used) left.Children.Add(new TextBlock { Text = used, FontSize = 12, Foreground = Accent });
            var right = new StackPanel { Spacing = 2, HorizontalAlignment = HorizontalAlignment.Right };
            right.Children.Add(ZdrIndicator(availability, 12));
            if (m.ContextText.Length > 0) right.Children.Add(new TextBlock { Text = m.ContextText, FontSize = 12, Foreground = Gray, HorizontalAlignment = HorizontalAlignment.Right });
            Grid.SetColumn(right, 1);
            row.Children.Add(left); row.Children.Add(right);
            if (selected)
            {
                var check = new FontIcon { Glyph = "", FontSize = 12, VerticalAlignment = VerticalAlignment.Center };
                Grid.SetColumn(check, 2); row.Children.Add(check);
            }
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(row, $"{m.Name}, {m.PriceText}, {OpenRouterZdrCatalog.Label(availability)}");
            return row;
        }

        void RenderOpenRouter()
        {
            list.Items.Clear();
            footerLeft.Children.Clear();
            if (hosted.Length == 0)
            {
                if (openRouterLoading) { list.Items.Add(Loading("Loading models…")); status.Text = ""; return; }
                list.Items.Add(Muted("Couldn't load OpenRouter models.", 14));
                var retry = new Button { Content = "Try again" };
                retry.Click += async (_, _) => await LoadHosted(force: true);
                list.Items.Add(retry);
                status.Text = "";
                return;
            }
            var query = search.Text.Trim();
            var selectedId = current?.Invoke()?.Trim();
            int matching = 0;
            foreach (var m in hosted)
            {
                if (!m.Matches(query)) continue;
                matching++;
                list.Items.Add(OpenRouterRow(m, m.Id == selectedId));
            }
            if (matching == 0) list.Items.Add(Muted("No matching models", 14));
            status.Text = "";
            if (openRouterLoading) footerLeft.Children.Add(Loading("Checking ZDR availability…"));
            else if (zdr is null)
            {
                var refresh = new Button { Content = "Refresh ZDR availability" };
                refresh.Click += async (_, _) => await LoadHosted(force: true);
                footerLeft.Children.Add(refresh);
            }
        }

        async Task LoadHosted(bool force)
        {
            openRouterLoading = true; if (tab == ModelProvider.OpenRouter) RenderOpenRouter();
            (hosted, zdr) = await OpenRouter.Load(force: force);
            openRouterLoading = false; if (tab == ModelProvider.OpenRouter) RenderOpenRouter();
        }

        void Render()
        {
            options.Children.Clear();
            footerLeft.Children.Clear();
            title.Text = providers.Count == 1 && providers[0] == ModelProvider.OpenRouter ? "OpenRouter models" : "Models";
            if (tab == ModelProvider.OpenRouter)
            {
                // As macOS: the ZDR setting lives with the model field and in Settings; the browser explains the labels.
                options.Children.Add(Muted($"For strict ZDR, choose a model marked {OpenRouterZdrCatalog.Label(OpenRouterZdrAvailability.Available)}. With ZDR requested, allow Safety Retention in OpenRouter to use a Safety Retention-only model."));
                RenderOpenRouter();
            }
            else RenderOllama();
        }

        void Tabs()
        {
            tabs.Children.Clear();
            tabs.Visibility = providers.Count > 1 ? Visibility.Visible : Visibility.Collapsed;
            foreach (var provider in providers)
            {
                var button = new ToggleButton { Content = provider == ModelProvider.OpenRouter ? "OpenRouter" : "Ollama", IsChecked = provider == tab, CornerRadius = new CornerRadius(14), Padding = new Thickness(14, 4, 14, 4) };
                button.Click += (_, _) => { tab = provider; search.Text = ""; Tabs(); Render(); };
                tabs.Children.Add(button);
            }
        }

        flyout.Opening += async (_, _) =>
        {
            tab = startOn?.Invoke() is { } start && providers.Contains(start) ? start : providers[0];
            search.Text = "";
            if (providers.Contains(ModelProvider.OpenRouter) && hosted.Length == 0) openRouterLoading = true;
            Tabs(); Render();
            var openRouterLoad = providers.Contains(ModelProvider.OpenRouter) ? LoadHosted(force: false) : Task.CompletedTask;
            if (providers.Contains(ModelProvider.Ollama))
            {
                installed = await OllamaSetup.Installed(TimeSpan.FromSeconds(3)) ?? [];
                StartLibrary(() => flyout.DispatcherQueue.TryEnqueue(() => { if (flyout.IsOpen && tab == ModelProvider.Ollama) RenderOllama(); }));
            }
            await openRouterLoad;
            Render();
        };
        search.TextChanged += (_, _) => Render();
        list.ItemClick += (_, args) =>
        {
            if ((args.ClickedItem as FrameworkElement)?.Tag is not string picked) return;
            flyout.Hide();
            if (picked.StartsWith("openrouter:", StringComparison.Ordinal)) chosen(ModelProvider.OpenRouter, picked["openrouter:".Length..]);
            else chosen(ModelProvider.Ollama, picked["ollama:".Length..]);
        };
        return flyout;
    }

    /// <summary>Loads the whole Ollama library once per app run (cached 10 minutes by the catalog), calling back as families arrive.</summary>
    private static void StartLibrary(Action changed)
    {
        if (libraryLoad is { IsCompleted: false }) { _ = libraryLoad.ContinueWith(_ => changed()); return; }
        if (libraryLoad is { IsCompletedSuccessfully: true } && familyOrder.Length > 0) return;
        familiesLoaded = 0;
        libraryLoad = Task.Run(async () =>
        {
            try
            {
                familyOrder = await Library.Families();
                var lastShown = DateTime.MinValue;
                await Library.LoadAll((family, variants) =>
                {
                    lock (families) { families[family] = variants; familiesLoaded++; }
                    // Redraw at most twice a second while pages arrive.
                    if (DateTime.UtcNow - lastShown > TimeSpan.FromMilliseconds(500)) { lastShown = DateTime.UtcNow; changed(); }
                });
            }
            catch (Exception error) when (error is LocalAgentException or HttpRequestException or TaskCanceledException) { }
            changed();
        });
    }

    private static FrameworkElement Loading(string text)
    {
        var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        row.Children.Add(new ProgressRing { IsActive = true, Width = 16, Height = 16 });
        row.Children.Add(new TextBlock { Text = text, FontSize = 12, VerticalAlignment = VerticalAlignment.Center });
        return row;
    }

    /// <summary>macOS's ZDR indicator: the label, with an orange warning for Safety Retention only and no public ZDR endpoint.</summary>
    public static FrameworkElement ZdrIndicator(OpenRouterZdrAvailability availability, double size = 14)
    {
        var line = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 4, HorizontalAlignment = HorizontalAlignment.Right };
        if (availability is OpenRouterZdrAvailability.SafetyRetentionOnly or OpenRouterZdrAvailability.Unavailable)
            line.Children.Add(new FontIcon { Glyph = "", FontSize = size, Foreground = Orange, VerticalAlignment = VerticalAlignment.Center });
        line.Children.Add(new TextBlock { Text = OpenRouterZdrCatalog.Label(availability), FontSize = size, FontWeight = Microsoft.UI.Text.FontWeights.Medium, Foreground = Gray, VerticalAlignment = VerticalAlignment.Center });
        return line;
    }

    private static Brush Gray => new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 128, 128, 128));
    private static Brush Green => new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 36, 150, 84));
    private static Brush Orange => new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 220, 120, 20));
    private static Brush Accent => (Brush)Application.Current.Resources["TextFillColorPrimaryBrush"];
}
