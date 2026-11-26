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

    /// <summary>A button that opens the browser; <paramref name="chosen"/> gets the provider and model id.</summary>
    public static Button Button(string label, IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn = null)
    {
        var button = new Button { Content = label };
        Attach(button, providers, chosen, startOn);
        return button;
    }

    /// <summary>Opens the browser at <paramref name="anchor"/> (for the model menu above the chat).</summary>
    public static void ShowAt(FrameworkElement anchor, IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn = null) =>
        Build(providers, chosen, startOn).ShowAt(anchor);

    public static void Attach(Button button, IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn = null) =>
        button.Flyout = Build(providers, chosen, startOn);

    private static Flyout Build(IReadOnlyList<ModelProvider> providers, Action<ModelProvider, string> chosen, Func<ModelProvider>? startOn)
    {
        var tabs = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        var search = new TextBox { PlaceholderText = "Search by name, provider, or model ID" };
        var options = new StackPanel { Spacing = 6 };
        var status = Muted("");
        var list = new ListView { SelectionMode = ListViewSelectionMode.None, IsItemClickEnabled = true, Height = 420 };
        var panel = new StackPanel { Spacing = 8, Width = 560 };
        panel.Children.Add(tabs); panel.Children.Add(search); panel.Children.Add(options); panel.Children.Add(status); panel.Children.Add(list);
        // Flyouts are capped near 456px wide by default; the lists need the room.
        var presenter = new Style(typeof(FlyoutPresenter));
        presenter.Setters.Add(new Setter(FrameworkElement.MaxWidthProperty, 620.0));
        presenter.Setters.Add(new Setter(FrameworkElement.MaxHeightProperty, 760.0));
        presenter.Setters.Add(new Setter(ScrollViewer.HorizontalScrollBarVisibilityProperty, ScrollBarVisibility.Disabled));
        var flyout = new Flyout { Content = panel, Placement = FlyoutPlacementMode.Bottom, FlyoutPresenterStyle = presenter };
        var tab = providers[0];
        InstalledModel[] installed = [];
        OpenRouterModel[] hosted = [];
        OpenRouterZdrCatalog? zdr = null;
        bool zdrOnly = AppSettings.Current.OpenRouterRequiresZdr;

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

        void RenderOpenRouter()
        {
            list.Items.Clear();
            var query = search.Text.Trim();
            int shown = 0, matching = 0;
            foreach (var m in hosted)
            {
                if (!m.Matches(query)) continue;
                var availability = zdr?.Availability(m.Id, isPublicModel: true) ?? OpenRouterZdrAvailability.Unknown;
                if (zdrOnly && availability != OpenRouterZdrAvailability.Available) continue;
                matching++;
                if (shown >= MaxRows) continue;
                var label = OpenRouterZdrCatalog.Label(availability);
                Brush color = availability == OpenRouterZdrAvailability.Available ? Green : AppSettings.Current.OpenRouterRequiresZdr ? Orange : Gray;
                list.Items.Add(Row($"{m.Name}  ({m.Id})",
                [
                    (OllamaModels.InputLabel(m.Input), null), (m.ContextText, null), (m.PriceText, null), (label, color),
                    (UsedBy(ModelProvider.OpenRouter, m.Id), Accent),
                ], "openrouter:" + m.Id));
                shown++;
            }
            status.Text = hosted.Length == 0 ? "Loading OpenRouter models…"
                : (matching > shown ? $"Showing {shown} of {matching} models; search to narrow." : $"{matching} models.")
                  + (zdr is null ? " ZDR availability couldn't be checked." : "");
        }

        void Render()
        {
            options.Children.Clear();
            if (tab == ModelProvider.OpenRouter)
            {
                // ZDR is requested for every OpenRouter agent (the proxy adds it to each request), as on macOS.
                var request = new ToggleSwitch { Header = "Request Zero Data Retention (ZDR)", IsOn = AppSettings.Current.OpenRouterRequiresZdr, OnContent = "On · applies to all OpenRouter agents", OffContent = "Off" };
                request.Toggled += (_, _) => { AppSettings.Current.OpenRouterRequiresZdr = request.IsOn; AppSettings.Current.Save(); zdrOnly = request.IsOn; Render(); };
                var only = new CheckBox { Content = "Only models with a ZDR endpoint", IsChecked = zdrOnly };
                only.Checked += (_, _) => { zdrOnly = true; RenderOpenRouter(); };
                only.Unchecked += (_, _) => { zdrOnly = false; RenderOpenRouter(); };
                options.Children.Add(request); options.Children.Add(only);
                if (AppSettings.Current.OpenRouterRequiresZdr) options.Children.Add(Muted("With ZDR requested, OpenRouter may reject models without a ZDR endpoint."));
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
            zdrOnly = AppSettings.Current.OpenRouterRequiresZdr;
            Tabs(); Render();
            if (providers.Contains(ModelProvider.Ollama))
            {
                installed = await OllamaSetup.Installed(TimeSpan.FromSeconds(3)) ?? [];
                StartLibrary(() => flyout.DispatcherQueue.TryEnqueue(() => { if (flyout.IsOpen && tab == ModelProvider.Ollama) RenderOllama(); }));
            }
            if (providers.Contains(ModelProvider.OpenRouter)) (hosted, zdr) = await OpenRouter.Load();
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

    private static Brush Gray => new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 128, 128, 128));
    private static Brush Green => new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 36, 150, 84));
    private static Brush Orange => new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 220, 120, 20));
    private static Brush Accent => (Brush)Application.Current.Resources["TextFillColorPrimaryBrush"];
}
