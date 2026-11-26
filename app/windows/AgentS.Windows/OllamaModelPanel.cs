using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>
/// The macOS app's OllamaModelSetup + OllamaModelDetails: whether Ollama is running and new enough, a
/// Download button with progress for the chosen model, its size and inputs, and a memory warning.
/// </summary>
internal sealed class OllamaModelPanel : StackPanel
{
    private static readonly OllamaCatalog Catalog = new();
    /// <summary>One download at a time across the app, like macOS.</summary>
    private static string? downloading;

    private readonly TextBlock status = Muted(""), details = Muted(""), warning = new() { TextWrapping = TextWrapping.Wrap, FontSize = 12 };
    private readonly ProgressBar progress = new() { Visibility = Visibility.Collapsed, Maximum = 100 };
    private readonly Button primary = new(), checkAgain = new() { Content = "Check again" };
    private readonly StackPanel buttons = new() { Orientation = Orientation.Horizontal, Spacing = 8 };
    private readonly Func<ulong> sandboxMemory;
    private string? version;
    private InstalledModel[]? installed;
    private bool checking, busy;
    private string? failure;
    private string? size;

    public string Model { get; private set; }
    /// <summary>Ollama is running, new enough for the model, and has it.</summary>
    public bool IsReady { get; private set; }
    public event Action? Changed;

    /// <param name="sandboxMemory">Memory a Private computer that is still off will take when it starts (0 when none).</param>
    public OllamaModelPanel(string model, Func<ulong> sandboxMemory)
    {
        Model = model; this.sandboxMemory = sandboxMemory;
        Spacing = 8;
        warning.Foreground = new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 220, 120, 20));
        if (Application.Current.Resources.TryGetValue("AccentButtonStyle", out var accent) && accent is Style style) primary.Style = style;
        buttons.Children.Add(primary); buttons.Children.Add(checkAgain);
        Children.Add(status); Children.Add(progress); Children.Add(buttons); Children.Add(details); Children.Add(warning);
        primary.Click += async (_, _) => await Act();
        checkAgain.Click += async (_, _) => await Refresh();
        Loaded += async (_, _) => await Refresh();
        Render();
    }

    public async Task Use(string model)
    {
        if (OllamaModels.Same(model, Model)) return;
        Model = model; size = null; failure = null;
        Render();
        await Refresh();
    }

    public async Task Refresh()
    {
        if (checking) return;
        checking = true; Render();
        try
        {
            version = await OllamaSetup.Version(TimeSpan.FromSeconds(3));
            installed = version is null ? null : await OllamaSetup.Installed();
        }
        finally { checking = false; }
        Render();
        await LoadSize();
    }

    private InstalledModel? Local => installed?.FirstOrDefault(m => OllamaModels.Same(m.Name, Model) && !m.Remote);
    private bool NewEnough => version is not null && OllamaModels.IsAtLeast(version, OllamaModels.MinimumVersionFor(Model));

    private void Render()
    {
        bool mine = downloading is not null && OllamaModels.Same(downloading, Model);
        var name = OllamaModels.DisplayName(Model);
        IsReady = NewEnough && Local is not null;
        progress.Visibility = mine ? Visibility.Visible : Visibility.Collapsed;
        checkAgain.Visibility = Visibility.Collapsed;
        primary.Visibility = Visibility.Visible;
        primary.IsEnabled = !busy && downloading is null;
        if (mine) { primary.Visibility = Visibility.Collapsed; }
        else if (checking && version is null) { status.Text = "Checking Ollama…"; primary.Visibility = Visibility.Collapsed; }
        else if (version is null)
        {
            status.Text = OllamaSetup.IsInstalled ? "Start Ollama to run this model on this PC." : "Ollama runs this model on this PC. Agent S can install it for you.";
            primary.Content = busy ? "Working…" : OllamaSetup.IsInstalled ? "Open Ollama" : "Install Ollama";
            checkAgain.Visibility = Visibility.Visible;
        }
        else if (!NewEnough)
        {
            status.Text = $"Ollama {OllamaModels.MinimumVersionFor(Model)} or later is required (this PC has {version}).";
            primary.Content = busy ? "Updating…" : "Update Ollama";
            checkAgain.Visibility = Visibility.Visible;
        }
        else if (Local is null)
        {
            status.Text = failure ?? "";
            primary.Content = $"Download {name}";
        }
        else { status.Text = "Model ready"; primary.Visibility = Visibility.Collapsed; }
        if (failure is not null && Local is null) status.Text = failure;
        status.Visibility = status.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        buttons.Visibility = primary.Visibility == Visibility.Visible || checkAgain.Visibility == Visibility.Visible ? Visibility.Visible : Visibility.Collapsed;

        // Details: input, size, and whether it may not fit in memory.
        var input = OllamaModels.InputLabel(OllamaModels.SuggestedInput(Model));
        var sizeText = Local is { Size: > 0 } local ? $"Installed · {OllamaModels.FormatBytes(local.Size)}"
            : size is not null ? $"Estimated download · {size}"
            : OllamaModels.PresetSize(Model) is { } preset ? $"Estimated download · {preset}" : "Checking download size…";
        details.Text = string.Join(" · ", new[] { input, sizeText }.Where(s => s.Length > 0));
        var estimate = Local is { Size: > 0 } l ? OllamaMemoryEstimate.FromBytes((ulong)l.Size)
            : OllamaMemoryEstimate.FromCatalogSize(size ?? OllamaModels.PresetSize(Model) ?? "");
        var available = OllamaSetup.AvailableMemory();
        bool tooBig = estimate is not null && available is { } free && estimate.ExceedsBudget(free, sandboxMemory());
        warning.Text = tooBig ? OllamaMemoryEstimate.Warning(available!.Value) : "";
        warning.Visibility = tooBig ? Visibility.Visible : Visibility.Collapsed;
        Changed?.Invoke();
    }

    private async Task LoadSize()
    {
        if (Local is not null || size is not null) return;
        var model = Model;
        var family = OllamaModels.NormalizedName(model).Split(':')[0];
        try
        {
            var variants = await Catalog.Variants(family);
            var match = variants.FirstOrDefault(v => OllamaModels.Same(v.Name, model.Contains(':') ? model : model + ":latest") || v.Name == model);
            size = match?.Size;
        }
        catch (Exception error) when (error is LocalAgentException or HttpRequestException or TaskCanceledException) { }
        if (Model == model) { if (size is null && OllamaModels.PresetSize(model) is null) details.Text = "Download size unavailable"; else Render(); }
    }

    private async Task Act()
    {
        if (busy || downloading is not null) return;
        busy = true; failure = null; Render();
        try
        {
            if (version is null)
            {
                if (!OllamaSetup.IsInstalled && !await OllamaSetup.Install(CancellationToken.None))
                {
                    await global::Windows.System.Launcher.LaunchUriAsync(OllamaSetup.DownloadPage);
                    failure = "Install Ollama from the page that just opened, then choose Check again.";
                    return;
                }
                if (!await OllamaSetup.Start(CancellationToken.None)) failure = "Ollama didn't start. Open Ollama from the Start menu, then choose Check again.";
            }
            else if (!NewEnough)
            {
                if (!await OllamaSetup.Upgrade(CancellationToken.None))
                {
                    await global::Windows.System.Launcher.LaunchUriAsync(OllamaSetup.DownloadPage);
                    failure = "Install the new Ollama from the page that just opened, then choose Check again.";
                    return;
                }
                await OllamaSetup.Start(CancellationToken.None);
            }
            else if (Local is null)
            {
                downloading = Model; progress.Value = 0; progress.IsIndeterminate = true; Render();
                await OllamaSetup.Pull(Model, new Progress<(string Step, double? Percent)>(update =>
                {
                    status.Text = update.Step; status.Visibility = Visibility.Visible;
                    progress.IsIndeterminate = update.Percent is null;
                    if (update.Percent is { } percent) progress.Value = percent;
                }), CancellationToken.None);
            }
        }
        catch (Exception error) when (error is PrivateComputerException or HttpRequestException or IOException or System.Text.Json.JsonException)
        {
            failure = error is PrivateComputerException ? error.Message : $"Ollama didn't finish downloading {OllamaModels.DisplayName(Model)}. Try again.";
        }
        finally
        {
            downloading = null; busy = false;
            version = await OllamaSetup.Version(TimeSpan.FromSeconds(3));
            installed = version is null ? null : await OllamaSetup.Installed();
            Render();
        }
    }

    private static TextBlock Muted(string text) => new() { Text = text, TextWrapping = TextWrapping.Wrap, FontSize = 12, Opacity = 0.75 };

    /// <summary>"Browse models…": the full Ollama list in the model browser (sizes, installed, memory, which agents use each).</summary>
    public static Button BrowseButton(string label, Action<string> chosen) =>
        ModelBrowser.Button(label, [ModelProvider.Ollama], (_, model) => chosen(model));
}
