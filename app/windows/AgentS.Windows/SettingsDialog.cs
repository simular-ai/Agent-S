using System.Net.Http.Headers;
using System.Reflection;
using System.Text.Json;
using AgentS.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;
using global::Windows.System;

namespace AgentS.Windows;

/// <summary>
/// The macOS Settings sheet: Computer, Accounts (each key with "Get a key", how-to, paste, prefix check,
/// "Saved" + Remove), OpenRouter Zero Data Retention, Ollama status, and checking for updates.
/// </summary>
public sealed partial class MainWindow
{
    private sealed record KeyKind(string Service, string Prefix, Uri Page, string HowTo, string AutomationId,
        Func<string?> Read, Action<string> Save, Action Delete);

    private KeyKind[] KeyKinds =>
    [
        new("Sai", "sapi_", new Uri("https://platform.simular.ai/api-keys"), "Sign in, click Create key, then copy it (it's shown once).", "SaiApiKey",
            Secrets.Read, key => _ = UseSaiKey(key), () => _ = UseSaiKey(null)),
        new("Simular Box", "cb_", Box.Portal, "Sign in, click Create key under Keys and devices, then copy it (it's shown once).", "BoxApiKey",
            Secrets.ReadBox, Secrets.SaveBox, Secrets.DeleteBox),
        new("OpenRouter", "sk-or-", new Uri("https://openrouter.ai/keys"), "Sign in, click Create Key, then copy it. Setting a credit limit on it is a good idea.", "OpenRouterApiKey",
            Secrets.ReadOpenRouter, Secrets.SaveOpenRouter, Secrets.DeleteOpenRouter),
    ];

    /// <summary>The gear button: Settings (it was Accounts).</summary>
    private async void Accounts(object sender, RoutedEventArgs e)
    {
        if (busy) return;
        var content = new StackPanel { Spacing = 10, MinWidth = 460 };
        ContentDialog? dialog = null;
        TextBlock Header(string text) => new() { Text = text, FontSize = 15, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Margin = new Thickness(0, 10, 0, 0) };

        // ---- Computer ----
        content.Children.Add(Header("Computer"));
        content.Children.Add(Muted(PrivateComputer.IsConfigured
            ? $"Linux: Debian ({(PrivateComputer.UsesWsl ? "WSL" : "Hyper-V")}), ready"
            : "Linux: not set up yet"));
        if (!PrivateComputer.IsConfigured)
        {
            var setUp = new Button { Content = "Set up the computer" };
            setUp.Click += (_, _) => { dialog?.Hide(); preferPrivate = true; NewAgent(this, new RoutedEventArgs()); };
            content.Children.Add(setUp);
            content.Children.Add(Muted("Downloads Debian and creates the private Linux computer. Windows may need one restart to turn on its virtual machine feature."));
        }
        else if (PrivateComputer.UsesWsl)
        {
            // WSL has one memory limit for every Linux distro on the PC, kept in the person's .wslconfig.
            // Agent S reads it, never changes it, and caps its own computer below it.
            string? wslconfig = null;
            try
            {
                var path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".wslconfig");
                if (File.Exists(path) && new FileInfo(path).Length < 64_000) wslconfig = File.ReadAllText(path);
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException) { }
            var total = (ulong)Math.Max(0, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes);
            var (limit, configured) = WslConfig.MemoryLimit(wslconfig, total);
            // 2/4/6 GB as on macOS, applied through the computer's memory cgroup; "WSL's limit" is no cap of its own
            // (setups from before the choice).
            var wslLimit = $"WSL's limit ({WslConfig.FormatGiB(limit)})";
            var memory = new ComboBox
            {
                Header = "Memory", HorizontalAlignment = HorizontalAlignment.Stretch,
                ItemsSource = PrivateComputer.MemoryChoices.Select(gb => $"{gb} GB").Append(wslLimit).ToArray(),
                SelectedIndex = WslComputer.MemoryGB is { } gb ? Array.IndexOf(PrivateComputer.MemoryChoices, gb) : PrivateComputer.MemoryChoices.Length,
                Description = $"The most the computer's desktop and apps can use. Applies to the next task. It can't exceed WSL's limit of {WslConfig.FormatGiB(limit)}"
                    + (configured ? ", set in .wslconfig" : ", half of this PC's memory") + ", which covers every Linux system on this PC. Ollama runs the model in this PC's memory separately.",
            };
            AutomationProperties.SetAutomationId(memory, "ComputerMemory");
            memory.SelectionChanged += (_, _) =>
            {
                if (memory.SelectedIndex < 0) return;
                AppSettings.Current.ComputerMemoryGB = memory.SelectedIndex < PrivateComputer.MemoryChoices.Length
                    ? PrivateComputer.MemoryChoices[memory.SelectedIndex] : null;
                AppSettings.Current.Save();
            };
            content.Children.Add(memory);
            content.Children.Add(new HyperlinkButton
            {
                Content = "How to change WSL's limit",
                NavigateUri = new Uri("https://learn.microsoft.com/windows/wsl/wsl-config#main-wsl-settings"),
                Padding = new Thickness(0),
            });
        }
        else
        {
            var memory = new ComboBox
            {
                Header = "Memory", HorizontalAlignment = HorizontalAlignment.Stretch,
                ItemsSource = PrivateComputer.MemoryChoices.Select(gb => $"{gb} GB").ToArray(),
                SelectedIndex = Array.IndexOf(PrivateComputer.MemoryChoices, PrivateComputer.MemoryGB),
                Description = "Applies the next time the computer starts. Stop computer to apply it now. Ollama runs the model in this PC's memory separately.",
            };
            AutomationProperties.SetAutomationId(memory, "ComputerMemory");
            memory.SelectionChanged += (_, _) =>
            {
                if (memory.SelectedIndex < 0) return;
                AppSettings.Current.ComputerMemoryGB = PrivateComputer.MemoryChoices[memory.SelectedIndex];
                AppSettings.Current.Save();
            };
            content.Children.Add(memory);
        }

        // ---- Accounts ----
        content.Children.Add(Header("Accounts"));
        foreach (var kind in KeyKinds) content.Children.Add(KeySetup(kind));
        var zdr = ZdrToggle(AppSettings.Current.OpenRouterRequiresZdr);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(zdr, "OpenRouterZdr");
        zdr.Toggled += (_, _) => { AppSettings.Current.OpenRouterRequiresZdr = zdr.IsOn; AppSettings.Current.Save(); UpdateBadges(); };
        content.Children.Add(zdr);
        content.Children.Add(Muted("Safety Retention follows your OpenRouter privacy settings."));
        content.Children.Add(Muted("Keys stay in Windows Credential Locker and are shared by all your agents. They never enter the computer."));

        // ---- Ollama ----
        content.Children.Add(Header("Ollama"));
        var ollamaStatus = new TextBlock { Text = "Unchecked", Opacity = 0.75 };
        var ollamaHelp = Muted("");
        ollamaHelp.Visibility = Visibility.Collapsed;
        var check = new Button { Content = "Check status" };
        async Task CheckOllama()
        {
            check.IsEnabled = false;
            var version = await OllamaSetup.Version(TimeSpan.FromSeconds(3));
            bool ready = version is not null && OllamaModels.IsAtLeast(version, OllamaModels.MinimumVersion);
            ollamaStatus.Text = version is null ? "Not running" : ready ? $"Running {version}" : $"{version}: needs {OllamaModels.MinimumVersion} or newer";
            ollamaStatus.Foreground = ready ? Theme("MutedText") : new Microsoft.UI.Xaml.Media.SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 221, 54, 72));
            ollamaHelp.Text = "Install Ollama from ollama.com and open it. It stays on this PC; the computer reaches it through this app only.";
            ollamaHelp.Visibility = version is null ? Visibility.Visible : Visibility.Collapsed;
            check.Content = "Check again"; check.IsEnabled = true;
        }
        check.Click += async (_, _) => await CheckOllama();
        content.Children.Add(ollamaStatus); content.Children.Add(ollamaHelp); content.Children.Add(check);

        // ---- Updates ----
        content.Children.Add(Header("Updates"));
        var version = Assembly.GetExecutingAssembly().GetName().Version;
        var current = version is null ? "0.0.0" : $"{version.Major}.{version.Minor}.{version.Build}";
        var updateStatus = Muted($"{AppIdentity.Name} {current}");
        var checkUpdates = new Button { Content = "Check for Updates…" };
        var restartUpdate = new Button { Content = "Restart to update", Visibility = Visibility.Collapsed };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(checkUpdates, "CheckForUpdates");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(restartUpdate, "RestartToUpdate");
        void RefreshUpdate()
        {
            if (availableUpdate is not { } available) return;
            updateStatus.Text = $"Agent S {available.TargetFullRelease.Version} is available.";
            restartUpdate.Visibility = Visibility.Visible;
        }
        RefreshUpdate();
        checkUpdates.Click += async (_, _) =>
        {
            checkUpdates.IsEnabled = false;
            updateStatus.Text = "Checking…";
            updateStatus.Text = updateManager is not null
                ? await CheckInstalledUpdate(notify: false) : await CheckForUpdates(current);
            restartUpdate.Visibility = availableUpdate is not null ? Visibility.Visible : Visibility.Collapsed;
            checkUpdates.IsEnabled = true;
        };
        restartUpdate.Click += async (_, _) =>
        {
            checkUpdates.IsEnabled = restartUpdate.IsEnabled = false;
            updateStatus.Text = "Preparing the update…";
            updateStatus.Text = await RestartWithUpdate();
            checkUpdates.IsEnabled = restartUpdate.IsEnabled = true;
        };
        content.Children.Add(updateStatus); content.Children.Add(checkUpdates); content.Children.Add(restartUpdate);

        content.Children.Add(Muted("Ollama runs models on this PC. Simular Box agents run their harness on this PC and use the Box's tools. Agent S adds your OpenRouter key to each OpenRouter request itself."));
        dialog = new ContentDialog
        {
            Title = "Settings", Content = new ScrollViewer { Content = content, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, MaxHeight = 620 },
            CloseButtonText = "Done", XamlRoot = Root.XamlRoot,
        };
        _ = CheckOllama();
        await dialog.ShowAsync();
        UpdateBadges(); Controls();
    }

    /// <summary>One key, as macOS KeySetup shows it: saved (with Remove), or "Get a … key", how-to, then paste and Save.</summary>
    private FrameworkElement KeySetup(KeyKind kind)
    {
        var panel = new StackPanel { Spacing = 6 };
        void Render(string? status = null, bool error = false)
        {
            panel.Children.Clear();
            panel.Children.Add(new TextBlock { Text = kind.Service, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
            if (kind.Read() is { } savedKey)
            {
                var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10 };
                var saved = new TextBlock { Text = "✓ Saved", Foreground = Green, VerticalAlignment = VerticalAlignment.Center };
                Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(saved, kind.AutomationId + "Saved");
                row.Children.Add(saved);
                var remove = new Button { Content = "Remove", Padding = new Thickness(10, 3, 10, 3) };
                remove.Click += (_, _) =>
                {
                    try { kind.Delete(); Render($"{kind.Service} key removed."); }
                    catch (Exception failure) { Render(Describe(failure), true); }
                };
                row.Children.Add(remove);
                panel.Children.Add(row);
                if (kind.AutomationId == "OpenRouterApiKey") panel.Children.Add(OpenRouterUsageControls(savedKey));
            }
            else
            {
                var getKey = new Button { Content = $"1. Get a {kind.Service} key" };
                getKey.Click += async (_, _) => await Launcher.LaunchUriAsync(kind.Page);
                panel.Children.Add(getKey);
                panel.Children.Add(Muted(kind.HowTo));
                var field = new PasswordBox { PlaceholderText = kind.Prefix + "…", Header = "2. Paste it here" };
                Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(field, kind.AutomationId);
                var save = new Button { Content = "Save" };
                void Commit()
                {
                    var key = field.Password.Trim();
                    field.Password = "";
                    if (key.Length == 0) return;
                    if (!key.StartsWith(kind.Prefix, StringComparison.Ordinal))
                    { Render($"That doesn't look like a {kind.Service} key (they start with {kind.Prefix}).", true); return; }
                    try { kind.Save(key); Render(); }
                    catch (Exception failure) { Render(Describe(failure), true); }
                }
                save.Click += (_, _) => Commit();
                field.KeyDown += (_, args) => { if (args.Key == VirtualKey.Enter) { args.Handled = true; Commit(); } };
                var row = new Grid { ColumnSpacing = 8 };
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                row.Children.Add(field);
                save.VerticalAlignment = VerticalAlignment.Bottom;
                Grid.SetColumn(save, 1); row.Children.Add(save);
                panel.Children.Add(row);
            }
            if (status is not null)
            {
                var note = Muted(status);
                if (error) { note.Foreground = Red; note.Opacity = 1; }
                panel.Children.Add(note);
            }
        }
        Render();
        return new Border { Child = panel, Padding = new Thickness(12), CornerRadius = new CornerRadius(10), Background = Theme("ChipBackground") };
    }


    private static FrameworkElement OpenRouterUsageControls(string key)
    {
        var usage = new StackPanel { Spacing = 6 };
        var result = Muted("");
        result.Visibility = Visibility.Collapsed;
        AutomationProperties.SetLiveSetting(result, AutomationLiveSetting.Polite);
        var check = new Button { Content = "Check OpenRouter spend" };
        check.Click += async (_, _) =>
        {
            check.IsEnabled = false;
            result.Visibility = Visibility.Visible;
            result.Text = "Checking OpenRouter spend...";
            try
            {
                var spend = await OpenRouterKeyUsage.Read(key);
                result.Text = $"Spent today: {OpenRouterKeyUsage.FormatSpend(spend.Daily)}\n"
                    + $"Spent this week: {OpenRouterKeyUsage.FormatSpend(spend.Weekly)}\n"
                    + $"Spent this month: {OpenRouterKeyUsage.FormatSpend(spend.Monthly)}\n"
                    + $"Total spent: {OpenRouterKeyUsage.FormatSpend(spend.Total)}";
                check.Content = "Refresh OpenRouter spend";
            }
            catch (HttpRequestException error) when (error.StatusCode is System.Net.HttpStatusCode.Unauthorized
                or System.Net.HttpStatusCode.Forbidden)
            {
                result.Text = "OpenRouter rejected the saved key.";
            }
            catch (Exception error) when (error is HttpRequestException or JsonException or IOException or OperationCanceledException)
            {
                result.Text = "Could not load OpenRouter spend. Try again.";
            }
            finally
            {
                check.IsEnabled = true;
            }
        };
        usage.Children.Add(check);
        usage.Children.Add(result);
        return usage;
    }

    /// <summary>
    /// "Check for Updates…": the newest GitHub release of Agent S. When it has a Windows download, offers it;
    /// installing stays a choice the person makes.
    /// </summary>
    private async Task<string> CheckForUpdates(string current)
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
            http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("Agent-S-Windows", current));
            http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
            using var document = JsonDocument.Parse(await http.GetStringAsync("https://api.github.com/repos/simular-ai/Agent-S/releases/latest"));
            var root = document.RootElement;
            var tag = root.TryGetProperty("tag_name", out var t) ? t.GetString() ?? "" : "";
            var latest = tag.TrimStart('v');
            if (latest.Length == 0) return "Couldn't read the latest release.";
            if (OllamaModels.IsAtLeast(current, latest)) return $"Agent S {current} is up to date.";
            var asset = root.TryGetProperty("assets", out var assets) && assets.ValueKind == JsonValueKind.Array
                ? assets.EnumerateArray().Where(a => a.TryGetProperty("name", out var n)
                    && n.GetString() is "AgentS-Windows-x64-Setup.exe" or "AgentS-Windows-x64.zip")
                    .OrderByDescending(a => a.GetProperty("name").GetString() == "AgentS-Windows-x64-Setup.exe")
                    .FirstOrDefault()
                : default;
            var page = asset.ValueKind == JsonValueKind.Object && asset.TryGetProperty("browser_download_url", out var u) ? u.GetString()
                : root.TryGetProperty("html_url", out var h) ? h.GetString() : null;
            if (page is null || !Uri.TryCreate(page, UriKind.Absolute, out var uri) || uri.Scheme != "https" || uri.Host != "github.com")
                return $"Agent S {latest} is available.";
            var open = await Dialog("Update available", $"Agent S {latest} is available (you have {current}).{(asset.ValueKind == JsonValueKind.Object ? "" : " It doesn't include a Windows download yet; the release page has the details.")}",
                asset.ValueKind == JsonValueKind.Object ? "Download" : "Open release page").ShowAsync();
            if (open == ContentDialogResult.Primary) await Launcher.LaunchUriAsync(uri);
            return $"Agent S {latest} is available.";
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException or JsonException)
        {
            return "Couldn't check for updates. Check your connection and try again.";
        }
    }
}
