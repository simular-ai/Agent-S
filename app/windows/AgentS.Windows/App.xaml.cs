using Microsoft.UI.Xaml;

namespace AgentS.Windows;

public partial class App : Application
{
    private MainWindow? window;
    public App()
    {
        // Opt-in startup-only diagnostics for the credential-free Windows CI smoke run.
        // Leave the exception unhandled: the smoke check must fail on a startup crash.
        UnhandledException += (_, args) =>
        {
            WriteStartupFailure(args.Exception);
            try
            {
                var log = Path.Combine(AppIdentity.LocalFolder, "crash.log");
                File.AppendAllText(log, $"{DateTimeOffset.Now:u}\n{args.Exception}\n\n");
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        };
        // Keep the embedded Sai screen's browser data in the user profile, not next to the executable.
        Environment.SetEnvironmentVariable("WEBVIEW2_USER_DATA_FOLDER", Path.Combine(AppIdentity.LocalFolder, "WebView2"));
        try { InitializeComponent(); }
        catch (Exception error) { WriteStartupFailure(error); throw; }
    }
    private bool startup = true;
    private void WriteStartupFailure(Exception error)
    {
        if (!startup || Environment.GetEnvironmentVariable("AGENT_S_STARTUP_DIAGNOSTICS") is not { } path) return;
        var text = error.ToString();
        foreach (System.Collections.DictionaryEntry detail in error.Data)
            if (detail.Value is string value) text += $"\n{detail.Key}: {value}";
        try { File.WriteAllText(path, text); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            window = new MainWindow();
            window.Activate();
            startup = false;
        }
        catch (Exception error) { WriteStartupFailure(error); throw; }
    }
}
