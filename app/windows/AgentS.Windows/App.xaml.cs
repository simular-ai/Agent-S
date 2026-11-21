using Microsoft.UI.Xaml;

namespace AgentS.Windows;

public partial class App : Application
{
    private MainWindow? window;
    public App()
    {
        // Opt-in startup-only diagnostics for the credential-free Windows CI smoke run.
        // Leave the exception unhandled: the smoke check must fail on a startup crash.
        UnhandledException += (_, args) => WriteStartupFailure(args.Exception.ToString());
        try { InitializeComponent(); }
        catch (Exception error) { WriteStartupFailure(error.ToString()); throw; }
    }
    private bool startup = true;
    private void WriteStartupFailure(string text)
    {
        if (!startup || Environment.GetEnvironmentVariable("AGENT_S_STARTUP_DIAGNOSTICS") is not { } path) return;
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
        catch (Exception error) { WriteStartupFailure(error.ToString()); throw; }
    }
}
