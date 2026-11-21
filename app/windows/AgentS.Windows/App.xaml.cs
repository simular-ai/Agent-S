using Microsoft.UI.Xaml;

namespace AgentS.Windows;

public partial class App : Application
{
    private MainWindow? window;
    public App() => InitializeComponent();
    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        window = new MainWindow();
        window.Activate();
    }
}
