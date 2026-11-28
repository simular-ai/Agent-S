using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Velopack;

namespace AgentS.Windows;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        // Installer hooks run before WinUI starts. A downloaded update needs an explicit restart.
        VelopackApp.Build().SetAutoApplyOnStartup(false).Run();
        WinRT.ComWrappersSupport.InitializeComWrappers();
        Application.Start(context =>
        {
            SynchronizationContext.SetSynchronizationContext(
                new DispatcherQueueSynchronizationContext(DispatcherQueue.GetForCurrentThread()));
            _ = new App();
        });
    }
}
