using Microsoft.UI.Xaml;
using Velopack;

namespace AgentS.Windows;

public sealed partial class MainWindow
{
    private UpdateManager? updateManager;
    private UpdateInfo? availableUpdate;
    private DispatcherTimer? updateTimer;
    private bool checkingUpdate, applyingUpdate, restartingForUpdate;
    private string? notifiedVersion;

    private void StartUpdateChecks()
    {
        if (updateTimer is not null || !storageReady) return;
        try { updateManager = WindowsUpdates.ForInstalledApp(); }
        catch (Exception error)
        {
            System.Diagnostics.Trace.TraceWarning($"Could not initialize the updater: {error}");
            return;
        }
        if (updateManager is null) return;
        updateTimer = new DispatcherTimer { Interval = TimeSpan.FromHours(6) };
        updateTimer.Tick += async (_, _) => await CheckInstalledUpdate(notify: true);
        updateTimer.Start();
        _ = CheckInstalledUpdate(notify: true);
    }

    private async Task<string> CheckInstalledUpdate(bool notify)
    {
        if (updateManager is not { } manager) return "Install Agent S to use in-app updates.";
        if (checkingUpdate || applyingUpdate) return "An update check is already in progress.";
        checkingUpdate = true;
        try
        {
            availableUpdate = await manager.CheckForUpdatesAsync();
            if (availableUpdate is not { } update) return "Agent S is up to date.";
            var version = update.TargetFullRelease.Version.ToString();
            if (notify && notifiedVersion != version && !running && !busy)
            {
                notifiedVersion = version;
                Show($"Agent S {version} is available. Open Settings to update.", false);
            }
            return $"Agent S {version} is available.";
        }
        catch (Exception error)
        {
            System.Diagnostics.Trace.TraceWarning($"Agent S update check failed: {error}");
            return "Couldn't check for updates. Try again from Settings.";
        }
        finally { checkingUpdate = false; }
    }

    private bool CanRestartForUpdate =>
        storageReady && !running && !busy && !closing
        && !followUps.Values.Any(queue => queue.Items.Count != 0)
        && string.IsNullOrWhiteSpace(Composer.Text)
        && !drafts.Any(draft => draft.Key != selected && !string.IsNullOrWhiteSpace(draft.Value));

    private async Task<string> RestartWithUpdate()
    {
        if (updateManager is not { } manager || availableUpdate is not { } update || applyingUpdate)
            return "Check for updates first.";
        const string wait = "Finish your active or queued work and clear any drafts before restarting.";
        if (!CanRestartForUpdate) return wait;
        applyingUpdate = true;
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(15));
            await manager.DownloadUpdatesAsync(update, cancelToken: timeout.Token);
            if (!CanRestartForUpdate) return wait;

            // Hold the idle state until history is safely on disk and Velopack is waiting for exit.
            restartingForUpdate = busy = true;
            Controls();
            try
            {
                await store.Save(conversations);
                manager.WaitExitThenApplyUpdates(update.TargetFullRelease);
                allowClose = true;
                Close();
                return "Restarting Agent S…";
            }
            finally { if (!allowClose) { restartingForUpdate = busy = false; Controls(); } }
        }
        catch (Exception error)
        {
            System.Diagnostics.Trace.TraceWarning($"Agent S could not apply the update: {error}");
            return "Couldn't prepare the update. Agent S is still open. Try again from Settings.";
        }
        finally { applyingUpdate = false; }
    }
}
