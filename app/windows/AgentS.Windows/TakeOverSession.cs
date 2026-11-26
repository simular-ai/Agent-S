using System.Diagnostics;
using System.Text;
using AgentS.Core;

namespace AgentS.Windows;

/// <summary>
/// A take-over of the WSL Private computer: one wsl.exe sends the person's input to the guest's xdotool
/// loop, another streams the display back as PNG frames. Disposing ends both.
/// </summary>
internal sealed class TakeOverSession : IDisposable
{
    private static string WslExe => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "wsl.exe");
    private readonly Process input, frames;
    private readonly SemaphoreSlim writeGate = new(1, 1);
    private readonly CancellationTokenSource stop = new();

    /// <summary>Called off the UI thread with each new frame.</summary>
    public event Action<byte[]>? Frame;
    public bool IsAlive => !input.HasExited && !frames.HasExited;

    public TakeOverSession()
    {
        input = Start(TakeOverInput.GuestLoop);
        frames = Start(TakeOverInput.FrameLoop);
        _ = input.StandardOutput.BaseStream.CopyToAsync(Stream.Null);
        _ = input.StandardError.BaseStream.CopyToAsync(Stream.Null);
        _ = frames.StandardError.BaseStream.CopyToAsync(Stream.Null);
        _ = Task.Run(ReadFrames);
    }

    private static Process Start(string script)
    {
        var start = new ProcessStartInfo(WslExe)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.ASCII, StandardInputEncoding = new UTF8Encoding(false),
        };
        foreach (var argument in new[] { "-d", WslComputer.Distro, "-u", "agent", "--cd", "~", "--exec", "/bin/bash", "-c", script.Replace("\r\n", "\n") })
            start.ArgumentList.Add(argument);
        return Process.Start(start) ?? throw new PrivateComputerException("Could not reach the Private computer in WSL.");
    }

    private async Task ReadFrames()
    {
        try
        {
            while (!stop.IsCancellationRequested && await frames.StandardOutput.ReadLineAsync(stop.Token) is { } line)
            {
                if (line.Length == 0) continue;
                try { Frame?.Invoke(Convert.FromBase64String(line)); } catch (FormatException) { }
            }
        }
        catch (Exception error) when (error is OperationCanceledException or IOException or ObjectDisposedException) { }
    }

    /// <summary>Sends one line from <see cref="TakeOverInput"/>; inputs arrive in order.</summary>
    public async Task Send(string? line)
    {
        if (line is null || stop.IsCancellationRequested || input.HasExited) return;
        await writeGate.WaitAsync();
        try { await input.StandardInput.WriteAsync(line + "\n"); await input.StandardInput.FlushAsync(); }
        catch (Exception error) when (error is IOException or ObjectDisposedException or InvalidOperationException) { }
        finally { writeGate.Release(); }
    }

    public void Dispose()
    {
        stop.Cancel();
        foreach (var process in new[] { input, frames })
        {
            try { if (!process.HasExited) process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            process.Dispose();
        }
    }
}
