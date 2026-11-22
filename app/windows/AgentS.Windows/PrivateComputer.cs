using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using AgentS.Core;

namespace AgentS.Windows;

/// A single app-owned Linux guest: a WSL 2 distro (default, works on Windows Home and ARM64) or,
/// when scripts/setup-private-vm.ps1 configured one, a Hyper-V VM. Ollama runs on the Windows host.
internal sealed class PrivateComputer : IPrivateDesktop
{
    private readonly bool wsl;

    private sealed record Configuration(Guid VmId, string Address, string PrivateKeyPath, string KnownHostsPath);

    private static readonly JsonSerializerOptions ConfigJson = new() { PropertyNameCaseInsensitive = true };
    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];
    private const string Checkpoint = "Agent S clean";
    private static string ConfigPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Simular", "Agent S", "private-vm.json");
    private static string System32 => Environment.GetFolderPath(Environment.SpecialFolder.System);

    private readonly Configuration configuration = null!;

    public static bool IsConfigured => File.Exists(ConfigPath) || WslComputer.IsConfigured;

    public PrivateComputer()
    {
        if (!IsConfigured)
            throw new PrivateComputerException("Set up the Private computer first: choose New agent › This PC › Set up.");
        if (!File.Exists(ConfigPath)) { wsl = true; return; }
        try
        {
            if (new FileInfo(ConfigPath).Length > 32_768)
                throw new InvalidDataException();
            var value = JsonSerializer.Deserialize<Configuration>(File.ReadAllText(ConfigPath), ConfigJson);
            if (value is null || value.VmId == Guid.Empty ||
                !IPAddress.TryParse(value.Address, out var address) || !address.Equals(IPAddress.Parse("192.168.247.2")) ||
                !File.Exists(value.PrivateKeyPath) || !File.Exists(value.KnownHostsPath))
                throw new InvalidDataException();
            configuration = value;
        }
        catch (Exception error) when (error is IOException or JsonException or InvalidDataException or UnauthorizedAccessException)
        {
            throw new PrivateComputerException("The Private computer configuration is unreadable. Check its VM ID, address, and SSH files.");
        }
    }

    public async Task Start(CancellationToken cancellation)
    {
        if (wsl)
        {
            try { _ = await Capture(cancellation); return; }
            catch (PrivateComputerException) { }
            WslComputer.LaunchDesktop();
            await WslComputer.WaitForScreenshot(TimeSpan.FromSeconds(60), cancellation);
            return;
        }
        await HyperV("if ($vm.State -ne 'Running') { Start-VM -VM $vm | Out-Null }", cancellation);
        var start = Stopwatch.StartNew();
        while (start.Elapsed < TimeSpan.FromSeconds(75))
        {
            cancellation.ThrowIfCancellationRequested();
            try
            {
                _ = await Capture(cancellation);
                return;
            }
            catch (PrivateComputerException)
            {
                await Task.Delay(1000, cancellation);
            }
        }
        throw new PrivateComputerException("The Private computer started but its guest desktop did not respond. Check the VM and its internal switch.");
    }

    public Task Stop(CancellationToken cancellation = default) => wsl ? WslComputer.Stop(cancellation) :
        HyperV("if ($vm.State -ne 'Off') { Stop-VM -VM $vm }", cancellation);

    public Task Reset(CancellationToken cancellation = default) => wsl ? WslComputer.Reset(cancellation) :
        HyperV($"$checkpoint = Get-VMSnapshot -VM $vm -Name '{Checkpoint}'; " +
            "if ($vm.State -ne 'Off') { Stop-VM -VM $vm }; Restore-VMSnapshot -VMSnapshot $checkpoint -Confirm:$false",
            cancellation);

    public async Task<byte[]> Capture(CancellationToken cancellation)
    {
        var output = await Guest("screenshot", 5 * 1024 * 1024, cancellation);
        try
        {
            var png = Convert.FromBase64String(output);
            if (!png.AsSpan().StartsWith(PngSignature)) throw new FormatException();
            return png;
        }
        catch (FormatException)
        {
            throw new PrivateComputerException("The Private computer returned an invalid screenshot.");
        }
    }

    public Task<string> Execute(PrivateAction action, CancellationToken cancellation)
    {
        var command = action switch
        {
            { Name: "click", X: >= 0 and < 1024, Y: >= 0 and < 768 } =>
                FormattableString.Invariant($"click {action.X} {action.Y}"),
            { Name: "open" or "type" or "key", Text: not null } =>
                $"{action.Name} {Encode(action.Text)}",
            { Name: "scroll", Text: "up" } => $"scroll {Encode("-1")}",
            { Name: "scroll", Text: "down" } => $"scroll {Encode("1")}",
            _ => throw new PrivateComputerException("The Private agent requested an unsupported desktop action.")
        };
        return Guest(command, 4096, cancellation);
    }

    private static string Encode(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        if (bytes.Length > 4096) throw new PrivateComputerException("The desktop action contains too much text.");
        return bytes.Length == 0 ? "''" : Convert.ToBase64String(bytes);
    }

    private async Task HyperV(string operation, CancellationToken cancellation)
    {
        var script = "$ErrorActionPreference = 'Stop'; " +
            $"$vm = Get-VM -Id '{configuration.VmId.ToString("D", CultureInfo.InvariantCulture)}'; " +
            "if ($vm.Notes -ne 'Agent S Private computer') { throw 'VM is not managed by Agent S.' }; " + operation;
        _ = await Run(Path.Combine(System32, "WindowsPowerShell", "v1.0", "powershell.exe"),
            ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script], 4096, TimeSpan.FromSeconds(75),
            "Hyper-V could not manage the Private computer. Check that it exists and your Windows account has Hyper-V access.", cancellation);
    }

    private Task<string> Guest(string command, int outputLimit, CancellationToken cancellation) => wsl
        ? WslComputer.Guest(command, outputLimit, cancellation)
        : Run(Path.Combine(System32, "OpenSSH", "ssh.exe"),
            ["-F", "NUL", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
             "-o", "StrictHostKeyChecking=yes", "-o", "GlobalKnownHostsFile=NUL",
             "-o", $"UserKnownHostsFile=\"{configuration.KnownHostsPath.Replace('\\', '/')}\"",
             "-o", "ConnectTimeout=4", "-i", configuration.PrivateKeyPath, $"agent@{configuration.Address}",
             $"/usr/local/bin/agent-s-guest {command}"], outputLimit, TimeSpan.FromSeconds(20),
            "Could not communicate with the Private guest. Check its internal switch and SSH setup.", cancellation);

    internal static async Task<string> Run(string executable, string[] arguments, int outputLimit, TimeSpan timeout,
        string failure, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        deadline.CancelAfter(timeout);
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        using var process = new Process { StartInfo = start };
        try
        {
            if (!process.Start()) throw new PrivateComputerException(failure);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            throw new PrivateComputerException($"Required Windows tool {Path.GetFileName(executable)} could not start.");
        }

        try
        {
            var errors = process.StandardError.BaseStream.CopyToAsync(Stream.Null, deadline.Token);
            var result = new StringBuilder();
            var buffer = new char[4096];
            while (await process.StandardOutput.ReadAsync(buffer.AsMemory(), deadline.Token) is var count && count != 0)
            {
                if (result.Length + count > outputLimit) throw new PrivateComputerException("The Private computer returned too much data.");
                result.Append(buffer, 0, count);
            }
            await process.WaitForExitAsync(deadline.Token);
            await errors;
            if (process.ExitCode != 0) throw new PrivateComputerException(failure);
            return result.ToString().Trim();
        }
        catch (OperationCanceledException) when (!cancellation.IsCancellationRequested)
        {
            throw new PrivateComputerException(failure);
        }
        finally
        {
            if (!process.HasExited)
            {
                try { process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { }
            }
        }
    }
}

internal sealed class PrivateComputerException(string message) : Exception(message);
