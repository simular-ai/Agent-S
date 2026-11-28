using System.Reflection;

namespace AgentS.Windows;

/// <summary>
/// Which Agent S this build is. Local and CI builds are "Agent S Dev"; the release workflow builds
/// "Agent S" with <c>-p:AgentSChannel=Release</c>. The two keep separate conversations, settings and
/// Private computers so a development build never touches the installed app's state. API keys in
/// Credential Locker are shared: they belong to the person, not the build.
/// </summary>
internal static class AppIdentity
{
    public static bool IsRelease { get; } = string.Equals(
        Assembly.GetExecutingAssembly().GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(attribute => attribute.Key == "AgentSChannel")?.Value,
        "Release", StringComparison.OrdinalIgnoreCase);

    /// <summary>"Agent S" or "Agent S Dev": the window title and the data folder's name.</summary>
    public static string Name => IsRelease ? "Agent S" : "Agent S Dev";

    /// <summary>%LOCALAPPDATA%\Simular\Agent S (or Agent S Dev): conversations, settings, the Private computer and CLI profiles.</summary>
    public static string LocalFolder => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Simular", Name);

    /// <summary>The WSL distro that is this build's Private computer.</summary>
    public static string WslDistro => IsRelease ? "AgentS-Private" : "AgentS-Dev-Private";

    /// <summary>Window icon and sidebar logo; development builds carry a DEV badge.</summary>
    public static string IconFile => IsRelease ? "AgentS.ico" : "AgentS-Dev.ico";
    public static string LogoFile => IsRelease ? "AgentS.png" : "AgentS-Dev.png";
}
