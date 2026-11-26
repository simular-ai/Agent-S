using System.Text;
using System.Text.RegularExpressions;

namespace AgentS.Core;

/// <summary>
/// "Take over" for the Private computer, as on macOS: the person's mouse and keyboard reach the guest's
/// 1024×768 X display. Each input becomes one line for a small guest loop that runs xdotool. Only these
/// shapes are produced (numbers, a fixed set of key names, base64 text), and the guest loop checks them again.
/// </summary>
public static class TakeOverInput
{
    public const int ScreenWidth = 1024, ScreenHeight = 768;
    public const int MaxTextBytes = 4096;

    private static readonly HashSet<string> Modifiers = ["ctrl", "alt", "shift", "super"];
    private static readonly HashSet<string> Named =
    [
        "Return", "Tab", "Escape", "BackSpace", "Delete", "Insert", "Home", "End", "Page_Up", "Page_Down",
        "Left", "Right", "Up", "Down", "space", "Menu",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
        "minus", "equal", "comma", "period", "slash", "semicolon", "apostrophe", "bracketleft", "bracketright", "backslash", "grave",
    ];
    private static readonly Regex Plain = new("^[a-z0-9]$");

    /// <summary>Maps a point on the shown screen (letterboxed with Stretch=Uniform) to guest pixels, or null outside the picture.</summary>
    public static (int X, int Y)? ToScreen(double x, double y, double viewWidth, double viewHeight)
    {
        if (viewWidth <= 0 || viewHeight <= 0) return null;
        var scale = Math.Min(viewWidth / ScreenWidth, viewHeight / ScreenHeight);
        double left = (viewWidth - ScreenWidth * scale) / 2, top = (viewHeight - ScreenHeight * scale) / 2;
        double gx = (x - left) / scale, gy = (y - top) / scale;
        if (gx < 0 || gy < 0 || gx >= ScreenWidth || gy >= ScreenHeight) return null;
        return ((int)gx, (int)gy);
    }

    public static string Move(int x, int y) =>
        FormattableString.Invariant($"move {Math.Clamp(x, 0, ScreenWidth - 1)} {Math.Clamp(y, 0, ScreenHeight - 1)}");

    /// <param name="button">1 left, 2 middle, 3 right.</param>
    public static string Button(int button, bool down) =>
        button is >= 1 and <= 3 ? $"{(down ? "down" : "up")} {button}" : throw new ArgumentOutOfRangeException(nameof(button));

    /// <summary>One wheel notch: positive <paramref name="delta"/> scrolls up (X buttons 4–7).</summary>
    public static string Wheel(int delta, bool horizontal) => $"wheel {(horizontal ? (delta > 0 ? 7 : 6) : (delta > 0 ? 4 : 5))}";

    /// <summary>"ctrl+shift+Tab": modifiers then one key from the allowed set, or null if anything else.</summary>
    public static string? Key(IEnumerable<string> modifiers, string key)
    {
        var mods = modifiers.Distinct().ToArray();
        if (mods.Any(m => !Modifiers.Contains(m))) return null;
        if (!Named.Contains(key) && !Plain.IsMatch(key)) return null;
        return "key " + string.Join('+', mods.Append(key));
    }

    /// <summary>Typed text, base64 so any character arrives intact; null when empty or too long.</summary>
    public static string? Text(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        return bytes.Length is 0 or > MaxTextBytes ? null : "text " + Convert.ToBase64String(bytes);
    }

    /// <summary>The guest loop: reads one input per line and runs xdotool on the Private computer's display.</summary>
    public const string GuestLoop = """
        if [[ -r /etc/agent-s/env ]]; then . /etc/agent-s/env; fi
        export DISPLAY=":${AGENT_S_DISPLAY:-0}"; unset WAYLAND_DISPLAY
        export XAUTHORITY="$HOME/.local/state/agent-s/Xauthority"
        while IFS=' ' read -r cmd a b; do
          case "$cmd" in
            move) [[ $a =~ ^[0-9]{1,4}$ && $b =~ ^[0-9]{1,4}$ ]] && xdotool mousemove -- "$a" "$b" ;;
            down) [[ $a =~ ^[1-3]$ ]] && xdotool mousedown "$a" ;;
            up) [[ $a =~ ^[1-3]$ ]] && xdotool mouseup "$a" ;;
            wheel) [[ $a =~ ^[4-7]$ ]] && xdotool click "$a" ;;
            key) [[ $a =~ ^[A-Za-z0-9_+]{1,64}$ ]] && xdotool key --clearmodifiers -- "$a" ;;
            text) [[ $a =~ ^[A-Za-z0-9+/]{1,5464}=*$ ]] && printf '%s' "$a" | base64 -d | xdotool type --clearmodifiers --delay 0 --file - ;;
          esac
        done
        """;

    /// <summary>The guest's frame stream: one base64 PNG of the display per line, about five a second.</summary>
    public const string FrameLoop = """
        if [[ -r /etc/agent-s/env ]]; then . /etc/agent-s/env; fi
        export DISPLAY=":${AGENT_S_DISPLAY:-0}"; unset WAYLAND_DISPLAY
        export XAUTHORITY="$HOME/.local/state/agent-s/Xauthority"
        d=$(mktemp -d); trap 'rm -rf -- "$d"' EXIT
        while :; do
          rm -f -- "$d/f.png"
          if timeout 5s scrot -z "$d/f.png" 2>/dev/null; then base64 -w0 "$d/f.png"; printf '\n'; fi
          sleep 0.15
        done
        """;
}
