using System.Text;
using System.Text.Json;

namespace AgentS.Core;

/// <summary>
/// Plain-language titles for tool steps, ported from the macOS app's ToolActivity: "Open a webpage",
/// "Run a terminal command", "Read a file". Commands are only inspected for display, never run.
/// </summary>
public static class ToolActivity
{
    public static string Title(string tool, string summary, string? detail = null)
    {
        var name = tool.ToLowerInvariant();
        var kind = name.StartsWith("mcp__box__", StringComparison.Ordinal) ? name["mcp__box__".Length..] : name;
        switch (kind)
        {
            case "bash" or "bash-local" or "bash_local" or "shell" or "exec_command" or "command_execution":
                return CommandTitle(detail ?? summary) ?? "Run a terminal command";
            case "read" or "read_file": return "Read a file";
            case "write" or "write_file": return "Write a file";
            case "edit" or "edit_file" or "multiedit" or "apply_patch" or "file_change": return "Edit a file";
            case "list_dir" or "list_files": return "List files";
            case "glob" or "find_files": return "Find files";
            case "grep" or "search_files": return "Search files";
            case "read_image" or "view_image": return "View an image";
            case "desktop": return summary.Length > 0 ? summary : "Use the computer";
        }
        var fallback = ComputerTitle(kind);
        if (ProviderTitle(summary, tool, detail) is { } provided &&
            (fallback is null or "Use the computer" || SameAction(provided, fallback))) return provided;
        return fallback ?? "Use a tool";
    }

    /// <summary>The input field worth showing for a step: the command, file, URL or query, as received.</summary>
    public static string? Detail(JsonElement input)
    {
        if (input.ValueKind != JsonValueKind.Object) return null;
        foreach (var key in new[] { "command", "cmd", "file_path", "filePath", "path", "url", "query", "pattern" })
            if (input.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String && value.GetString() is { } text && text.Trim().Length > 0)
                return text;
        return null;
    }

    private static string? ComputerTitle(string tool) => tool switch
    {
        "browser_navigate" or "browser_open" => "Open a webpage",
        "browser_read" or "browser_fetch" => "Read a webpage",
        "browser_snapshot" => "Inspect the webpage",
        "browser_click" => "Click a webpage element",
        "browser_fill" or "browser_type" => "Type on the webpage",
        "browser_key" or "browser_press_key" => "Press a key in the browser",
        "browser_scroll" => "Scroll the webpage",
        "browser_back" => "Go back in the browser",
        "browser_tabs" => "List browser tabs",
        "screenshot" or "computer_screenshot" => "Capture the computer screen",
        "computer_state" => "Inspect the computer",
        "computer_act" or "desk" or "gui_plan" => "Use the computer",
        "box_status" => "Check the Box status",
        "box_wake" => "Wake the Box",
        "box_hibernate" => "Put the Box to sleep",
        "box_save" => "Save the Box",
        _ => null,
    };

    private static readonly HashSet<string> ProviderVerbs =
    [
        "browse", "capture", "check", "click", "close", "enter", "find", "go", "inspect", "list", "look", "navigate",
        "open", "press", "read", "review", "scroll", "search", "select", "switch", "take", "type", "wait",
    ];

    /// <summary>A provider's own prose ("Check Saturday opening hours"), but not a raw command or path.</summary>
    private static string? ProviderTitle(string summary, string tool, string? detail)
    {
        var title = summary.Trim();
        if (title.Length is 0 or > 120 || title.Equals(tool, StringComparison.OrdinalIgnoreCase)) return null;
        if (detail is not null && detail.Trim().StartsWith(title, StringComparison.Ordinal)) return null;
        if (title.Any(c => c is '\n' or '\r' || "/\\`$;|<>={}[]".Contains(c))) return null;
        var words = title.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length < 2 || !char.IsUpper(words[0][0]) || !ProviderVerbs.Contains(words[0].ToLowerInvariant())) return null;
        if (words.Skip(1).Any(w => w.Trim('.', ',', '!', '?').Contains('.') || w.StartsWith('-'))) return null;
        return title;
    }

    private static bool SameAction(string title, string? fallback) =>
        fallback is not null && fallback.StartsWith(title.Split(' ')[0] + " ", StringComparison.Ordinal);

    // ---- Commands ---------------------------------------------------------------------------------

    private static string? CommandTitle(string source, bool wrapped = false)
    {
        var command = source.Trim();
        if (Tokens(command) is not { } parsed) return null;
        var action = WithoutDirectoryChange(parsed);
        if (!wrapped && action.Count == 3 && action[0].Word is "bash" or "/bin/bash" or "sh" or "/bin/sh" or "zsh" or "/bin/zsh"
            && action[1].Word is "-c" or "-lc" && action[2] is { Word: { } script, Quoted: true })
            return CommandTitle(script, wrapped: true);
        if (IsFileRedirect(action)) return "Write a file";
        if (action.Any(t => t.Word is null)) return BrowserSequenceTitle(action);
        return action.Count == 0 ? null : SimpleCommandTitle(action.Select(t => t.Word!).ToArray());
    }

    private static string? SimpleCommandTitle(string[] words)
    {
        var executable = words[0];
        var verb = words.Length > 1 ? words[1] : null;
        if (executable == "pc" || executable.EndsWith("/pc", StringComparison.Ordinal))
            return verb switch
            {
                null or "tree" => "Inspect the front window",
                "screenshot" => "Capture the screen",
                "apps" => "List open apps",
                "open" => "Open an app",
                "focus" => "Switch to a window",
                "click" => "Click a control",
                "type" => "Type in an app",
                "key" => "Press keys",
                _ => null,
            };
        if ((executable == "simulang" || executable.EndsWith("/simulang", StringComparison.Ordinal)) && verb == "run" && words.Length > 2 && !words[2].StartsWith('-'))
            return "Run a computer-use script";
        if (verb is not null && (executable == "ui" || executable.EndsWith("/ui", StringComparison.Ordinal)))
            return verb switch
            {
                "screenshot" => "Capture the computer screen",
                "tree" => "Inspect the active window",
                "open" => "Open a webpage",
                "launch" when words.Length > 2 && !words[2].StartsWith('-') => "Launch an app",
                "click" => "Click an on-screen element",
                "type" => "Type in an app",
                "key" => "Press keys in an app",
                _ => null,
            };
        if (BrowserVerb(words) is { } browser) return ComputerTitle("browser_" + browser);
        return executable switch
        {
            "cat" or "/bin/cat" => words.Skip(1).Any(w => w.Length > 0 && !w.StartsWith('-')) ? "Read a file" : null,
            "ls" or "/bin/ls" => "List files",
            "pwd" or "/bin/pwd" => "Check the working directory",
            _ => null,
        };
    }

    private static string? BrowserVerb(string[] words) =>
        words.Length >= 2 && (words[0] == "web" || words[0].EndsWith("/web", StringComparison.Ordinal))
        && words[1] is "open" or "snapshot" or "read" or "click" or "type" or "key" or "back" or "scroll" or "tabs" ? words[1] : null;

    /// <summary>"web open x && sleep 2 && web snapshot | head -50" reads as "Open and inspect a webpage".</summary>
    private static string? BrowserSequenceTitle(List<Token> tokens)
    {
        var actions = new List<string>();
        foreach (var command in Split(tokens, t => t.Op is "&&" or ";"))
        {
            if (actions.Count > 0 && command.Count == 2 && command[0].Word == "sleep" && double.TryParse(command[1].Word, out var seconds) && seconds >= 0) continue;
            var pipeline = Split(command, t => t.Op == "|");
            var browser = pipeline[0];
            if (pipeline.Skip(1).Any(stage => !IsOutputFilter(stage))) return null;
            if (browser.Count >= 2 && browser[^1].Word == "/dev/null" && browser[^2].Op is ">" or "2>") browser = browser[..^2];
            if (browser.Any(t => t.Word is null) || BrowserVerb(browser.Select(t => t.Word!).ToArray()) is not { } verb) return null;
            actions.Add(verb);
        }
        if (actions.Count == 0 || ComputerTitle("browser_" + actions[0]) is not { } title) return null;
        if (actions.Count == 1) return title;
        return string.Join(",", actions) switch
        {
            "click,snapshot" => "Click and inspect the webpage",
            "click,read" => "Click and read the webpage",
            "open,snapshot" => "Open and inspect a webpage",
            "open,read" => "Open and read a webpage",
            "click,open,read" => "Click, open, and read a webpage",
            _ => title + " and continue in the browser",
        };
    }

    private static bool IsOutputFilter(List<Token> stage)
    {
        if (stage.Count == 0 || stage.Any(t => t.Word is null)) return false;
        var words = stage.Select(t => t.Word!).ToArray();
        return words[0] switch
        {
            "head" or "tail" => words.Length == 1 || (words.Length == 2 && words[1].StartsWith('-') && int.TryParse(words[1][1..], out _))
                || (words.Length == 3 && words[1] is "-n" or "-c" && int.TryParse(words[2], out _)),
            "grep" => words.Length >= 2 && !words[^1].StartsWith('-'),
            _ => false,
        };
    }

    private static List<Token> WithoutDirectoryChange(List<Token> tokens) =>
        tokens.Count > 3 && tokens[0].Word == "cd" && tokens[1].Word is { Length: > 0 } dir && !dir.StartsWith('-') && tokens[2].Op == "&&"
            ? tokens[3..] : tokens;

    private static bool IsFileRedirect(List<Token> tokens) =>
        tokens.Count >= 3 && tokens[0].Word is "cat" or "/bin/cat" && tokens[^1].Word is { Length: > 0 }
        && tokens[^2].Op is ">" or ">>" && tokens[..^2].All(t => t.Word is not null);

    private static List<List<Token>> Split(List<Token> tokens, Func<Token, bool> separator)
    {
        var parts = new List<List<Token>> { new() };
        foreach (var token in tokens) { if (separator(token)) parts.Add([]); else parts[^1].Add(token); }
        return parts;
    }

    private sealed record Token(string? Word, bool Quoted = false, string? Op = null);

    /// <summary>Words, quotes and the operators && ; | > >> 2>. Anything else ($, backticks, subshells…) gives up.</summary>
    private static List<Token>? Tokens(string command)
    {
        var tokens = new List<Token>();
        var i = 0;
        while (i < command.Length)
        {
            var c = command[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (command[i..].StartsWith("&&", StringComparison.Ordinal)) { tokens.Add(new(null, Op: "&&")); i += 2; continue; }
            if (command[i..].StartsWith(">>", StringComparison.Ordinal)) { tokens.Add(new(null, Op: ">>")); i += 2; continue; }
            if (command[i..].StartsWith("2>", StringComparison.Ordinal)) { tokens.Add(new(null, Op: "2>")); i += 2; continue; }
            if (c is ';' or '|' or '>') { tokens.Add(new(null, Op: c.ToString())); i++; continue; }
            if (c is '$' or '`' or '(' or ')' or '<' or '&' or '\n') return null;
            var word = new StringBuilder();
            var quoted = false;
            while (i < command.Length && !char.IsWhiteSpace(command[i]) && command[i] is not (';' or '|' or '>' or '&'))
            {
                var ch = command[i];
                if (ch is '\'' or '"')
                {
                    var end = command.IndexOf(ch, i + 1);
                    if (end < 0) return null;
                    var inner = command[(i + 1)..end];
                    if (ch == '"' && inner.IndexOfAny(['$', '`', '\\']) >= 0) return null;
                    word.Append(inner); quoted = true; i = end + 1; continue;
                }
                if (ch is '$' or '`' or '(' or ')' or '<' or '\\') return null;
                word.Append(ch); i++;
            }
            tokens.Add(new(word.ToString(), quoted));
        }
        return tokens;
    }
}
