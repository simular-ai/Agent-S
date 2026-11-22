using System.Text;
using System.Text.RegularExpressions;

namespace AgentS.Core;

/// <summary>Inline styles a run of Markdown text can carry (combined, e.g. bold italic).</summary>
[Flags]
public enum MdStyle { None = 0, Bold = 1, Italic = 2, Code = 4, Strike = 8 }

/// <summary>A run of text with its styles, and a link target when it is a link.</summary>
public sealed record MdInline(string Text, MdStyle Style = MdStyle.None, Uri? Link = null);

public abstract record MdBlock;
public sealed record MdParagraph(IReadOnlyList<MdInline> Inlines) : MdBlock;
public sealed record MdHeading(int Level, IReadOnlyList<MdInline> Inlines) : MdBlock;
/// <summary>One list item; nested lists are flattened into items with a larger <see cref="Depth"/>.</summary>
public sealed record MdListItem(int Depth, string Marker, IReadOnlyList<MdInline> Inlines) : MdBlock;
public sealed record MdCode(string Language, string Text) : MdBlock;
public sealed record MdQuote(IReadOnlyList<MdBlock> Blocks) : MdBlock;
public sealed record MdTable(IReadOnlyList<IReadOnlyList<MdInline>> Header, IReadOnlyList<IReadOnlyList<IReadOnlyList<MdInline>>> Rows) : MdBlock;
public sealed record MdRule : MdBlock;

/// <summary>
/// A small CommonMark-style parser for agent replies: headings, paragraphs, lists, code, quotes, tables,
/// rules, emphasis, inline code and links. It never fails: anything it doesn't recognize stays plain text.
/// An unclosed code fence (a reply still streaming in) shows as code up to the end.
/// </summary>
public static partial class Markdown
{
    public static IReadOnlyList<MdBlock> Parse(string text)
    {
        var lines = text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        return Blocks(lines, 0, lines.Length);
    }

    private static List<MdBlock> Blocks(string[] lines, int start, int end)
    {
        var blocks = new List<MdBlock>();
        var paragraph = new List<string>();
        void Flush()
        {
            if (paragraph.Count == 0) return;
            // A line ending in two spaces or a backslash is a hard break; other line breaks join with a space.
            var joined = new StringBuilder();
            for (int i = 0; i < paragraph.Count; i++)
            {
                var line = paragraph[i];
                bool hard = line.EndsWith("  ", StringComparison.Ordinal) || line.EndsWith('\\');
                joined.Append(line.TrimEnd(' ', '\\').Trim());
                if (i < paragraph.Count - 1) joined.Append(hard ? "\n" : " ");
            }
            blocks.Add(new MdParagraph(Inlines(joined.ToString())));
            paragraph.Clear();
        }

        for (int i = start; i < end; i++)
        {
            var line = lines[i];
            var trimmed = line.Trim();
            if (trimmed.Length == 0) { Flush(); continue; }

            // Fenced code: ``` or ~~~, optionally with a language; runs to the closing fence or the end.
            var fence = FenceRegex().Match(line);
            if (fence.Success)
            {
                Flush();
                var marker = fence.Groups[2].Value;
                var language = fence.Groups[3].Value.Trim();
                int indent = fence.Groups[1].Value.Length;
                var code = new List<string>();
                i++;
                for (; i < end; i++)
                {
                    if (lines[i].TrimStart().StartsWith(marker[..3], StringComparison.Ordinal) && lines[i].Trim().Trim(marker[0]).Length == 0) break;
                    code.Add(lines[i].Length >= indent && lines[i][..indent].Trim().Length == 0 ? lines[i][indent..] : lines[i].TrimStart());
                }
                blocks.Add(new MdCode(language, string.Join('\n', code)));
                continue;
            }

            var heading = HeadingRegex().Match(line);
            if (heading.Success)
            {
                Flush();
                blocks.Add(new MdHeading(heading.Groups[1].Value.Length, Inlines(heading.Groups[2].Value.Trim().TrimEnd('#').Trim())));
                continue;
            }

            if (RuleRegex().IsMatch(line) && paragraph.Count == 0) { blocks.Add(new MdRule()); continue; }

            // Setext headings: a paragraph line underlined with === or ---.
            if (paragraph.Count > 0 && SetextRegex().Match(line) is { Success: true } setext)
            {
                var title = string.Join(' ', paragraph.Select(p => p.Trim()));
                paragraph.Clear();
                blocks.Add(new MdHeading(setext.Groups[1].Value[0] == '=' ? 1 : 2, Inlines(title)));
                continue;
            }

            if (trimmed.StartsWith('>'))
            {
                Flush();
                var quoted = new List<string>();
                for (; i < end && lines[i].Trim().StartsWith('>'); i++)
                {
                    var q = lines[i].TrimStart()[1..];
                    quoted.Add(q.StartsWith(' ') ? q[1..] : q);
                }
                i--;
                blocks.Add(new MdQuote(Blocks([.. quoted], 0, quoted.Count)));
                continue;
            }

            var item = ListRegex().Match(line);
            if (item.Success)
            {
                Flush();
                int indent = item.Groups[1].Value.Replace("\t", "    ").Length;
                var marker = item.Groups[2].Value;
                var content = new StringBuilder(item.Groups[3].Value.Trim());
                // Lazy continuation lines (indented, not a new item or block) belong to this item.
                while (i + 1 < end && lines[i + 1].Trim().Length > 0 && !ListRegex().IsMatch(lines[i + 1]) &&
                       !FenceRegex().IsMatch(lines[i + 1]) && !HeadingRegex().IsMatch(lines[i + 1]) && !lines[i + 1].Trim().StartsWith('>') &&
                       (lines[i + 1].StartsWith(' ') || lines[i + 1].StartsWith('\t')))
                {
                    content.Append(' ').Append(lines[++i].Trim());
                }
                var bullet = char.IsDigit(marker[0]) ? marker.TrimEnd(')', '.') + "." : "•";
                var body = content.ToString();
                // Task lists: [ ] and [x].
                if (body.StartsWith("[ ] ", StringComparison.Ordinal)) { bullet = "☐"; body = body[4..]; }
                else if (body.StartsWith("[x] ", StringComparison.OrdinalIgnoreCase)) { bullet = "☑"; body = body[4..]; }
                blocks.Add(new MdListItem(Math.Min(indent / 2, 6), bullet, Inlines(body)));
                continue;
            }

            // Tables: a header row, a | --- | separator, then rows.
            if (trimmed.Contains('|') && i + 1 < end && TableSeparatorRegex().IsMatch(lines[i + 1].Trim()))
            {
                Flush();
                var header = Cells(trimmed).Select(Inlines).ToList();
                var rows = new List<IReadOnlyList<IReadOnlyList<MdInline>>>();
                i += 2;
                for (; i < end && lines[i].Trim().Contains('|'); i++)
                    rows.Add(Cells(lines[i].Trim()).Select(Inlines).ToList());
                i--;
                blocks.Add(new MdTable(header, rows));
                continue;
            }

            paragraph.Add(line);
        }
        Flush();
        return blocks;
    }

    private static List<string> Cells(string row)
    {
        var trimmed = row.Trim();
        if (trimmed.StartsWith('|')) trimmed = trimmed[1..];
        if (trimmed.EndsWith('|') && !trimmed.EndsWith("\\|", StringComparison.Ordinal)) trimmed = trimmed[..^1];
        return PipeRegex().Split(trimmed).Select(c => c.Replace("\\|", "|").Trim()).ToList();
    }

    /// <summary>Inline Markdown: emphasis, strikethrough, code spans, links and bare URLs.</summary>
    public static IReadOnlyList<MdInline> Inlines(string text)
    {
        var result = new List<MdInline>();
        Inlines(text, MdStyle.None, result);
        // Merge neighbours with the same style so renderers make fewer runs.
        var merged = new List<MdInline>();
        foreach (var run in result)
        {
            if (run.Text.Length == 0) continue;
            if (merged.Count > 0 && merged[^1].Style == run.Style && merged[^1].Link == run.Link)
                merged[^1] = merged[^1] with { Text = merged[^1].Text + run.Text };
            else merged.Add(run);
        }
        return merged;
    }

    private static void Inlines(string text, MdStyle style, List<MdInline> output, Uri? link = null)
    {
        var plain = new StringBuilder();
        void Emit() { if (plain.Length > 0) { output.Add(new MdInline(plain.ToString(), style, link)); plain.Clear(); } }
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            // Backslash escapes a punctuation character.
            if (c == '\\' && i + 1 < text.Length && char.IsPunctuation(text[i + 1]) | char.IsSymbol(text[i + 1]))
            { plain.Append(text[i + 1]); i += 2; continue; }

            if (c == '`')
            {
                int ticks = 1; while (i + ticks < text.Length && text[i + ticks] == '`') ticks++;
                var fence = new string('`', ticks);
                int close = text.IndexOf(fence, i + ticks, StringComparison.Ordinal);
                if (close > 0)
                {
                    Emit();
                    var code = text[(i + ticks)..close];
                    if (code.Length > 1 && code[0] == ' ' && code[^1] == ' ') code = code[1..^1];
                    output.Add(new MdInline(code, style | MdStyle.Code, link));
                    i = close + ticks; continue;
                }
            }

            // [text](url) links; only http(s) and mailto targets become clickable.
            if (c == '[' && link is null && LinkRegex().Match(text, i) is { Success: true } found && found.Index == i)
            {
                Emit();
                var target = SafeLink(found.Groups[2].Value);
                Inlines(found.Groups[1].Value, style, output, target);
                if (target is null && found.Groups[2].Value.Length > 0) output.Add(new MdInline($" ({found.Groups[2].Value})", style));
                i += found.Length; continue;
            }

            // <https://…> and bare https:// URLs.
            if (link is null && (c == '<' || c == 'h') && AutoLinkRegex().Match(text, i) is { Success: true } auto && auto.Index == i)
            {
                Emit();
                var url = auto.Groups[1].Success && auto.Groups[1].Length > 0 ? auto.Groups[1].Value : auto.Groups[2].Value;
                // Trailing sentence punctuation isn't part of a bare URL.
                int trim = 0;
                if (!auto.Groups[1].Success || auto.Groups[1].Length == 0)
                    while (url.Length - trim > 0 && ".,;:!?)".Contains(url[url.Length - 1 - trim])) trim++;
                url = url[..^trim];
                output.Add(new MdInline(url, style, SafeLink(url)));
                i += auto.Length - trim; continue;
            }

            // Emphasis: **bold**, __bold__, *italic*, _italic_, ***both***, ~~strike~~.
            if (c is '*' or '_' or '~')
            {
                int run = 1; while (i + run < text.Length && text[i + run] == c) run++;
                bool wordInside = c == '_' && i > 0 && char.IsLetterOrDigit(text[i - 1]);
                if (!wordInside && (c != '~' || run == 2) && i + run < text.Length && !char.IsWhiteSpace(text[i + run]))
                {
                    int use = Math.Min(run, 3);
                    var marker = new string(c, use);
                    int close = FindClose(text, i + use, marker, c);
                    if (close > i + use)
                    {
                        Emit();
                        var inner = text[(i + use)..close];
                        var added = c == '~' ? MdStyle.Strike : use switch { 1 => MdStyle.Italic, 2 => MdStyle.Bold, _ => MdStyle.Bold | MdStyle.Italic };
                        Inlines(inner, style | added, output, link);
                        i = close + use; continue;
                    }
                }
                plain.Append(text, i, run); i += run; continue;
            }

            plain.Append(c); i++;
        }
        Emit();
    }

    /// <summary>The closing delimiter: not preceded by whitespace, and for _ not followed by a word character.</summary>
    private static int FindClose(string text, int from, string marker, char c)
    {
        for (int at = text.IndexOf(marker, from, StringComparison.Ordinal); at >= 0; at = text.IndexOf(marker, at + 1, StringComparison.Ordinal))
        {
            if (char.IsWhiteSpace(text[at - 1])) continue;
            int after = at + marker.Length;
            // A longer run here (*** where * was wanted) closes an inner emphasis instead.
            if (after < text.Length && text[after] == c && marker.Length < 3) { at = after - 1; continue; }
            if (c == '_' && after < text.Length && char.IsLetterOrDigit(text[after])) continue;
            // Skip over code spans, which emphasis can't close inside.
            if (text[from..at].Count(ch => ch == '`') % 2 == 1) continue;
            return at;
        }
        return -1;
    }

    public static Uri? SafeLink(string? value) =>
        Uri.TryCreate(value?.Trim(), UriKind.Absolute, out var uri) && uri.Scheme is "https" or "http" or "mailto" && string.IsNullOrEmpty(uri.UserInfo) ? uri : null;

    /// <summary>The text without Markdown, for previews (the sidebar and the context panel).</summary>
    public static string Plain(string text) => string.Join(' ', Parse(text).Select(PlainBlock)).Trim();

    private static string PlainBlock(MdBlock block) => block switch
    {
        MdParagraph p => Join(p.Inlines),
        MdHeading h => Join(h.Inlines),
        MdListItem l => $"{l.Marker} {Join(l.Inlines)}",
        MdCode c => c.Text,
        MdQuote q => string.Join(' ', q.Blocks.Select(PlainBlock)),
        MdTable t => string.Join(' ', t.Header.Select(Join)),
        _ => "",
    };
    private static string Join(IReadOnlyList<MdInline> inlines) => string.Concat(inlines.Select(i => i.Text));

    [GeneratedRegex(@"^( {0,3})(```+|~~~+)(.*)$")] private static partial Regex FenceRegex();
    [GeneratedRegex(@"^ {0,3}(#{1,6})\s+(.*)$")] private static partial Regex HeadingRegex();
    [GeneratedRegex(@"^ {0,3}([-*_])( *\1){2,} *$")] private static partial Regex RuleRegex();
    [GeneratedRegex(@"^ {0,3}(=+|-+) *$")] private static partial Regex SetextRegex();
    [GeneratedRegex(@"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")] private static partial Regex ListRegex();
    [GeneratedRegex(@"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")] private static partial Regex TableSeparatorRegex();
    [GeneratedRegex(@"(?<!\\)\|")] private static partial Regex PipeRegex();
    [GeneratedRegex(@"\[([^\]]*)\]\(\s*<?([^)\s>]*)>?(?:\s+""[^""]*"")?\s*\)")] private static partial Regex LinkRegex();
    [GeneratedRegex(@"<(https?://[^>\s]+)>|(https?://[^\s<>()\[\]]+(?:\([^\s<>()]*\)[^\s<>()\[\]]*)*)")] private static partial Regex AutoLinkRegex();
}
