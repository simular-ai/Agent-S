using AgentS.Core;

/// <summary>The Markdown parser behind agent replies in the chat.</summary>
internal static class MarkdownTests
{
    public static void Run(Action<bool, string> check)
    {
        var blocks = Markdown.Parse("""
            # Title
            Some **bold**, *italic*, ***both***, ~~gone~~ and `code`.
            Next line joins.

            ## Steps
            1. First
            2. Second with [a link](https://example.com)
               - nested *item*
            - [x] done
            - [ ] todo

            > Quoted **text**

            ```python
            print("hi")
              indented
            ```

            | Name | Value |
            |------|------:|
            | a | `1` |
            | b \| c | 2 |

            ---
            Visit https://example.com/path. Or <https://example.org>.
            """);
        check(blocks[0] is MdHeading { Level: 1 } h1 && h1.Inlines.Single().Text == "Title", "Headings");
        var paragraph = (MdParagraph)blocks[1];
        check(paragraph.Inlines.Any(i => i is { Text: "bold", Style: MdStyle.Bold }) && paragraph.Inlines.Any(i => i is { Text: "italic", Style: MdStyle.Italic }) &&
              paragraph.Inlines.Any(i => i is { Text: "both", Style: MdStyle.Bold | MdStyle.Italic }) && paragraph.Inlines.Any(i => i is { Text: "gone", Style: MdStyle.Strike }) &&
              paragraph.Inlines.Any(i => i is { Text: "code", Style: MdStyle.Code }), "Bold, italic, both, strikethrough and inline code");
        check(string.Concat(paragraph.Inlines.Select(i => i.Text)).EndsWith("code. Next line joins.", StringComparison.Ordinal), "Soft line breaks join a paragraph");
        var items = blocks.OfType<MdListItem>().ToList();
        check(items.Count == 5 && items[0].Marker == "1." && items[1].Marker == "2." && items[2] is { Depth: 1, Marker: "•" } && items[3].Marker == "☑" && items[4].Marker == "☐",
            "Numbered, nested and task lists");
        check(items[1].Inlines.Any(i => i.Text == "a link" && i.Link == new Uri("https://example.com")), "Links keep their text and target");
        check(blocks.OfType<MdQuote>().Single().Blocks.Single() is MdParagraph quoted && quoted.Inlines.Any(i => i.Style == MdStyle.Bold), "Block quotes hold Markdown");
        check(blocks.OfType<MdCode>().Single() is { Language: "python" } code && code.Text == "print(\"hi\")\n  indented", "Fenced code keeps its text and indentation");
        var table = blocks.OfType<MdTable>().Single();
        check(table.Header.Count == 2 && table.Rows.Count == 2 && table.Rows[0][1].Single().Style == MdStyle.Code && table.Rows[1][0].Single().Text == "b | c", "Tables, with escaped pipes");
        check(blocks.OfType<MdRule>().Count() == 1, "Horizontal rules");
        var last = (MdParagraph)blocks[^1];
        check(last.Inlines.Any(i => i.Link == new Uri("https://example.com/path") && i.Text == "https://example.com/path") &&
              last.Inlines.Any(i => i.Link == new Uri("https://example.org")), "Bare and angle-bracket URLs become links, without trailing punctuation");

        check(Markdown.Inlines("[click](javascript:alert(1))").All(i => i.Link is null) &&
              Markdown.Inlines("[f](file:///C:/x)").All(i => i.Link is null), "Only http(s) and mailto links are clickable");
        check(Markdown.Inlines("snake_case_name and 2 * 3 * 4").All(i => i.Style == MdStyle.None), "Underscores inside words and spaced asterisks stay text");
        check(Markdown.Inlines(@"\*not italic\*").Single() is { Text: "*not italic*", Style: MdStyle.None }, "Backslash escapes");
        check(Markdown.Inlines("**unclosed bold").Single().Text == "**unclosed bold", "Unclosed emphasis stays as typed");
        check(Markdown.Parse("```\nstill streaming") is [MdCode { Text: "still streaming" }], "An unclosed fence (still streaming) shows as code");
        check(Markdown.Parse("line one  \nline two") is [MdParagraph p] && string.Concat(p.Inlines.Select(i => i.Text)) == "line one\nline two", "Hard line breaks");
        check(Markdown.Parse("Title\n===") is [MdHeading { Level: 1 }], "Underlined headings");
        check(Markdown.Plain("## Done\n**Opened** the [page](https://x.y).") == "Done Opened the page.", "Plain text for previews");
        check(Markdown.Parse("") is [] && Markdown.Parse("just text") is [MdParagraph], "Empty and plain replies");
    }
}
