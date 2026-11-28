using AgentS.Core;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>
/// Renders agent replies as Markdown with native WinUI text (see <see cref="Markdown"/>). As on macOS, the reply is
/// one selectable text: paragraphs, headings, lists, quotes and code blocks are paragraphs of a single
/// <see cref="RichTextBlock"/>, so one drag selects across them and Ctrl+C copies plain text. "Copy code" sits below
/// the reply (a menu when there are several blocks). Tables need a grid, so each one splits the text around it.
/// </summary>
internal static class MarkdownView
{
    private static readonly FontFamily Mono = new("Cascadia Mono, Consolas");
    private const double BodySize = 16, BodyLine = 24, BlockGap = 10;

    /// <summary>A panel showing <paramref name="text"/>; call <see cref="Fill"/> to update it as the reply streams.</summary>
    public static StackPanel Create(string text, Func<string, Brush> theme)
    {
        var panel = new StackPanel { Spacing = BlockGap, MaxWidth = 720, HorizontalAlignment = HorizontalAlignment.Left };
        Fill(panel, text, theme);
        return panel;
    }

    public static void Fill(StackPanel panel, string text, Func<string, Brush> theme)
    {
        panel.Children.Clear();
        RichTextBlock? flow = null;
        var codeBlocks = new List<MdCode>();
        foreach (var block in Markdown.Parse(text))
        {
            if (block is MdTable table)
            {
                flow = null;
                panel.Children.Add(Table(table, theme));
                continue;
            }
            if (flow is null) { flow = Flow(); panel.Children.Add(flow); }
            Add(flow.Blocks, block, theme, quoteDepth: 0, codeBlocks);
        }
        if (codeBlocks.Count > 0) panel.Children.Add(CopyCode(codeBlocks, theme));
    }

    /// <summary>The shared selectable text for a run of blocks.</summary>
    private static RichTextBlock Flow() => new()
    {
        TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true, FontSize = BodySize, LineHeight = BodyLine,
    };

    /// <summary>Adds <paramref name="block"/> as one or more paragraphs.</summary>
    private static void Add(BlockCollection blocks, MdBlock block, Func<string, Brush> theme, int quoteDepth, List<MdCode> codeBlocks)
    {
        // Space between blocks, as the stacked views had; none before the first.
        Thickness Gap(double left = 0, double top = BlockGap) => new(left + quoteDepth * 17, blocks.Count == 0 ? 0 : top, 0, 0);
        switch (block)
        {
            case MdHeading heading:
            {
                var paragraph = Paragraph(heading.Inlines, theme);
                paragraph.FontWeight = FontWeights.SemiBold;
                paragraph.FontSize = heading.Level switch { 1 => 22, 2 => 19, 3 => 17, _ => 16 };
                paragraph.LineHeight = paragraph.FontSize * 1.4;
                paragraph.Margin = Gap(top: BlockGap + (heading.Level <= 2 ? 6 : 2));
                Quote(paragraph, quoteDepth, theme);
                blocks.Add(paragraph);
                break;
            }
            case MdListItem item:
            {
                // A hanging indent: the marker sits in the indent and wrapped lines align with the text.
                double markerWidth = item.Marker == "•" ? 22 : 34;
                var paragraph = Paragraph(item.Inlines, theme);
                paragraph.Inlines.Insert(0, new Run { Text = item.Marker + (item.Marker == "•" ? "   " : "  "), Foreground = theme("MutedText") });
                paragraph.Margin = Gap(left: item.Depth * 22 + markerWidth, top: 4);
                paragraph.TextIndent = -markerWidth;
                Quote(paragraph, quoteDepth, theme);
                blocks.Add(paragraph);
                break;
            }
            case MdCode code:
            {
                // Monospace and indented, inside the shared text so it selects with the prose around it.
                var paragraph = new Paragraph
                {
                    FontFamily = Mono, FontSize = 13.5, LineHeight = 20, Margin = Gap(left: 14, top: BlockGap + 2),
                };
                var lines = code.Text.Split('\n');
                for (int i = 0; i < lines.Length; i++)
                {
                    if (i > 0) paragraph.Inlines.Add(new LineBreak());
                    // Keep leading spaces: a run of only spaces still renders its width.
                    if (lines[i].Length > 0) paragraph.Inlines.Add(new Run { Text = lines[i].Replace("\t", "    ") });
                }
                blocks.Add(paragraph);
                codeBlocks.Add(code);
                break;
            }
            case MdQuote quote:
                foreach (var child in quote.Blocks) Add(blocks, child, theme, quoteDepth + 1, codeBlocks);
                break;
            case MdRule:
            {
                var line = new Border { Height = 1, Width = 680, Background = theme("CardBorder") };
                var paragraph = new Paragraph { Margin = Gap(top: BlockGap + 6), LineHeight = 8 };
                paragraph.Inlines.Add(new InlineUIContainer { Child = line });
                blocks.Add(paragraph);
                break;
            }
            case MdParagraph text:
            {
                var paragraph = Paragraph(text.Inlines, theme);
                paragraph.Margin = Gap();
                Quote(paragraph, quoteDepth, theme);
                blocks.Add(paragraph);
                break;
            }
            case MdTable:
                break; // Rendered as its own grid by Fill.
        }
    }

    /// <summary>Quoted text: indented (by the margin) and muted, as the bordered quote was.</summary>
    private static void Quote(Paragraph paragraph, int depth, Func<string, Brush> theme)
    {
        if (depth == 0) return;
        paragraph.Foreground = theme("MutedText");
        paragraph.Inlines.Insert(0, new Run { Text = "▎ ", Foreground = theme("CardBorder") });
    }

    /// <summary>"Copy code" below the reply: one button, or a menu of blocks, as on macOS.</summary>
    private static UIElement CopyCode(IReadOnlyList<MdCode> codeBlocks, Func<string, Brush> theme)
    {
        static void Copy(string text)
        {
            var package = new global::Windows.ApplicationModel.DataTransfer.DataPackage();
            package.SetText(text);
            global::Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
        }
        var icon = new FontIcon { Glyph = "", FontSize = 12 };
        var transparent = new SolidColorBrush(Microsoft.UI.Colors.Transparent);
        Button button;
        if (codeBlocks.Count == 1)
        {
            button = new Button { Content = icon };
            var source = codeBlocks[0].Text;
            button.Click += (_, _) => { Copy(source); ToolTipService.SetToolTip(button, "Copied"); };
            ToolTipService.SetToolTip(button, "Copy code");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(button, "Copy code");
        }
        else
        {
            var menu = new MenuFlyout();
            for (int i = 0; i < codeBlocks.Count; i++)
            {
                var source = codeBlocks[i].Text;
                var language = codeBlocks[i].Language is { Length: > 0 } name ? name : "code";
                var item = new MenuFlyoutItem { Text = $"Copy {language} block {i + 1}" };
                item.Click += (_, _) => Copy(source);
                menu.Items.Add(item);
            }
            button = new Button { Content = icon, Flyout = menu };
            ToolTipService.SetToolTip(button, "Copy a code block");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(button, "Copy a code block");
        }
        button.Padding = new Thickness(6);
        button.BorderThickness = new Thickness(0);
        button.Background = transparent;
        button.Foreground = theme("MutedText");
        button.HorizontalAlignment = HorizontalAlignment.Right;
        return button;
    }

    private static UIElement Table(MdTable table, Func<string, Brush> theme)
    {
        var grid = new Grid { BorderBrush = theme("CardBorder"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(8), HorizontalAlignment = HorizontalAlignment.Left };
        int columns = Math.Max(table.Header.Count, table.Rows.Select(r => r.Count).DefaultIfEmpty(0).Max());
        for (int c = 0; c < columns; c++) grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        void Row(IReadOnlyList<IReadOnlyList<MdInline>> cells, int index, bool header)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            for (int c = 0; c < columns; c++)
            {
                var view = Flow();
                view.Blocks.Add(Paragraph(c < cells.Count ? cells[c] : [], theme));
                if (header) view.FontWeight = FontWeights.SemiBold;
                view.FontSize = 14;
                var cell = new Border
                {
                    Child = view, Padding = new Thickness(10, 6, 10, 6),
                    BorderBrush = theme("CardBorder"), BorderThickness = new Thickness(c == 0 ? 0 : 1, index == 0 ? 0 : 1, 0, 0),
                };
                if (header) cell.Background = theme("ChipBackground");
                Grid.SetRow(cell, index); Grid.SetColumn(cell, c);
                grid.Children.Add(cell);
            }
        }
        Row(table.Header, 0, header: true);
        for (int r = 0; r < table.Rows.Count; r++) Row(table.Rows[r], r + 1, header: false);
        return new ScrollViewer
        {
            Content = grid, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
            HorizontalScrollMode = ScrollMode.Auto, VerticalScrollMode = ScrollMode.Disabled,
        };
    }

    /// <summary>A paragraph of styled inlines (bold, italic, strikethrough, code, links, hard line breaks).</summary>
    private static Paragraph Paragraph(IReadOnlyList<MdInline> inlines, Func<string, Brush> theme)
    {
        var paragraph = new Paragraph();
        foreach (var inline in inlines)
        {
            var parts = inline.Text.Split('\n');
            for (int i = 0; i < parts.Length; i++)
            {
                if (i > 0) paragraph.Inlines.Add(new LineBreak());
                if (parts[i].Length == 0) continue;
                var target = new Run { Text = parts[i] };
                Inline run = target;
                if (inline.Style.HasFlag(MdStyle.Bold)) target.FontWeight = FontWeights.SemiBold;
                if (inline.Style.HasFlag(MdStyle.Italic)) target.FontStyle = global::Windows.UI.Text.FontStyle.Italic;
                if (inline.Style.HasFlag(MdStyle.Strike)) target.TextDecorations = global::Windows.UI.Text.TextDecorations.Strikethrough;
                if (inline.Style.HasFlag(MdStyle.Code))
                {
                    target.FontFamily = Mono; target.FontSize = 14.5;
                    target.Foreground = theme("CodeText");
                }
                if (inline.Link is { } uri)
                {
                    var link = new Hyperlink { NavigateUri = uri };
                    ToolTipService.SetToolTip(link, uri.AbsoluteUri);
                    link.Inlines.Add(run);
                    run = link;
                }
                paragraph.Inlines.Add(run);
            }
        }
        return paragraph;
    }
}
