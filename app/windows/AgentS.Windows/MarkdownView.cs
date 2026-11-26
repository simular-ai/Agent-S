using AgentS.Core;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using Microsoft.UI.Xaml.Media;

namespace AgentS.Windows;

/// <summary>Renders agent replies as Markdown with native WinUI text (see <see cref="Markdown"/>).</summary>
internal static class MarkdownView
{
    private static readonly FontFamily Mono = new("Cascadia Mono, Consolas");

    /// <summary>A panel showing <paramref name="text"/>; call <see cref="Fill"/> to update it as the reply streams.</summary>
    public static StackPanel Create(string text, Func<string, Brush> theme)
    {
        var panel = new StackPanel { Spacing = 10, MaxWidth = 720, HorizontalAlignment = HorizontalAlignment.Left };
        Fill(panel, text, theme);
        return panel;
    }

    public static void Fill(StackPanel panel, string text, Func<string, Brush> theme)
    {
        panel.Children.Clear();
        foreach (var block in Markdown.Parse(text)) panel.Children.Add(Block(block, theme));
    }

    private static UIElement Block(MdBlock block, Func<string, Brush> theme)
    {
        switch (block)
        {
            case MdHeading heading:
            {
                var view = Rich(heading.Inlines, theme);
                view.FontWeight = FontWeights.SemiBold;
                view.FontSize = heading.Level switch { 1 => 22, 2 => 19, 3 => 17, _ => 16 };
                view.Margin = new Thickness(0, heading.Level <= 2 ? 6 : 2, 0, 0);
                return view;
            }
            case MdListItem item:
            {
                var row = new Grid { ColumnSpacing = 8, Margin = new Thickness(item.Depth * 22, 0, 0, 0) };
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(item.Marker == "•" ? 14 : 26) });
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                var marker = new TextBlock { Text = item.Marker, FontSize = 16, LineHeight = 24, HorizontalAlignment = HorizontalAlignment.Right, Foreground = theme("MutedText") };
                var body = Rich(item.Inlines, theme);
                Grid.SetColumn(body, 1);
                row.Children.Add(marker); row.Children.Add(body);
                return row;
            }
            case MdCode code:
            {
                var view = new TextBlock
                {
                    Text = code.Text, FontFamily = Mono, FontSize = 13.5, LineHeight = 20, IsTextSelectionEnabled = true,
                    TextWrapping = TextWrapping.NoWrap,
                };
                var scroll = new ScrollViewer
                {
                    Content = view, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
                    HorizontalScrollMode = ScrollMode.Auto, VerticalScrollMode = ScrollMode.Disabled,
                };
                var box = new StackPanel { Spacing = 6 };
                // The language, and "Copy code", as on macOS.
                var head = new Grid();
                head.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                head.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                head.Children.Add(new TextBlock { Text = code.Language, FontSize = 11, Foreground = theme("MutedText"), VerticalAlignment = VerticalAlignment.Center });
                var copy = new Button
                {
                    Content = new FontIcon { Glyph = "", FontSize = 12 }, Padding = new Thickness(6), BorderThickness = new Thickness(0),
                    Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.Transparent), Foreground = theme("MutedText"),
                };
                ToolTipService.SetToolTip(copy, "Copy code");
                Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(copy, "Copy code");
                var source = code.Text;
                copy.Click += (_, _) =>
                {
                    var package = new global::Windows.ApplicationModel.DataTransfer.DataPackage();
                    package.SetText(source);
                    global::Windows.ApplicationModel.DataTransfer.Clipboard.SetContent(package);
                    ToolTipService.SetToolTip(copy, "Copied");
                };
                Grid.SetColumn(copy, 1); head.Children.Add(copy);
                box.Children.Add(head);
                box.Children.Add(scroll);
                return new Border
                {
                    Child = box, Background = theme("ChipBackground"), BorderBrush = theme("CardBorder"), BorderThickness = new Thickness(1),
                    CornerRadius = new CornerRadius(10), Padding = new Thickness(14, 10, 14, 12),
                };
            }
            case MdQuote quote:
            {
                var inner = new StackPanel { Spacing = 8 };
                foreach (var child in quote.Blocks) inner.Children.Add(Block(child, theme));
                return new Border
                {
                    Child = inner, BorderBrush = theme("CardBorder"), BorderThickness = new Thickness(3, 0, 0, 0),
                    Padding = new Thickness(14, 2, 0, 2), Opacity = 0.85,
                };
            }
            case MdTable table:
            {
                var grid = new Grid { BorderBrush = theme("CardBorder"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(8), HorizontalAlignment = HorizontalAlignment.Left };
                int columns = Math.Max(table.Header.Count, table.Rows.Select(r => r.Count).DefaultIfEmpty(0).Max());
                for (int c = 0; c < columns; c++) grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                void Row(IReadOnlyList<IReadOnlyList<MdInline>> cells, int index, bool header)
                {
                    grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
                    for (int c = 0; c < columns; c++)
                    {
                        var view = Rich(c < cells.Count ? cells[c] : [], theme);
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
            case MdRule:
                return new Border { Height = 1, Background = theme("CardBorder"), Margin = new Thickness(0, 6, 0, 6) };
            case MdParagraph paragraph:
                return Rich(paragraph.Inlines, theme);
            default:
                return new TextBlock();
        }
    }

    /// <summary>Selectable rich text for a run of inlines.</summary>
    private static RichTextBlock Rich(IReadOnlyList<MdInline> inlines, Func<string, Brush> theme)
    {
        var view = new RichTextBlock { TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true, FontSize = 16, LineHeight = 24 };
        var paragraph = new Paragraph();
        foreach (var inline in inlines)
        {
            // Hard line breaks inside a paragraph.
            var parts = inline.Text.Split('\n');
            for (int i = 0; i < parts.Length; i++)
            {
                if (i > 0) paragraph.Inlines.Add(new LineBreak());
                if (parts[i].Length == 0) continue;
                Inline run = new Run { Text = parts[i] };
                var target = (Run)run;
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
                    Microsoft.UI.Xaml.Controls.ToolTipService.SetToolTip(link, uri.AbsoluteUri);
                    link.Inlines.Add(run);
                    run = link;
                }
                paragraph.Inlines.Add(run);
            }
        }
        view.Blocks.Add(paragraph);
        return view;
    }
}
