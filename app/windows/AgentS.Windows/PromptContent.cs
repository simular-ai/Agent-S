using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using global::Windows.ApplicationModel.DataTransfer;

namespace AgentS.Windows;

public sealed partial class MainWindow
{
    private static StackPanel PromptRow(TextBlock text, Border bubble)
    {
        const int previewLines = 5;
        text.MaxLines = previewLines;
        text.TextTrimming = TextTrimming.CharacterEllipsis;

        var transparent = new SolidColorBrush(Microsoft.UI.Colors.Transparent);
        var expand = new Button
        {
            Content = "Show more",
            Background = transparent,
            BorderThickness = new Thickness(0),
            Padding = new Thickness(4, 2, 4, 2),
            FontSize = 12,
        };
        var actions = new StackPanel
        {
            Orientation = Orientation.Horizontal, Spacing = 6,
            HorizontalAlignment = HorizontalAlignment.Right, Visibility = Visibility.Collapsed,
        };
        bool expanded = false;
        text.IsTextTrimmedChanged += (_, _) =>
            actions.Visibility = expanded || text.IsTextTrimmed ? Visibility.Visible : Visibility.Collapsed;
        expand.Click += (_, _) =>
        {
            expanded = !expanded;
            text.MaxLines = expanded ? 0 : previewLines;
            expand.Content = expanded ? "Show less" : "Show more";
        };

        var copy = new Button
        {
            Content = new FontIcon { Glyph = "\uE8C8", FontSize = 14 },
            Background = transparent, BorderThickness = new Thickness(0),
            Padding = new Thickness(4, 2, 4, 2), Opacity = 0, IsHitTestVisible = false,
        };
        AutomationProperties.SetName(copy, "Copy prompt");
        ToolTipService.SetToolTip(copy, "Copy the full prompt");
        copy.Click += (_, _) =>
        {
            var package = new DataPackage();
            package.SetText(text.Text);
            Clipboard.SetContent(package);
        };

        actions.Children.Add(copy);
        actions.Children.Add(expand);

        var row = new StackPanel { HorizontalAlignment = HorizontalAlignment.Right, Background = transparent, Spacing = 4 };
        row.Children.Add(actions);
        row.Children.Add(bubble);

        bool hovered = false, copyFocused = false;
        void UpdateCopy()
        {
            copy.Opacity = hovered || copyFocused ? 1 : 0;
            copy.IsHitTestVisible = hovered;
        }
        row.PointerEntered += (_, _) => { hovered = true; UpdateCopy(); };
        row.PointerExited += (_, _) => { hovered = false; UpdateCopy(); };
        copy.GotFocus += (_, _) => { copyFocused = true; UpdateCopy(); };
        copy.LostFocus += (_, _) => { copyFocused = false; UpdateCopy(); };
        return row;
    }
}
