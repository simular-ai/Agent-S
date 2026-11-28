using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using global::Windows.ApplicationModel.DataTransfer;

namespace AgentS.Windows;

public sealed partial class MainWindow
{
    private static StackPanel PromptContent(TextBlock text)
    {
        const int previewLines = 5;
        text.MaxLines = previewLines;
        text.TextTrimming = TextTrimming.CharacterEllipsis;

        var expand = new Button
        {
            Content = "Show more",
            Visibility = Visibility.Collapsed,
        };
        bool expanded = false;
        text.IsTextTrimmedChanged += (_, _) =>
            expand.Visibility = expanded || text.IsTextTrimmed ? Visibility.Visible : Visibility.Collapsed;
        expand.Click += (_, _) =>
        {
            expanded = !expanded;
            text.MaxLines = expanded ? 0 : previewLines;
            expand.Content = expanded ? "Show less" : "Show more";
        };

        var copy = new Button { Content = new FontIcon { Glyph = "\uE8C8", FontSize = 14 } };
        AutomationProperties.SetName(copy, "Copy prompt");
        ToolTipService.SetToolTip(copy, "Copy prompt");
        copy.Click += (_, _) =>
        {
            var package = new DataPackage();
            package.SetText(text.Text);
            Clipboard.SetContent(package);
        };

        var actions = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        actions.Children.Add(expand);
        actions.Children.Add(copy);

        var content = new StackPanel { Spacing = 4 };
        content.Children.Add(text);
        content.Children.Add(actions);
        return content;
    }
}
