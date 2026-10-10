using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Input;
using Windows.System;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace NanoSAM2App;

/// <summary>
/// The application window. This hosts a Frame that displays pages. Add your
/// UI and logic to MainPage.xaml / MainPage.xaml.cs instead of here so you
/// can use Page features such as navigation events and the Loaded lifecycle.
/// </summary>
public sealed partial class MainWindow : Window
{
    private InputKeyboardSource? _keyboardSource;

    public MainWindow()
    {
        InitializeComponent();
        ((UIElement)Content).PreviewKeyDown += Window_PreviewKeyDown;

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);

        AppWindow.SetIcon("Assets/AppIcon.ico");
        AppWindow.Resize(new Windows.Graphics.SizeInt32(1280, 860));
        AppWindow.Title = "NanoSAM2";

        // Navigate the root frame to the main page on startup.
        RootFrame.Navigate(typeof(MainPage));
        RootFrame.Loaded += (_, _) =>
        {
            if (_keyboardSource is not null) return;
            var island = Microsoft.UI.Content.ContentIsland.FindAllForCurrentThread()
                .First(candidate => candidate.Environment.AppWindowId.Value == AppWindow.Id.Value);
            _keyboardSource = InputKeyboardSource.GetForIsland(island);
            _keyboardSource.KeyDown += Window_KeyDown;
        };
        Closed += (_, _) =>
        {
            if (_keyboardSource is not null) _keyboardSource.KeyDown -= Window_KeyDown;
        };
    }

    private void Window_KeyDown(InputKeyboardSource sender, KeyEventArgs args)
    {
        if (args.Handled) return;
        if (args.VirtualKey == VirtualKey.Space && RootFrame.Content is MainPage page)
            args.Handled = page.HandlePlaybackSpace(args.KeyStatus.WasKeyDown);
    }

    private void Window_PreviewKeyDown(object sender, KeyRoutedEventArgs args)
    {
        if (args.Key == VirtualKey.Space && RootFrame.Content is MainPage page)
            args.Handled = page.HandlePlaybackSpace(args.KeyStatus.WasKeyDown);
    }
}
