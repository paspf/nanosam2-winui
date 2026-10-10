using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Microsoft.Windows.AI.MachineLearning;
using OpenCvSharp;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Windows.Devices.Enumeration;
using Windows.Media.Capture;
using Windows.Security.Authorization.AppCapabilityAccess;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.System;
using Point = Windows.Foundation.Point;
using Rectangle = Microsoft.UI.Xaml.Shapes.Rectangle;

namespace NanoSAM2App;

public sealed partial class MainPage : Page
{
    private readonly ObservableCollection<TrackedPrompt> _objects = [];
    private readonly DispatcherTimer _timer = new();
    private readonly SemaphoreSlim _operation = new(1);
    private readonly Dictionary<int, (List<float> Coordinates, List<int> Labels)> _prompts = [];
    private NanoSamTracker? _tracker;
    private VideoCapture? _capture;
    private CameraSource? _camera;
    private Mat? _frame;
    private bool _isBusy;
    private bool _busy
    {
        get => _isBusy;
        set
        {
            _isBusy = value;
            ResetButton.IsEnabled = ObjectList.IsEnabled = NewObjectButton.IsEnabled = !value;
        }
    }
    private bool _processingFrame;
    private bool _playing;
    private bool _ended;
    private bool _loaded;
    private int _nextId = 1;
    private string _tool = "box";
    private Point? _dragStart;
    private bool _resumeAfterDrag;
    private Rectangle? _dragBox;
    private static readonly Windows.UI.Color[] Palette =
        [Microsoft.UI.Colors.MediumSeaGreen, Microsoft.UI.Colors.Coral, Microsoft.UI.Colors.DeepSkyBlue,
         Microsoft.UI.Colors.Gold, Microsoft.UI.Colors.Orchid, Microsoft.UI.Colors.Turquoise];

    public MainPage()
    {
        InitializeComponent();
        EncoderHardwareCombo.SelectionChanged += HardwareSelection_Changed;
        TrackingHardwareCombo.SelectionChanged += HardwareSelection_Changed;
        ObjectList.ItemsSource = _objects;
        _timer.Interval = TimeSpan.FromMilliseconds(33);
        _timer.Tick += FrameTick;
        Loaded += PageLoaded;
        Unloaded += async (_, _) => await StopAsync();
        PromptCanvas.SizeChanged += (_, _) => CancelDrawing();
    }

    private async void PageLoaded(object sender, RoutedEventArgs args)
    {
        if (_loaded) return;
        _loaded = true;
        App.MainWindow.Closed += async (_, _) => await StopAsync();
        try
        {
            EncoderHardwareCombo.ItemsSource = HardwareChoice.Enumerate();
            TrackingHardwareCombo.ItemsSource = HardwareChoice.Enumerate();
            EncoderHardwareCombo.SelectedIndex = TrackingHardwareCombo.SelectedIndex = 0;
            await RefreshHardwareAsync(false);
            await RefreshCamerasAsync();
        }
        catch (Exception error) { SetupError.Text = error.Message; }
    }

    private async Task RefreshCamerasAsync()
    {
        var selectedId = (CameraCombo.SelectedItem as DeviceInformation)?.Id;
        RefreshCamerasButton.IsEnabled = CameraCombo.IsEnabled = false;
        CameraStatus.Text = "Checking webcams...";
        try
        {
            var cameras = await DeviceInformation.FindAllAsync(DeviceClass.VideoCapture);
            CameraCombo.ItemsSource = cameras;
            CameraCombo.SelectedItem = cameras.FirstOrDefault(device => device.Id == selectedId) ?? cameras.FirstOrDefault();
            CameraStatus.Text = cameras.Count == 0 ? "No webcams found." : $"{cameras.Count} camera(s) available";
        }
        catch (Exception error) { CameraStatus.Text = $"Could not refresh webcams: {error.Message}"; }
        finally { RefreshCamerasButton.IsEnabled = CameraCombo.IsEnabled = true; }
    }

    private async void RefreshCameras_Click(object sender, RoutedEventArgs args) => await RefreshCamerasAsync();

    private async Task<bool> RequestCameraAccessAsync(string deviceId)
    {
        if (!OperatingSystem.IsWindowsVersionAtLeast(10, 0, 18362))
            return true;
        var capability = AppCapability.Create("webcam");
        var deviceAccess = DeviceAccessInformation.CreateFromId(deviceId);
        var status = capability.CheckAccess();
        if (status == AppCapabilityAccessStatus.Allowed &&
            deviceAccess.CurrentStatus is not (DeviceAccessStatus.DeniedBySystem or DeviceAccessStatus.DeniedByUser))
            return true;
        SetupStatus.Text = "REQUESTING CAMERA ACCESS...";
        status = await capability.RequestAccessAsync();
        if (status == AppCapabilityAccessStatus.Allowed &&
            deviceAccess.CurrentStatus is not (DeviceAccessStatus.DeniedBySystem or DeviceAccessStatus.DeniedByUser))
            return true;
        await ShowCameraAccessDeniedAsync();
        return false;
    }

    private async Task ShowCameraAccessDeniedAsync()
    {
        SetupError.Text = "Camera access was not granted. Enable camera access in Windows Settings, then click Next again.";
        var dialog = new ContentDialog
        {
            XamlRoot = XamlRoot, Title = "Camera access required",
            Content = "Windows has not granted camera access. Check Camera access, app access, and desktop-app access in Camera privacy settings. Your organization may also restrict access.",
            PrimaryButtonText = "Open settings", CloseButtonText = "Cancel"
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary &&
            !await Launcher.LaunchUriAsync(new Uri("ms-settings:privacy-webcam")))
            SetupError.Text = "Could not open Camera settings. Open Windows Settings > Privacy & security > Camera manually.";
    }

    private void HardwareSelection_Changed(object sender, SelectionChangedEventArgs args)
    {
        var encoderUsesDirectMl = (EncoderHardwareCombo.SelectedItem as HardwareChoice)?.Device?.EpName == "DmlExecutionProvider";
        var trackingUsesDirectMl = (TrackingHardwareCombo.SelectedItem as HardwareChoice)?.Device?.EpName == "DmlExecutionProvider";
        DirectMlOptimizationsCheckBox.Visibility = encoderUsesDirectMl || trackingUsesDirectMl
            ? Visibility.Visible : Visibility.Collapsed;
    }

    private async Task RefreshHardwareAsync(bool acquire)
    {
        HardwareStatus.Text = acquire ? "Preparing compatible providers..." : "Checking installed providers...";
        EncoderHardwareCombo.IsEnabled = TrackingHardwareCombo.IsEnabled = false;
        try
        {
            var catalog = ExecutionProviderCatalog.GetDefault();
            if (catalog is not null)
            {
                if (acquire) await catalog.EnsureAndRegisterCertifiedAsync();
                else await catalog.RegisterCertifiedAsync();
            }
            var choices = HardwareChoice.Enumerate();
            var encoder = (EncoderHardwareCombo.SelectedItem as HardwareChoice)?.Label;
            var tracking = (TrackingHardwareCombo.SelectedItem as HardwareChoice)?.Label;
            EncoderHardwareCombo.ItemsSource = choices;
            TrackingHardwareCombo.ItemsSource = choices;
            EncoderHardwareCombo.SelectedItem = choices.FirstOrDefault(choice => choice.Label == encoder) ?? choices[0];
            TrackingHardwareCombo.SelectedItem = choices.FirstOrDefault(choice => choice.Label == tracking) ?? choices[0];
            HardwareStatus.Text = $"{choices.Count} device/provider choices available. Unsupported operators may run on CPU.";
        }
        catch (Exception error) { HardwareStatus.Text = $"Provider discovery: {error.Message}. CPU remains available."; }
        finally { EncoderHardwareCombo.IsEnabled = TrackingHardwareCombo.IsEnabled = true; }
    }

    private async void RefreshHardware_Click(object sender, RoutedEventArgs args) => await RefreshHardwareAsync(false);
    private async void DownloadProviders_Click(object sender, RoutedEventArgs args)
    {
        var dialog = new ContentDialog
        {
            XamlRoot = XamlRoot, Title = "Install hardware providers?",
            Content = "Windows ML may download certified providers compatible with this PC. Their vendor license terms apply. The download may take up to several minutes.",
            PrimaryButtonText = "Download", CloseButtonText = "Cancel"
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary) await RefreshHardwareAsync(true);
    }

    private async void BrowseModelFolder_Click(object sender, RoutedEventArgs args)
    {
        try
        {
            var picker = new FolderPicker(); picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.MainWindow));
            var folder = await picker.PickSingleFolderAsync();
            if (folder is null) return;
            ModelPathBox.Text = folder.Path;
            ValidateBundle();
        }
        catch (Exception error) { SetupError.Text = error.Message; }
    }

    private ModelBundle ValidateBundle()
    {
        var bundle = ModelBundle.Load(ModelPathBox.Text.Trim());
        ModelFilesList.ItemsSource = bundle.Files.Select(Path.GetFileName).ToArray();
        ModelBundleStatus.Text = "Model bundle found";
        return bundle;
    }

    private async void BrowseVideo_Click(object sender, RoutedEventArgs args)
    {
        try
        {
            var picker = new FileOpenPicker();
            foreach (var extension in new[] { ".mp4", ".avi", ".mov", ".mkv", ".wmv" }) picker.FileTypeFilter.Add(extension);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(App.MainWindow));
            var file = await picker.PickSingleFileAsync();
            if (file is not null) VideoPathBox.Text = file.Path;
        }
        catch (Exception error) { SetupError.Text = error.Message; }
    }

    private void SourceMode_Checked(object sender, RoutedEventArgs args)
    {
        if (VideoPickerRow is null) return;
        var camera = CameraModeRadio.IsChecked == true;
        CameraPickerRow.Visibility = camera ? Visibility.Visible : Visibility.Collapsed;
        VideoPickerRow.Visibility = camera ? Visibility.Collapsed : Visibility.Visible;
    }
    private void CameraCombo_SelectionChanged(object sender, SelectionChangedEventArgs args)
    {
        if (CameraStatus is not null && CameraCombo.SelectedItem is DeviceInformation device)
            CameraStatus.Text = device.Name;
    }

    private async void Next_Click(object sender, RoutedEventArgs args)
    {
        if (_busy) return;
        var locked = false;
        var openingCamera = false;
        SetupError.Text = "";
        try
        {
            var bundle = ValidateBundle();
            var encoder = EncoderHardwareCombo.SelectedItem as HardwareChoice ?? throw new InvalidOperationException("Select image encoder hardware.");
            var tracking = TrackingHardwareCombo.SelectedItem as HardwareChoice ?? throw new InvalidOperationException("Select tracking hardware.");
            var enableDirectMlGraphOptimizations = DirectMlOptimizationsCheckBox.IsChecked == true;
            var camera = CameraModeRadio.IsChecked == true;
            var cameraDevice = CameraCombo.SelectedItem as DeviceInformation;
            var source = VideoPathBox.Text;
            if (camera && cameraDevice is null) throw new InvalidOperationException("Select a webcam.");
            if (!camera && !File.Exists(source)) throw new FileNotFoundException("Select an existing video file.");
            _busy = true; SetupScreen.IsHitTestVisible = false;
            await _operation.WaitAsync(); locked = true;
            SetupStatus.Text = "LOADING MODELS...";
            if (camera)
            {
                if (!await RequestCameraAccessAsync(cameraDevice!.Id)) return;
                SetupStatus.Text = "LOADING MODELS...";
                _camera = new CameraSource();
                openingCamera = true;
                await _camera.StartAsync(cameraDevice.Id);
                openingCamera = false;
            }
            await Task.Run(() =>
            {
                using var encoderOptions = encoder.CreateOptions(enableDirectMlGraphOptimizations);
                using var trackingOptions = tracking.CreateOptions(enableDirectMlGraphOptimizations);
                _tracker = new(bundle, encoderOptions, trackingOptions);
                if (!camera)
                {
                    _capture = new VideoCapture(source);
                    if (!_capture.IsOpened()) throw new InvalidOperationException("Cannot open the video source. Check codec and file access.");
                }
                _frame = new Mat();
                if (!ReadFrame() || _frame.Empty()) throw new InvalidOperationException("The source did not provide a frame.");
                _tracker.SetFrame(_frame, false);
            });
            _ended = false;
            LiveSourceName.Text = camera ? ((DeviceInformation)CameraCombo.SelectedItem).Name : Path.GetFileName(source);
            LiveHardwareText.Text = $"Encoder: {encoder.Label}\nTracking: {tracking.Label}";
            var fps = _capture?.Fps ?? 30;
            _timer.Interval = TimeSpan.FromMilliseconds(1000 / (fps > 0 && fps <= 120 ? fps : 30));
            SetupScreen.Visibility = Visibility.Collapsed; LiveScreen.Visibility = Visibility.Visible;
            CameraPreview.Visibility = Visibility.Visible; EmptyPreview.Visibility = Visibility.Collapsed;
            await RenderAsync();
            SetPlaying(camera);
        }
        catch (Exception error)
        {
            _tracker?.Dispose(); _tracker = null; _capture?.Dispose(); _capture = null; _frame?.Dispose(); _frame = null;
            if (_camera is not null) { await _camera.DisposeAsync(); _camera = null; }
            if (openingCamera && error.HResult == unchecked((int)0x80070005))
                await ShowCameraAccessDeniedAsync();
            else SetupError.Text = $"Could not start the session: {error.Message}";
        }
        finally
        {
            _busy = false; SetupScreen.IsHitTestVisible = true; SetupStatus.Text = "READY TO CONFIGURE";
            if (locked) _operation.Release();
        }
    }

    private void SetPlaying(bool play)
    {
        _playing = play;
        if (play) _timer.Start(); else _timer.Stop();
        PlayPauseIcon.Glyph = play ? "\uE769" : "\uE768";
        RuntimeStateText.Text = _ended ? "ENDED" : play ? "TRACKING" : "PAUSED";
    }

    private async void FrameTick(object? sender, object args)
    {
        if (_busy || _processingFrame || !_playing || _tracker is null) return;
        _processingFrame = true; await _operation.WaitAsync();
        try
        {
            var stopwatch = Stopwatch.StartNew();
            var read = await Task.Run(() =>
            {
                if (!ReadFrame() || _frame!.Empty()) return false;
                _tracker.SetFrame(_frame, true); _tracker.Track(); return true;
            });
            if (!read) { _ended = true; SetPlaying(false); return; }
            _prompts.Clear();
            FpsText.Text = (1000 / Math.Max(1, stopwatch.Elapsed.TotalMilliseconds)).ToString("F1");
            await RenderAsync();
        }
        catch (Exception error) { SetPlaying(false); LiveError.Text = error.Message; }
        finally { _processingFrame = false; _operation.Release(); }
    }

    private async Task RenderAsync()
    {
        if (_frame is null || _frame.Empty()) return;
        var colors = _objects.ToDictionary(item => item.Id, item => item.Color.Color);
        var encoded = await Task.Run(() =>
        {
            using var display = _frame.Clone();
            foreach (var pair in _tracker!.Masks())
            {
                if (!colors.TryGetValue(pair.Key, out var color)) continue;
                var mask = pair.Value;
                using var logits = new Mat(mask.Shape[2], mask.Shape[3], MatType.CV_32FC1);
                System.Runtime.InteropServices.Marshal.Copy(mask.Data, 0, logits.Data, mask.Data.Length);
                using var resized = new Mat(); using var binary = new Mat();
                Cv2.Resize(logits, resized, display.Size());
                Cv2.Compare(resized, 0, binary, CmpType.GT);
                using var tint = new Mat(display.Size(), display.Type(), new Scalar(color.B, color.G, color.R));
                using var blended = new Mat();
                Cv2.AddWeighted(display, 0.55, tint, 0.45, 0, blended); blended.CopyTo(display, binary);
                Cv2.FindContours(binary, out OpenCvSharp.Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                Cv2.DrawContours(display, contours, -1, new Scalar(color.B, color.G, color.R), 2);
            }
            Cv2.ImEncode(".png", display, out var bytes); return bytes;
        });
        using var stream = new MemoryStream(encoded);
        var bitmap = new BitmapImage(); await bitmap.SetSourceAsync(stream.AsRandomAccessStream());
        CameraPreview.Source = bitmap;
    }

    private bool ReadFrame()
    {
        using var next = new Mat();
        var read = _camera is not null ? _camera.Read(next) : _capture!.Read(next);
        if (!read || next.Empty()) return false;
        next.CopyTo(_frame!);
        return true;
    }

    internal bool HandlePlaybackSpace(bool isRepeat)
    {
        if (LiveScreen.Visibility != Visibility.Visible || _tracker is null || _ended) return false;
        if (!isRepeat) TogglePlayback();
        return true;
    }

    private void PlayPause_Click(object sender, RoutedEventArgs args) => TogglePlayback();

    private void TogglePlayback()
    {
        if (_tracker is null || _ended) return;
        if (_playing) { SetPlaying(false); return; }
        if (_busy) return;
        CancelDrawing(); SetPlaying(!_playing);
    }
    private async void BackToSetup_Click(object sender, RoutedEventArgs args) => await StopAsync();
    private async Task StopAsync()
    {
        SetPlaying(false); CancelDrawing();
        _busy = true;
        await _operation.WaitAsync();
        try
        {
            _tracker?.Dispose(); _tracker = null; _capture?.Dispose(); _capture = null; _frame?.Dispose(); _frame = null;
            if (_camera is not null) { await _camera.DisposeAsync(); _camera = null; }
            _objects.Clear(); _prompts.Clear(); _nextId = 1;
            CameraPreview.Source = null; FpsText.Text = "0.0"; LiveError.Text = "";
            LiveScreen.Visibility = Visibility.Collapsed; SetupScreen.Visibility = Visibility.Visible;
        }
        finally { _busy = false; _operation.Release(); }
    }

    private async void ResetObjects_Click(object sender, RoutedEventArgs args)
    {
        if (_busy || _tracker is null) return;
        _busy = true; SetPlaying(false); CancelDrawing(); await _operation.WaitAsync();
        try
        {
            if (_tracker is null) return;
            _tracker.Reset(); _objects.Clear(); _prompts.Clear(); _nextId = 1; await RenderAsync();
        }
        catch (Exception error) { LiveError.Text = error.Message; }
        finally { _busy = false; _operation.Release(); }
    }
    private async void RemoveObject_Click(object sender, RoutedEventArgs args)
    {
        if (_busy || _tracker is null || (sender as Button)?.Tag is not TrackedPrompt item) return;
        _busy = true; SetPlaying(false); await _operation.WaitAsync();
        try
        {
            if (_tracker is null) return;
            _tracker.RemoveObject(item.Id); _prompts.Remove(item.Id); _objects.Remove(item); await RenderAsync();
        }
        catch (Exception error) { LiveError.Text = error.Message; }
        finally { _busy = false; _operation.Release(); }
    }

    private async void NewObject_Click(object sender, RoutedEventArgs args)
    {
        if (_busy || _tracker is null) return;
        _busy = true; SetPlaying(false); await _operation.WaitAsync();
        try
        {
            if (_tracker is null) return;
            CancelDrawing(); ObjectList.SelectedItem = null;
            _tool = "positive"; BoxTool.IsChecked = false; PositiveTool.IsChecked = true; NegativeTool.IsChecked = false;
        }
        finally { _busy = false; _operation.Release(); }
    }

    private void PromptTool_Click(object sender, RoutedEventArgs args)
    {
        CancelDrawing();
        _tool = ReferenceEquals(sender, BoxTool) ? "box" : ReferenceEquals(sender, PositiveTool) ? "positive" : "negative";
        BoxTool.IsChecked = _tool == "box"; PositiveTool.IsChecked = _tool == "positive"; NegativeTool.IsChecked = _tool == "negative";
    }

    private Point? ToFrame(Point position)
    {
        if (_frame is null) return null;
        var scale = Math.Min(PromptCanvas.ActualWidth / _frame.Width, PromptCanvas.ActualHeight / _frame.Height);
        if (scale <= 0) return null;
        var left = (PromptCanvas.ActualWidth - _frame.Width * scale) / 2;
        var top = (PromptCanvas.ActualHeight - _frame.Height * scale) / 2;
        var mapped = new Point((position.X - left) / scale, (position.Y - top) / scale);
        return mapped.X < 0 || mapped.Y < 0 || mapped.X > _frame.Width || mapped.Y > _frame.Height ? null : mapped;
    }

    private async void PromptCanvas_PointerPressed(object sender, PointerRoutedEventArgs args)
    {
        if (_busy || _tracker is null) return;
        var point = args.GetCurrentPoint(PromptCanvas);
        if (!point.Properties.IsLeftButtonPressed || ToFrame(point.Position) is not Point framePoint) return;
        args.Handled = true;
        var resumeAfterPrompt = _playing;
        SetPlaying(false); LiveError.Text = "";
        if (_tool != "box")
        {
            var tool = _tool;
            var selected = ObjectList.SelectedItem as TrackedPrompt;
            if (tool == "negative" && selected is null) { LiveError.Text = "Select an object before adding a negative point."; return; }
            int id;
            float[] coordinates;
            int[] labels;
            _busy = true;
            await _operation.WaitAsync();
            try
            {
                if (_tracker is null) return;
                id = selected?.Id ?? _nextId++;
                if (!_prompts.TryGetValue(id, out var prompt))
                {
                    prompt = ([], []);
                    if (selected is not null && _tracker.Masks().TryGetValue(id, out var mask))
                    {
                        var maximum = Array.IndexOf(mask.Data, mask.Data.Max());
                        if (mask.Data[maximum] > 0)
                        {
                            prompt.Coordinates.Add((maximum % mask.Shape[3] + .5f) / mask.Shape[3] * _frame!.Width);
                            prompt.Coordinates.Add((maximum / mask.Shape[3] + .5f) / mask.Shape[2] * _frame!.Height);
                            prompt.Labels.Add(1);
                        }
                        else if (tool == "negative") { LiveError.Text = "The object has no visible mask. Add a positive point first."; return; }
                    }
                }
                prompt.Coordinates.Add((float)framePoint.X); prompt.Coordinates.Add((float)framePoint.Y);
                prompt.Labels.Add(tool == "negative" ? 0 : 1); _prompts[id] = prompt;
                coordinates = prompt.Coordinates.ToArray(); labels = prompt.Labels.ToArray();
            }
            finally { _busy = false; _operation.Release(); }
            await SubmitPromptAsync(id, coordinates, labels, "Point prompt", resumeAfterPrompt);
            return;
        }
        _dragStart = point.Position;
        _resumeAfterDrag = resumeAfterPrompt;
        _dragBox = new Rectangle { Stroke = new SolidColorBrush(Microsoft.UI.Colors.White), StrokeThickness = 2,
            Fill = new SolidColorBrush(Windows.UI.Color.FromArgb(35, 255, 255, 255)) };
        PromptCanvas.Children.Add(_dragBox); PromptCanvas.CapturePointer(args.Pointer);
        Canvas.SetLeft(_dragBox, point.Position.X); Canvas.SetTop(_dragBox, point.Position.Y);
    }

    private void PromptCanvas_PointerMoved(object sender, PointerRoutedEventArgs args)
    {
        if (_dragStart is not Point start || _dragBox is null) return;
        var position = args.GetCurrentPoint(PromptCanvas).Position;
        Canvas.SetLeft(_dragBox, Math.Min(start.X, position.X)); Canvas.SetTop(_dragBox, Math.Min(start.Y, position.Y));
        _dragBox.Width = Math.Abs(position.X - start.X); _dragBox.Height = Math.Abs(position.Y - start.Y);
    }
    private async void PromptCanvas_PointerReleased(object sender, PointerRoutedEventArgs args)
    {
        if (_dragStart is not Point start) return;
        var resumeAfterPrompt = _resumeAfterDrag;
        var first = ToFrame(start); var last = ToFrame(args.GetCurrentPoint(PromptCanvas).Position); CancelDrawing();
        if (first is not Point begin || last is not Point end || Math.Abs(begin.X - end.X) < 4 || Math.Abs(begin.Y - end.Y) < 4) return;
        var id = _nextId++;
        float[] coordinates = [(float)Math.Min(begin.X, end.X), (float)Math.Min(begin.Y, end.Y),
            (float)Math.Max(begin.X, end.X), (float)Math.Max(begin.Y, end.Y)];
        _prompts[id] = (coordinates.ToList(), new List<int> { 2, 3 });
        await SubmitPromptAsync(id, coordinates, [2, 3], "Box prompt", resumeAfterPrompt);
    }
    private void PromptCanvas_PointerCanceled(object sender, PointerRoutedEventArgs args) => CancelDrawing();
    private void CancelDrawing() { _dragStart = null; _resumeAfterDrag = false; _dragBox = null; PromptCanvas.Children.Clear(); PromptCanvas.ReleasePointerCaptures(); }

    private async Task SubmitPromptAsync(int id, float[] coordinates, int[] labels, string detail, bool resumeAfterPrompt)
    {
        if (_tracker is null || _busy) return;
        var succeeded = false;
        _busy = true; await _operation.WaitAsync();
        try
        {
            if (_tracker is null) return;
            _prompts[id] = (coordinates.ToList(), labels.ToList());
            RuntimeStateText.Text = "SEGMENTING";
            var stopwatch = Stopwatch.StartNew();
            await Task.Run(() => _tracker.AddObject(id, coordinates, labels));
            var item = _objects.FirstOrDefault(item => item.Id == id);
            if (item is null)
            {
                item = new TrackedPrompt { Id = id, Detail = detail, Color = new SolidColorBrush(Palette[(id - 1) % Palette.Length]) };
                _objects.Add(item);
            }
            ObjectList.SelectedItem = item;
            FpsText.Text = (1000 / Math.Max(1, stopwatch.Elapsed.TotalMilliseconds)).ToString("F1");
            await RenderAsync();
            ShowPointMarkers();
            succeeded = true;
        }
        catch (Exception error) { LiveError.Text = error.Message; }
        finally
        {
            _busy = false; _operation.Release();
            var resume = succeeded && resumeAfterPrompt && !_ended && _tracker is not null;
            if (resume) CancelDrawing();
            SetPlaying(resume);
        }
    }

    private void ShowPointMarkers()
    {
        CancelDrawing();
        if (_frame is null) return;
        var scale = Math.Min(PromptCanvas.ActualWidth / _frame.Width, PromptCanvas.ActualHeight / _frame.Height);
        var left = (PromptCanvas.ActualWidth - _frame.Width * scale) / 2;
        var top = (PromptCanvas.ActualHeight - _frame.Height * scale) / 2;
        foreach (var prompt in _prompts.Values)
            for (var index = 0; index < prompt.Labels.Count; index++)
            {
                if (prompt.Labels[index] is not (0 or 1)) continue;
                var marker = new Microsoft.UI.Xaml.Shapes.Ellipse
                {
                    Width = 10, Height = 10, StrokeThickness = 2, IsHitTestVisible = false,
                    Stroke = new SolidColorBrush(Microsoft.UI.Colors.White),
                    Fill = new SolidColorBrush(prompt.Labels[index] == 1 ? Microsoft.UI.Colors.SeaGreen : Microsoft.UI.Colors.IndianRed)
                };
                Canvas.SetLeft(marker, left + prompt.Coordinates[index * 2] * scale - 5);
                Canvas.SetTop(marker, top + prompt.Coordinates[index * 2 + 1] * scale - 5);
                PromptCanvas.Children.Add(marker);
            }
    }
}
