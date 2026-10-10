using OpenCvSharp;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;

namespace NanoSAM2App;

public sealed class CameraSource : IAsyncDisposable
{
    private readonly MediaCapture _capture = new();
    private readonly object _sync = new();
    private readonly TaskCompletionSource _firstFrame = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private MediaFrameReader? _reader;
    private byte[]? _pixels;
    private int _width;
    private int _height;
    private DateTime _lastFrame;
    private Exception? _error;

    public async Task StartAsync(string deviceId)
    {
        var initialization = _capture.InitializeAsync(new MediaCaptureInitializationSettings
        {
            VideoDeviceId = deviceId, StreamingCaptureMode = StreamingCaptureMode.Video,
            MemoryPreference = MediaCaptureMemoryPreference.Cpu, SharingMode = MediaCaptureSharingMode.SharedReadOnly
        });
        try { await initialization.AsTask().WaitAsync(TimeSpan.FromSeconds(15)); }
        catch (TimeoutException)
        {
            initialization.Cancel();
            throw new TimeoutException("Camera initialization timed out. Check camera privacy settings, availability, and whether another app is using the device.");
        }
        var source = _capture.FrameSources.Values.FirstOrDefault(source => source.Info.SourceKind == MediaFrameSourceKind.Color)
            ?? throw new InvalidOperationException("The selected camera has no color stream.");
        _reader = await _capture.CreateFrameReaderAsync(source, MediaEncodingSubtypes.Bgra8);
        _reader.AcquisitionMode = MediaFrameReaderAcquisitionMode.Realtime;
        _reader.FrameArrived += FrameArrived;
        var status = await _reader.StartAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(15));
        if (status != MediaFrameReaderStartStatus.Success) throw new InvalidOperationException($"Camera start failed: {status}");
        await _firstFrame.Task.WaitAsync(TimeSpan.FromSeconds(10));
    }

    private void FrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        try
        {
            using var frame = sender.TryAcquireLatestFrame();
            var source = frame?.VideoMediaFrame?.SoftwareBitmap;
            if (source is null) return;
            using (source)
            using (var bitmap = SoftwareBitmap.Convert(source, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore))
            {
                var buffer = new Windows.Storage.Streams.Buffer((uint)(bitmap.PixelWidth * bitmap.PixelHeight * 4));
                bitmap.CopyToBuffer(buffer);
                lock (_sync)
                {
                    _pixels = buffer.ToArray(); _width = bitmap.PixelWidth; _height = bitmap.PixelHeight;
                    _lastFrame = DateTime.UtcNow;
                }
                _firstFrame.TrySetResult();
            }
        }
        catch (Exception error)
        {
            lock (_sync) _error = error;
            _firstFrame.TrySetException(error);
        }
    }

    public bool Read(Mat destination)
    {
        lock (_sync)
        {
            if (_error is not null) throw new InvalidOperationException("Camera frame capture failed.", _error);
            if (_pixels is null) return false;
            if (DateTime.UtcNow - _lastFrame > TimeSpan.FromSeconds(10))
                throw new InvalidOperationException("The camera stopped supplying frames. Check the connection and camera privacy settings.");
            using var bgra = new Mat(_height, _width, MatType.CV_8UC4);
            Marshal.Copy(_pixels, 0, bgra.Data, _pixels.Length);
            Cv2.CvtColor(bgra, destination, ColorConversionCodes.BGRA2BGR);
            return true;
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_reader is not null)
        {
            _reader.FrameArrived -= FrameArrived;
            try { await _reader.StopAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(5)); }
            catch (Exception error) { System.Diagnostics.Debug.WriteLine($"Camera teardown: {error.Message}"); }
            finally { _reader.Dispose(); _reader = null; }
        }
        _capture.Dispose();
    }
}