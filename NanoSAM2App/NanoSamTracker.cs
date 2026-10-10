using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace NanoSAM2App;

public sealed class NanoSamTracker : IDisposable
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _heads;
    private readonly InferenceSession _memoryEncoder;
    private readonly InferenceSession _attention;
    private readonly Dictionary<string, FloatTensor> _constants;
    private readonly Dictionary<int, ObjectState> _objects = [];
    private FloatTensor[]? _features;
    private int _frameIndex;
    private int _width;
    private int _height;
    public int ImageSize { get; }
    public IReadOnlyList<int> ObjectIds => _objects.Keys.ToArray();

    private sealed record MemoryFrame(int Index, FloatTensor Features, FloatTensor Position, FloatTensor Pointer);
    private sealed class ObjectState
    {
        public required MemoryFrame Conditioning { get; init; }
        public List<MemoryFrame> Recent { get; } = [];
        public required FloatTensor Mask { get; set; }
    }

    public NanoSamTracker(ModelBundle bundle, SessionOptions encoderOptions, SessionOptions trackingOptions)
    {
        var created = new List<InferenceSession>();
        try
        {
            _encoder = new(bundle.Encoder, encoderOptions); created.Add(_encoder);
            _heads = new(bundle.Heads, trackingOptions); created.Add(_heads);
            _memoryEncoder = new(bundle.MemoryEncoder, trackingOptions); created.Add(_memoryEncoder);
            _attention = new(bundle.MemoryAttention, trackingOptions); created.Add(_attention);
            _constants = ModelBundle.ReadConstants(bundle.Constants);
            foreach (var key in new[] { "no_mem_embed", "maskmem_tpos_enc", "tpos_proj_w", "tpos_proj_b" })
                if (!_constants.ContainsKey(key)) throw new InvalidDataException($"Missing constant: {key}");
            if (_constants["maskmem_tpos_enc"].Shape[0] != 7)
                throw new InvalidDataException("This predictor supports exports with seven spatial memory slots.");
            var shape = _encoder.InputMetadata["image"].Dimensions;
            ImageSize = shape[2];
            if (ImageSize <= 0 || shape[3] != ImageSize || ImageSize % 16 != 0)
                throw new InvalidDataException("The image encoder must have a fixed, square input size divisible by 16.");
            if (!_attention.InputMetadata.ContainsKey("rope_k_mask"))
                throw new InvalidDataException("Re-export memory attention with the rope_k_mask input before using this app.");
        }
        catch { foreach (var session in created) session.Dispose(); throw; }
    }

    public void SetFrame(Mat frame, bool advance)
    {
        if (advance) _frameIndex++;
        _width = frame.Width; _height = frame.Height;
        using var resized = new Mat();
        Cv2.Resize(frame, resized, new Size(ImageSize, ImageSize));
        var bytes = new byte[ImageSize * ImageSize * 3];
        Marshal.Copy(resized.Data, bytes, 0, bytes.Length);
        var data = new float[bytes.Length];
        float[] means = [0.485f, 0.456f, 0.406f];
        float[] stds = [0.229f, 0.224f, 0.225f];
        var pixels = ImageSize * ImageSize;
        for (var pixel = 0; pixel < pixels; pixel++)
            for (var channel = 0; channel < 3; channel++)
                data[channel * pixels + pixel] = (bytes[pixel * 3 + 2 - channel] / 255f - means[channel]) / stds[channel];
        _features = Run(_encoder, ("image", new FloatTensor(data, [1, 3, ImageSize, ImageSize])));
    }

    public FloatTensor AddObject(int id, float[] coordinates, int[] labels)
    {
        if (_features is null) throw new InvalidOperationException("No frame is loaded.");
        if (coordinates.Length != labels.Length * 2 || labels.Length == 0)
            throw new ArgumentException("Provide a box or at least one point.");
        var points = new float[coordinates.Length];
        for (var index = 0; index < labels.Length; index++)
        {
            points[index * 2] = coordinates[index * 2] / _width * ImageSize;
            points[index * 2 + 1] = coordinates[index * 2 + 1] / _height * ImageSize;
        }
        var raw = _features[2];
        var conditioned = new FloatTensor((float[])raw.Data.Clone(), raw.Shape);
        var channels = raw.Shape[1];
        var spatial = raw.Shape[2] * raw.Shape[3];
        for (var channel = 0; channel < channels; channel++)
            for (var pixel = 0; pixel < spatial; pixel++)
                conditioned.Data[channel * spatial + pixel] += _constants["no_mem_embed"].Data[channel];
        var outputs = Heads(conditioned, points, labels);
        _objects[id] = new() { Conditioning = EncodeMemory(outputs), Mask = outputs[1] };
        return outputs[1];
    }

    public Dictionary<int, FloatTensor> Track()
    {
        if (_features is null) throw new InvalidOperationException("No frame is loaded.");
        foreach (var state in _objects.Values)
        {
            var conditioned = Attend(state);
            var outputs = Heads(conditioned, [0, 0], [-1]);
            state.Mask = outputs[1];
            state.Recent.Add(EncodeMemory(outputs));
            if (state.Recent.Count > 15) state.Recent.RemoveAt(0);
        }
        return Masks();
    }

    public Dictionary<int, FloatTensor> Masks() => _objects.ToDictionary(pair => pair.Key, pair => pair.Value.Mask);
    public void RemoveObject(int id) => _objects.Remove(id);
    public void Reset() { _objects.Clear(); _frameIndex = 0; }

    private FloatTensor[] Heads(FloatTensor embedding, float[] coordinates, int[] labels)
    {
        using var results = _heads.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("image_embedding", new DenseTensor<float>(embedding.Data, embedding.Shape)),
            NamedOnnxValue.CreateFromTensor("high_res_feat_0", new DenseTensor<float>(_features![0].Data, _features[0].Shape)),
            NamedOnnxValue.CreateFromTensor("high_res_feat_1", new DenseTensor<float>(_features[1].Data, _features[1].Shape)),
            NamedOnnxValue.CreateFromTensor("point_coords", new DenseTensor<float>(coordinates, [1, labels.Length, 2])),
            NamedOnnxValue.CreateFromTensor("point_labels", new DenseTensor<int>(labels, [1, labels.Length]))
        });
        return results.Select(Copy).ToArray();
    }

    private MemoryFrame EncodeMemory(FloatTensor[] outputs)
    {
        var mask = outputs[1];
        var scaled = mask.Data.Select(value => 20f / (1f + MathF.Exp(-value)) - 10f).ToArray();
        var memory = Run(_memoryEncoder, ("pix_feat", _features![2]), ("mask", new FloatTensor(scaled, mask.Shape)));
        return new(_frameIndex, memory[0], memory[1], outputs[3]);
    }

    private FloatTensor Attend(ObjectState state)
    {
        var raw = _features![2];
        var channels = raw.Shape[1];
        var spatial = raw.Shape[2] * raw.Shape[3];
        var memoryDim = state.Conditioning.Features.Shape[1];
        var frames = new[] { state.Conditioning }.Concat(state.Recent.Where(frame =>
            frame.Index < _frameIndex && _frameIndex - frame.Index <= 6)).ToArray();
        var pointers = new[] { state.Conditioning }.Concat(state.Recent.AsEnumerable().Reverse()
            .Where(frame => frame.Index < _frameIndex && _frameIndex - frame.Index < 16)).ToArray();
        var pointerTokens = pointers.Length * channels / memoryDim;
        var length = frames.Length * spatial + pointerTokens;
        var memory = new float[length * memoryDim];
        var positions = new float[memory.Length];
        var cursor = 0;
        foreach (var frame in frames)
        {
            var temporalIndex = frame == state.Conditioning ? 6 : _frameIndex - frame.Index - 1;
            var flattened = ToSequence(frame.Features).Data;
            var position = ToSequence(frame.Position).Data;
            Array.Copy(flattened, 0, memory, cursor * memoryDim, flattened.Length);
            for (var index = 0; index < position.Length; index++)
                positions[cursor * memoryDim + index] = position[index] + _constants["maskmem_tpos_enc"].Data[temporalIndex * memoryDim + index % memoryDim];
            cursor += spatial;
        }
        var maxDifference = Math.Min(_frameIndex + 1, 16) - 1;
        var projection = _constants["tpos_proj_w"].Data;
        var bias = _constants["tpos_proj_b"].Data;
        foreach (var pointer in pointers)
        {
            var peSize = projection.Length > 0 ? channels : memoryDim;
            var sine = new float[peSize];
            for (var index = 0; index < peSize / 2; index++)
            {
                var value = (_frameIndex - pointer.Index) / (float)Math.Max(maxDifference, 1) /
                    MathF.Pow(10000, 2f * (index / 2) / (peSize / 2));
                sine[index] = MathF.Sin(value); sine[index + peSize / 2] = MathF.Cos(value);
            }
            var projected = new float[memoryDim];
            for (var channel = 0; channel < memoryDim; channel++)
            {
                if (projection.Length == 0) projected[channel] = sine[channel];
                else
                {
                    projected[channel] = bias[channel];
                    for (var index = 0; index < peSize; index++)
                        projected[channel] += projection[channel * peSize + index] * sine[index];
                }
            }
            for (var token = 0; token < channels / memoryDim; token++)
            {
                Array.Copy(pointer.Pointer.Data, token * memoryDim, memory, cursor * memoryDim, memoryDim);
                Array.Copy(projected, 0, positions, cursor * memoryDim, memoryDim);
                cursor++;
            }
        }
        var rope = Enumerable.Repeat(1f, length).ToArray();
        Array.Clear(rope, length - pointerTokens, pointerTokens);
        var result = Run(_attention, ("curr", ToSequence(raw)), ("curr_pos", ToSequence(_features[3])),
            ("memory", new FloatTensor(memory, [length, 1, memoryDim])),
            ("memory_pos", new FloatTensor(positions, [length, 1, memoryDim])),
            ("rope_k_mask", new FloatTensor(rope, [length, 1, 1])))[0];
        var data = new float[raw.Data.Length];
        for (var channel = 0; channel < channels; channel++)
            for (var pixel = 0; pixel < spatial; pixel++) data[channel * spatial + pixel] = result.Data[pixel * channels + channel];
        return new(data, raw.Shape);
    }

    public static FloatTensor ToSequence(FloatTensor tensor)
    {
        var channels = tensor.Shape[1];
        var spatial = tensor.Shape[2] * tensor.Shape[3];
        var data = new float[tensor.Data.Length];
        for (var channel = 0; channel < channels; channel++)
            for (var pixel = 0; pixel < spatial; pixel++) data[pixel * channels + channel] = tensor.Data[channel * spatial + pixel];
        return new(data, [spatial, 1, channels]);
    }

    private static FloatTensor[] Run(InferenceSession session, params (string Name, FloatTensor Tensor)[] inputs)
    {
        using var outputs = session.Run(inputs.Select(input => NamedOnnxValue.CreateFromTensor(input.Name,
            new DenseTensor<float>(input.Tensor.Data, input.Tensor.Shape))).ToArray());
        return outputs.Select(Copy).ToArray();
    }
    private static FloatTensor Copy(DisposableNamedOnnxValue value)
    {
        var tensor = value.AsTensor<float>();
        return new(tensor.ToArray(), tensor.Dimensions.ToArray());
    }
    public void Dispose() { _encoder.Dispose(); _heads.Dispose(); _memoryEncoder.Dispose(); _attention.Dispose(); }
}