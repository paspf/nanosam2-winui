using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace NanoSAM2App;

public sealed record FloatTensor(float[] Data, int[] Shape);

public sealed record ModelBundle(string Encoder, string Heads, string MemoryEncoder,
    string MemoryAttention, string Constants)
{
    public IEnumerable<string> Files => [Encoder, Heads, MemoryEncoder, MemoryAttention, Constants];

    public static ModelBundle Load(string directory)
    {
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException("Select an existing model folder.");
        var encoders = Directory.GetFiles(directory, "*-image-encoder-sa1-*.onnx");
        var prefixes = encoders.Select(path => Path.GetFileName(path).Split("-image-encoder-sa1-")[0]).Distinct().ToArray();
        if (prefixes.Length != 1)
            throw new InvalidDataException("The folder must contain exactly one model family. Keep different encoders in separate folders.");
        string Find(string block)
        {
            var candidates = Directory.GetFiles(directory, $"{prefixes[0]}-{block}-sa1-*.onnx");
            return candidates.OrderByDescending(path =>
                int.TryParse(Regex.Match(path, @"-op(\d+)").Groups[1].Value, out var opset) ? opset : 0)
                .ThenBy(path => path.Contains("-simply") ? 1 : 0).FirstOrDefault()
                ?? throw new FileNotFoundException($"Missing {block} ONNX model.");
        }
        var constants = Path.Combine(directory, $"{prefixes[0]}-constants.npz");
        if (!File.Exists(constants)) throw new FileNotFoundException("Missing exported constants .npz file.");
        return new(Find("image-encoder"), Find("sam-heads"), Find("memory-encoder"), Find("memory-attention"), constants);
    }

    public static Dictionary<string, FloatTensor> ReadConstants(string path)
    {
        using var archive = ZipFile.OpenRead(path);
        var result = new Dictionary<string, FloatTensor>();
        foreach (var entry in archive.Entries.Where(entry => entry.Name.EndsWith(".npy")))
        {
            using var reader = new BinaryReader(entry.Open());
            if (!reader.ReadBytes(6).SequenceEqual(new byte[] { 147, 78, 85, 77, 80, 89 }))
                throw new InvalidDataException("Invalid NumPy constant header.");
            var major = reader.ReadByte();
            reader.ReadByte();
            var length = major == 1 ? reader.ReadUInt16() : reader.ReadInt32();
            var header = Encoding.ASCII.GetString(reader.ReadBytes(length));
            if (!Regex.IsMatch(header, "'descr':\\s*'[<|]f4'") || !Regex.IsMatch(header, "'fortran_order':\\s*False"))
                throw new InvalidDataException("Constants must be little-endian float32, C-order arrays.");
            var shapeText = Regex.Match(header, "'shape':\\s*\\(([^)]*)\\)").Groups[1].Value;
            var shape = shapeText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
            var count = shape.Aggregate(1, (total, size) => checked(total * size));
            var data = new float[count];
            for (var index = 0; index < count; index++) data[index] = reader.ReadSingle();
            result.Add(Path.GetFileNameWithoutExtension(entry.Name), new(data, shape));
        }
        return result;
    }
}