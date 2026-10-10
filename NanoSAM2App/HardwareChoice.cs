using Microsoft.ML.OnnxRuntime;

namespace NanoSAM2App;

public sealed record HardwareChoice(string Label, OrtEpDevice? Device = null)
{
    public SessionOptions CreateOptions(bool enableDirectMlGraphOptimizations = false)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = Math.Clamp(Environment.ProcessorCount / 2, 1, 4), InterOpNumThreads = 1
        };
        try
        {
            if (Device is not null && Device.EpName != "CPUExecutionProvider")
            {
                options.EnableMemoryPattern = false;
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                if (Device.EpName == "DmlExecutionProvider" && !enableDirectMlGraphOptimizations)
                    options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                options.AppendExecutionProvider(OrtEnv.Instance(), new[] { Device }, new Dictionary<string, string>());
            }
            return options;
        }
        catch { options.Dispose(); throw; }
    }

    public static List<HardwareChoice> Enumerate()
    {
        var choices = new List<HardwareChoice> { new("CPU - ONNX Runtime") };
        var ordinal = 0;
        foreach (var device in OrtEnv.Instance().GetEpDevices())
        {
            if (device.EpName == "CPUExecutionProvider") continue;
            choices.Add(new($"{device.HardwareDevice.Type} - {device.EpName} [{++ordinal}]", device));
        }
        return choices;
    }
}