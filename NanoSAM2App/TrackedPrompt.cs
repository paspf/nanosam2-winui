using Microsoft.UI.Xaml.Media;

namespace NanoSAM2App;

public sealed class TrackedPrompt
{
    public int Id { get; init; }
    public string Name => $"Object {Id}";
    public string DeleteLabel => $"Delete object {Id}";
    public string Detail { get; set; } = "Box prompt";
    public SolidColorBrush Color { get; init; } = new(Microsoft.UI.Colors.SeaGreen);
}