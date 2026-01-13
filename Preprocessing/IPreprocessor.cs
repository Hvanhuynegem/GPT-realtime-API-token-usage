using Thesis.Dataset;
namespace Thesis.Preprocessing;

public sealed class RoiCrop
{
    public int Index { get; init; }
    public string? Label { get; init; }          // optional
    public float Confidence { get; init; }
    public int X1 { get; init; }
    public int Y1 { get; init; }
    public int X2 { get; init; }
    public int Y2 { get; init; }
    public string ImageDataUrl { get; init; } = "";
}

public class PreprocessedSample
{
    public string Text { get; set; } = "";

    public string? ImageDataUrl { get; set; }

    public List<RoiCrop> Rois { get; set; } = new();

    public string? GlobalThumbnailDataUrl { get; set; }
}


public interface IPreprocessor
{
    PreprocessedSample Preprocess(DatasetSample sample);
}