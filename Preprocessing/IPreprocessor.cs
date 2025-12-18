using Thesis.Dataset;
namespace Thesis.Preprocessing;

public class PreprocessedSample
{
    public string Text { get; set; } = "";
    public string? ImageDataUrl { get; set; }
}

public interface IPreprocessor
{
    PreprocessedSample Preprocess(DatasetSample sample);
}