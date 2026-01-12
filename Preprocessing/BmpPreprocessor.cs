using SixLabors.ImageSharp.Formats.Bmp;
using Thesis.Dataset;

namespace Thesis.Preprocessing;

public sealed class BmpPreprocessor : IPreprocessor
{
    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        return new PreprocessedSample
        {
            Text = sample.Text,
            ImageDataUrl = ImageEncoding.EncodeFileAsDataUrl(sample.ImagePath ?? "", "image/bmp", new BmpEncoder())
        };
    }
}
