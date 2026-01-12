using SixLabors.ImageSharp.Formats.Webp;
using Thesis.Dataset;

namespace Thesis.Preprocessing;

public sealed class WebPPreprocessor : IPreprocessor
{
    private readonly int _quality;
    private readonly bool _lossless;

    public WebPPreprocessor(int quality = 75, bool lossless = false)
    {
        _quality = quality;
        _lossless = lossless;
    }

    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        var encoder = new WebpEncoder
        {
            FileFormat = _lossless ? WebpFileFormatType.Lossless : WebpFileFormatType.Lossy,
            Quality = _quality
        };

        return new PreprocessedSample
        {
            Text = sample.Text,
            ImageDataUrl = ImageEncoding.EncodeFileAsDataUrl(sample.ImagePath ?? "", "image/webp", encoder)
        };
    }
}
