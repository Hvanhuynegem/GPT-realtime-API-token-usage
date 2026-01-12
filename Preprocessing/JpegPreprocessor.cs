using SixLabors.ImageSharp.Formats.Jpeg;
using Thesis.Dataset;

namespace Thesis.Preprocessing;

public sealed class JpegPreprocessor : IPreprocessor
{
    private readonly int _quality;
    private readonly bool _useOriginalBytes;

    // useOriginalBytes=true means: do not re-encode, just base64 the existing JPG file bytes.
    public JpegPreprocessor(int quality = 85, bool useOriginalBytes = false)
    {
        _quality = quality;
        _useOriginalBytes = useOriginalBytes;
    }

    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        string path = sample.ImagePath ?? "";

        string dataUrl = _useOriginalBytes
            ? ImageEncoding.ReadFileAsDataUrl(path, "image/jpeg")
            : ImageEncoding.EncodeFileAsDataUrl(path, "image/jpeg", new JpegEncoder { Quality = _quality });

        return new PreprocessedSample
        {
            Text = sample.Text,
            ImageDataUrl = dataUrl
        };
    }
}
