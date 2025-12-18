using Thesis.Dataset;

namespace Thesis.Preprocessing;


// Simple identity preprocessor: no text change, just convert image to data URL.
public class IdentityPreprocessor : IPreprocessor
{
    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        return new PreprocessedSample
        {
            Text = sample.Text,
            ImageDataUrl = ImageDataUrl.ImageFileToDataUrl(sample.ImagePath ?? "")
        };
    }
}