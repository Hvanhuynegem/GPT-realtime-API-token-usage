using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;
using Thesis.Dataset;

namespace Thesis.Preprocessing
{
    public sealed class GrayscalePreprocessor : IPreprocessor
    {
        public PreprocessedSample Preprocess(DatasetSample sample)
        {
            if (string.IsNullOrWhiteSpace(sample.ImagePath))
                throw new ArgumentException("Sample has no image path.");

            string dataUrl = ConvertToGrayscaleJpegDataUrl(sample.ImagePath);

            return new PreprocessedSample
            {
                Text = sample.Text,
                ImageDataUrl = dataUrl
            };
        }

        private static string ConvertToGrayscaleJpegDataUrl(string imagePath)
        {
            using var input = File.OpenRead(imagePath);
            using var image = Image.Load(input);

            // Convert to grayscale
            image.Mutate(ctx => ctx.Grayscale());

            using var ms = new MemoryStream();
            image.Save(ms, new JpegEncoder
            {
                Quality = 85
            });

            string base64 = Convert.ToBase64String(ms.ToArray());
            return $"data:image/jpeg;base64,{base64}";
        }
    }
}
