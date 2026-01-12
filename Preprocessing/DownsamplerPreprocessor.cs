using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;
using Thesis.Dataset;

namespace Thesis.Preprocessing
{
    // Downsamples the image to an explicit width and height.
    public sealed class DownsamplerPreprocessor : IPreprocessor
    {
        private readonly int _targetWidth;
        private readonly int _targetHeight;

        public DownsamplerPreprocessor(int targetWidth, int targetHeight)
        {
            if (targetWidth <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetWidth));
            if (targetHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetHeight));

            _targetWidth = targetWidth;
            _targetHeight = targetHeight;
        }

        public PreprocessedSample Preprocess(DatasetSample sample)
        {
            if (string.IsNullOrWhiteSpace(sample.ImagePath))
                throw new ArgumentException("Sample has no image path.");

            string dataUrl = ResizeImageFileToJpegDataUrl(
                sample.ImagePath,
                _targetWidth,
                _targetHeight
            );

            return new PreprocessedSample
            {
                Text = sample.Text,
                ImageDataUrl = dataUrl
            };
        }

        private static string ResizeImageFileToJpegDataUrl(
            string imagePath,
            int targetWidth,
            int targetHeight)
        {
            using var input = File.OpenRead(imagePath);
            using var image = Image.Load(input);

            image.Mutate(ctx =>
            {
                ctx.Resize(new ResizeOptions
                {
                    Size = new Size(targetWidth, targetHeight),
                    Mode = ResizeMode.Stretch
                });
            });

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
