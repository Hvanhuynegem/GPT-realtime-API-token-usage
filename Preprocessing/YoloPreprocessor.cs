using Thesis.Dataset;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;


namespace Thesis.Preprocessing;

public sealed class YoloPreprocessor : IPreprocessor, IDisposable
{
    private readonly YoloOnnxDetector _detector;
    private readonly int _thumbW;
    private readonly int _thumbH;
    private readonly int _jpegQuality;

    public YoloPreprocessor(
        string onnxRelativePath = @"Preprocessing\Yolov12\yolo12n.onnx",
        int inputSize = 640,
        float confThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int thumbW = 28,
        int thumbH = 28,
        int jpegQuality = 90)
    {
        _detector = new YoloOnnxDetector(onnxRelativePath, inputSize, confThreshold, iouThreshold);
        _thumbW = thumbW;
        _thumbH = thumbH;
        _jpegQuality = jpegQuality;
    }

    public void Dispose() => _detector.Dispose();

    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        var imagePath = sample.ImagePath ?? "";
        using var img = Image.Load<Rgb24>(imagePath);

        // Global thumbnail
        using var thumb = img.Clone(ctx => ctx.Resize(_thumbW, _thumbH));

        // Detect objects
        var dets = _detector.Detect(img);

        var rois = new List<RoiCrop>(dets.Count);
        for (int i = 0; i < dets.Count; i++)
        {
            var d = dets[i];

            int x1 = (int)MathF.Floor(d.X1);
            int y1 = (int)MathF.Floor(d.Y1);
            int x2 = (int)MathF.Ceiling(d.X2);
            int y2 = (int)MathF.Ceiling(d.Y2);

            int cw = Math.Max(1, x2 - x1);
            int ch = Math.Max(1, y2 - y1);

            using var crop = img.Clone(ctx => ctx.Crop(new Rectangle(x1, y1, cw, ch)));
            string cropUrl = ImageEncoding.EncodeImageAsDataUrl(
                crop,
                "image/jpeg",
                new JpegEncoder { Quality = _jpegQuality }
            );


            rois.Add(new RoiCrop
            {
                Index = i,
                Label = $"class_{d.ClassId}",     // map to COCO labels if you want
                Confidence = d.Score,
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                ImageDataUrl = cropUrl
            });
        }

        return new PreprocessedSample
        {
            Text = sample.Text,

            // If you want the original file bytes unchanged:
            // ImageDataUrl = ImageEncoding.ReadFileAsDataUrl(imagePath, "image/jpeg"),

            // If you want to re-encode the full image to JPEG with a quality:
            ImageDataUrl = ImageEncoding.EncodeFileAsDataUrl(
                imagePath,
                "image/jpeg",
                new JpegEncoder { Quality = _jpegQuality }
            ),

            GlobalThumbnailDataUrl = ImageEncoding.EncodeImageAsDataUrl(
                thumb,
                "image/jpeg",
                new JpegEncoder { Quality = _jpegQuality }
            ),

            Rois = rois
        };
    }
}
