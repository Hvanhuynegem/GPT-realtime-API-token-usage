using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Thesis.Preprocessing;

public sealed class YoloOnnxDetector : IDisposable
{
    private readonly InferenceSession _session;

    public int InputSize { get; }
    public float ConfThreshold { get; }
    public float IouThreshold { get; }

    public YoloOnnxDetector(string onnxPath, int inputSize = 640, float confThreshold = 0.25f, float iouThreshold = 0.45f)
    {
        InputSize = inputSize;
        ConfThreshold = confThreshold;
        IouThreshold = iouThreshold;

        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        _session = new InferenceSession(onnxPath, opts);
    }

    public void Dispose() => _session.Dispose();

    public sealed record Det(int ClassId, float Score, float X1, float Y1, float X2, float Y2);

    public List<Det> Detect(Image<Rgb24> original)
    {
        var (inputTensor, scale, padX, padY) = BuildInputTensorLetterbox(original, InputSize);

        var inputName = _session.InputMetadata.Keys.First();

        var input = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
        using var results = _session.Run(new[] { input });

        var output = results.First().AsTensor<float>();
        var dets = DecodeUltralytics(output, ConfThreshold, original.Width, original.Height, scale, padX, padY);
        return Nms(dets, IouThreshold);

    }


    private static (DenseTensor<float> tensor, float scale, float padX, float padY) BuildInputTensorLetterbox(Image<Rgb24> img, int size)
    {
        int w = img.Width;
        int h = img.Height;

        float r = MathF.Min((float)size / w, (float)size / h);
        int newW = (int)MathF.Round(w * r);
        int newH = (int)MathF.Round(h * r);

        float padX = (size - newW) / 2f;
        float padY = (size - newH) / 2f;

        using var resized = img.Clone(ctx => ctx.Resize(newW, newH));
        using var canvas = new Image<Rgb24>(size, size, Color.Black);
        canvas.Mutate(ctx => ctx.DrawImage(resized, new Point((int)padX, (int)padY), 1f));

        // CHW float32 [1,3,size,size] in 0..1
        var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
        canvas.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                {
                    var p = row[x];
                    tensor[0, 0, y, x] = p.R / 255f;
                    tensor[0, 1, y, x] = p.G / 255f;
                    tensor[0, 2, y, x] = p.B / 255f;
                }
            }
        });

        return (tensor, r, padX, padY);
    }

    private static List<Det> DecodeUltralytics(
        Tensor<float> output,
        float confThreshold,
        int origW,
        int origH,
        float scale,
        float padX,
        float padY)
    {
        // Expected: [1, (4 + numClasses), numBoxes]
        int b = output.Dimensions[0];
        int ch = output.Dimensions[1];
        int n = output.Dimensions[2];
        if (b != 1 || ch < 5)
            throw new InvalidOperationException($"Unexpected output shape: [{string.Join("x", output.Dimensions.ToArray())}]");


        int numClasses = ch - 4;

        var dets = new List<Det>(capacity: 256);

        for (int i = 0; i < n; i++)
        {
            float cx = output[0, 0, i];
            float cy = output[0, 1, i];
            float w  = output[0, 2, i];
            float h  = output[0, 3, i];

            // best class score
            int bestCls = -1;
            float bestScore = 0f;

            for (int c = 0; c < numClasses; c++)
            {
                float s = output[0, 4 + c, i];
                if (s > bestScore)
                {
                    bestScore = s;
                    bestCls = c;
                }
            }

            if (bestScore < confThreshold || bestCls < 0)
                continue;

            // Convert from center to corners in letterboxed space
            float x1 = cx - w / 2f;
            float y1 = cy - h / 2f;
            float x2 = cx + w / 2f;
            float y2 = cy + h / 2f;

            // Undo letterbox: remove padding then divide by scale
            x1 = (x1 - padX) / scale;
            y1 = (y1 - padY) / scale;
            x2 = (x2 - padX) / scale;
            y2 = (y2 - padY) / scale;

            // Clamp to original image bounds
            x1 = MathF.Max(0, MathF.Min(origW - 1, x1));
            y1 = MathF.Max(0, MathF.Min(origH - 1, y1));
            x2 = MathF.Max(0, MathF.Min(origW - 1, x2));
            y2 = MathF.Max(0, MathF.Min(origH - 1, y2));

            if (x2 <= x1 || y2 <= y1)
                continue;

            dets.Add(new Det(bestCls, bestScore, x1, y1, x2, y2));
        }

        return dets;
    }

    private static List<Det> Nms(List<Det> dets, float iouThreshold)
    {
        dets.Sort((a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<Det>(dets.Count);

        var suppressed = new bool[dets.Count];

        for (int i = 0; i < dets.Count; i++)
        {
            if (suppressed[i]) continue;

            var a = dets[i];
            kept.Add(a);

            for (int j = i + 1; j < dets.Count; j++)
            {
                if (suppressed[j]) continue;

                var b = dets[j];
                if (IoU(a, b) > iouThreshold)
                    suppressed[j] = true;
            }
        }

        return kept;
    }

    private static float IoU(Det a, Det b)
    {
        float xx1 = MathF.Max(a.X1, b.X1);
        float yy1 = MathF.Max(a.Y1, b.Y1);
        float xx2 = MathF.Min(a.X2, b.X2);
        float yy2 = MathF.Min(a.Y2, b.Y2);

        float w = MathF.Max(0, xx2 - xx1);
        float h = MathF.Max(0, yy2 - yy1);
        float inter = w * h;

        float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
        float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);

        return inter / (areaA + areaB - inter + 1e-6f);
    }
}
