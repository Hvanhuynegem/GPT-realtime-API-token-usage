using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

using Thesis.Preprocessing;
using Thesis.Dataset;

public sealed class GazeRoiAndThumbnailPreprocessor : IPreprocessor
{
    private readonly float _rho;
    private readonly float _minRoiRelSize;
    private readonly float _gaussianSigma;
    private readonly Size _globalThumbSize;
    private readonly int _jpegQuality;

    public GazeRoiAndThumbnailPreprocessor(
        float rho = 0.5f,
        float minRoiRelSize = 0.2f,
        float gaussianSigma = 15f,
        int globalThumbW = 28,
        int globalThumbH = 28,
        int jpegQuality = 90)
    {
        _rho = Clamp01(rho);
        _minRoiRelSize = Math.Max(0f, minRoiRelSize);
        _gaussianSigma = Math.Max(0.1f, gaussianSigma);
        _globalThumbSize = new Size(globalThumbW, globalThumbH);
        _jpegQuality = Math.Clamp(jpegQuality, 1, 100);
    }

    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        if (string.IsNullOrWhiteSpace(sample.ImagePath))
            throw new ArgumentException("DatasetSample.ImagePath is required.");

        using Image<Rgb24> img = Image.Load<Rgb24>(sample.ImagePath);
        int width = img.Width;
        int height = img.Height;

        // Global thumbnail (always produced)
        using Image<Rgb24> thumb = img.Clone(ctx =>
        {
            ctx.Resize(new ResizeOptions
            {
                Size = _globalThumbSize,
                Mode = ResizeMode.Max
            });
        });

        string globalThumbDataUrl = ImageToJpegDataUrl(thumb, _jpegQuality);

        // If no gaze, return only thumbnail (and optionally original)
        if (sample.Trace == null || sample.Trace.Count == 0)
        {
            return new PreprocessedSample
            {
                Text = sample.Text,
                GlobalThumbnailDataUrl = globalThumbDataUrl,
                ImageDataUrl = ImageFileToDataUrl(sample.ImagePath)
            };
        }

        // Convert gaze trace to pixel coordinates with clamping
        List<(int x, int y)> gazePixels = new(sample.Trace.Count);
        foreach (var (xNormRaw, yNormRaw) in sample.Trace)
        {
            float xNorm = Clamp01(xNormRaw);
            float yNorm = Clamp01(yNormRaw);
            int xPix = (int)MathF.Round(xNorm * (width - 1));
            int yPix = (int)MathF.Round(yNorm * (height - 1));
            gazePixels.Add((xPix, yPix));
        }

        // Build heatmap and blur
        float[] hm = new float[width * height];
        foreach (var (x, y) in gazePixels)
        {
            if ((uint)x < (uint)width && (uint)y < (uint)height)
                hm[y * width + x] += 1f;
        }

        float[] blurred = GaussianBlurSeparable(hm, width, height, _gaussianSigma);

        // Probability heatmap: normalize by sum
        float sum = 0f;
        for (int i = 0; i < blurred.Length; i++) sum += blurred[i];

        float invSum = sum > 1e-12f ? 1f / sum : 1f / (width * height);
        float[] hmProb = new float[blurred.Length];
        for (int i = 0; i < blurred.Length; i++) hmProb[i] = blurred[i] * invSum;

        // Extract ROI from hmProb
        var roi = ExtractRoiFromHeatmap(hmProb, width, height, _rho, _minRoiRelSize);

        // Crop ROI
        using Image<Rgb24> roiImg = img.Clone(ctx => ctx.Crop(roi));
        string roiDataUrl = ImageToJpegDataUrl(roiImg, _jpegQuality);

        return new PreprocessedSample
        {
            Text = sample.Text,
            GlobalThumbnailDataUrl = globalThumbDataUrl,

            // optional
            ImageDataUrl = ImageFileToDataUrl(sample.ImagePath),

            Rois = new List<RoiCrop>
            {
                new RoiCrop
                {
                    Index = 0,
                    Label = null,          // or "gaze"
                    Confidence = 1.0f,     // or rho, or leave 1.0f as "not a detector"
                    X1 = roi.X,
                    Y1 = roi.Y,
                    X2 = roi.X + roi.Width,
                    Y2 = roi.Y + roi.Height,
                    ImageDataUrl = roiDataUrl
                }
            }
        };
    }

    private static Rectangle ExtractRoiFromHeatmap(
        float[] hmProb,
        int w,
        int h,
        float rho,
        float minRelSize)
    {
        int n = w * h;

        // Indices sorted descending by probability mass
        int[] order = Enumerable.Range(0, n)
            .OrderByDescending(i => hmProb[i])
            .ToArray();

        float csum = 0f;
        int k = 0;
        for (; k < n; k++)
        {
            csum += hmProb[order[k]];
            if (csum >= rho) break;
        }

        int xmin = w - 1, xmax = 0, ymin = h - 1, ymax = 0;
        for (int i = 0; i <= k; i++)
        {
            int idx = order[i];
            int y = idx / w;
            int x = idx - y * w;
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
        }

        // Enforce minimum crop size
        int minW = (int)MathF.Floor(minRelSize * w);
        int minH = (int)MathF.Floor(minRelSize * h);

        int curW = xmax - xmin + 1;
        int curH = ymax - ymin + 1;

        float cx = (xmin + xmax) * 0.5f;
        float cy = (ymin + ymax) * 0.5f;

        if (curW < minW)
        {
            float half = (minW - 1) * 0.5f;
            xmin = (int)MathF.Round(cx - half);
            xmax = (int)MathF.Round(cx + half);
        }

        if (curH < minH)
        {
            float half = (minH - 1) * 0.5f;
            ymin = (int)MathF.Round(cy - half);
            ymax = (int)MathF.Round(cy + half);
        }

        // Clip
        xmin = Math.Clamp(xmin, 0, w - 1);
        xmax = Math.Clamp(xmax, 0, w - 1);
        ymin = Math.Clamp(ymin, 0, h - 1);
        ymax = Math.Clamp(ymax, 0, h - 1);

        int rw = Math.Max(1, xmax - xmin + 1);
        int rh = Math.Max(1, ymax - ymin + 1);

        return new Rectangle(xmin, ymin, rw, rh);
    }

    // ----- Gaussian blur on a float heatmap (separable convolution) -----

    private static float[] GaussianBlurSeparable(float[] src, int w, int h, float sigma)
    {
        int radius = Math.Max(1, (int)MathF.Ceiling(3f * sigma));
        float[] kernel = BuildGaussianKernel1D(sigma, radius);

        float[] tmp = new float[w * h];
        float[] dst = new float[w * h];

        // horizontal
        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                float acc = 0f;
                for (int k = -radius; k <= radius; k++)
                {
                    int xx = x + k;
                    if ((uint)xx < (uint)w)
                        acc += src[row + xx] * kernel[k + radius];
                }
                tmp[row + x] = acc;
            }
        }

        // vertical
        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                float acc = 0f;
                for (int k = -radius; k <= radius; k++)
                {
                    int yy = y + k;
                    if ((uint)yy < (uint)h)
                        acc += tmp[yy * w + x] * kernel[k + radius];
                }
                dst[row + x] = acc;
            }
        }

        return dst;
    }

    private static float[] BuildGaussianKernel1D(float sigma, int radius)
    {
        int len = 2 * radius + 1;
        float[] k = new float[len];

        float s2 = sigma * sigma;
        float inv2s2 = 1f / (2f * s2);

        float sum = 0f;
        for (int i = -radius; i <= radius; i++)
        {
            float v = MathF.Exp(-(i * i) * inv2s2);
            k[i + radius] = v;
            sum += v;
        }

        float invSum = 1f / sum;
        for (int i = 0; i < len; i++) k[i] *= invSum;

        return k;
    }

    // ----- Data URL helpers -----

    public static string ImageFileToDataUrl(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        string b64 = Convert.ToBase64String(bytes);
        string mime = GuessMimeFromPath(path);
        return $"data:{mime};base64,{b64}";
    }

    private static string ImageToJpegDataUrl(Image<Rgb24> image, int quality)
    {
        using var ms = new MemoryStream();
        image.Save(ms, new JpegEncoder { Quality = quality });
        string b64 = Convert.ToBase64String(ms.ToArray());
        return $"data:image/jpeg;base64,{b64}";
    }

    private static string GuessMimeFromPath(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        return ext switch
        {
            ".jpg" or ".jpeg" => "image/jpeg",
            ".png" => "image/png",
            ".webp" => "image/webp",
            _ => "application/octet-stream"
        };
    }

    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
}
