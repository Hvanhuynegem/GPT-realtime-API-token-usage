using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Thesis.Dataset;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Thesis.Preprocessing;

public sealed class SpectralResidualSalientPreprocessor : IPreprocessor
{
    private readonly int _spectralSize;           // paper uses 64 px (width or height)
    private readonly int _avgFilterSize;          // n=3 in paper
    private readonly float _gaussianSigma;        // sigma=8 in paper (on saliency map)
    private readonly float _roiThresholdMul;      // threshold = mean * 3 in paper
    private readonly float _minRoiRelSize;        // practical guard: ensure ROI not too tiny
    private readonly Size? _globalThumbSize;      // optional thumbnail
    private readonly int _jpegQuality;

    public SpectralResidualSalientPreprocessor(
            int spectralSize = 64,
            int avgFilterSize = 3,
            float gaussianSigma = 8f,
            float roiThresholdMul = 3f,
            float minRoiRelSize = 0f,
            int globalThumbW = 28,
            int globalThumbH = 28,
            int jpegQuality = 90)
    {
        if (!IsPowerOfTwo(spectralSize))
            throw new ArgumentException("spectralSize must be a power of two for FFT.", nameof(spectralSize));
        if (avgFilterSize < 1 || (avgFilterSize % 2) == 0)
            throw new ArgumentException("avgFilterSize must be odd and >= 1.", nameof(avgFilterSize));
        if (gaussianSigma <= 0) throw new ArgumentOutOfRangeException(nameof(gaussianSigma));
        if (roiThresholdMul <= 0) throw new ArgumentOutOfRangeException(nameof(roiThresholdMul));
        if (minRoiRelSize < 0) throw new ArgumentOutOfRangeException(nameof(minRoiRelSize));
        if (jpegQuality is < 1 or > 100) throw new ArgumentOutOfRangeException(nameof(jpegQuality));

        _spectralSize = spectralSize;
        _avgFilterSize = avgFilterSize;
        _gaussianSigma = gaussianSigma;
        _roiThresholdMul = roiThresholdMul;
        _minRoiRelSize = minRoiRelSize;
        _globalThumbSize = new Size(globalThumbW, globalThumbH);
        _jpegQuality = jpegQuality;
    }

    public PreprocessedSample Preprocess(DatasetSample sample)
    {
        var path = sample.ImagePath ?? "";
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return new PreprocessedSample { Text = sample.Text };
        }

        using var src = Image.Load<Rgba32>(path);

        // 1) Compute spectral-residual saliency map on a fixed scale (64x64)
        using var saliencySmall = ComputeSpectralResidualSaliency(src, _spectralSize, _avgFilterSize, _gaussianSigma);

        // 2) Upsample saliency to original size for ROI selection
        using var saliencyFull = saliencySmall.Clone(ctx => ctx.Resize(src.Width, src.Height, KnownResamplers.Triangle));

        // 3) ROI = largest connected region above mean*3 (paper threshold), fallback to max point
        var roi = FindSalientRoi(saliencyFull, _roiThresholdMul, _minRoiRelSize);

        // 4) Encode outputs
        string? imageDataUrl = ImageDataUrl.ImageFileToDataUrl(path);

        string? globalThumbUrl = null;
        if (_globalThumbSize is not null)
        {
            using var thumb = src.Clone(ctx => ctx.Resize(new ResizeOptions
                {
                    Size = _globalThumbSize.Value,
                    Sampler = KnownResamplers.Bicubic,
                    Mode = ResizeMode.Stretch
                }));
            globalThumbUrl = ImageToJpegDataUrl(thumb, _jpegQuality);
        }

        List<RoiCrop> rois = new();

        if (roi.Width > 0 && roi.Height > 0)
        {
            using var roiImg = src.Clone(ctx => ctx.Crop(roi));
            string roiDataUrl = ImageToJpegDataUrl(roiImg, _jpegQuality);

            rois.Add(new RoiCrop
            {
                Index = 0,
                Label = "spectral_residual",
                Confidence = 1.0f,                 // no detector confidence available, so keep 1.0
                X1 = roi.X,
                Y1 = roi.Y,
                X2 = roi.X + roi.Width,            // exclusive
                Y2 = roi.Y + roi.Height,           // exclusive
                ImageDataUrl = roiDataUrl
            });
        }

        return new PreprocessedSample
        {
            Text = sample.Text,
            ImageDataUrl = imageDataUrl,
            GlobalThumbnailDataUrl = globalThumbUrl,
            Rois = rois
        };
    }

    /// <summary>
    /// Implements Eq. (5)-(9): use original phase, spectral residual of log amplitude, inverse FFT, squared magnitude, Gaussian smooth.
    /// </summary>
    private static Image<L8> ComputeSpectralResidualSaliency(
        Image<Rgba32> src,
        int spectralSize,
        int avgFilterSize,
        float gaussianSigma)
    {
        // Downsample to fixed square
        using var small = src.Clone(ctx => ctx.Resize(spectralSize, spectralSize, KnownResamplers.Bicubic));

        // Process each color channel independently and sum (paper mentions per-channel processing)
        float[,] sum = new float[spectralSize, spectralSize];

        AccumulateChannel(sum, small, spectralSize, avgFilterSize, gaussianSigma, c => c.R / 255f);
        AccumulateChannel(sum, small, spectralSize, avgFilterSize, gaussianSigma, c => c.G / 255f);
        AccumulateChannel(sum, small, spectralSize, avgFilterSize, gaussianSigma, c => c.B / 255f);

        // Normalize summed map to 0..255 for an L8 image
        float min = float.MaxValue, max = float.MinValue;
        for (int y = 0; y < spectralSize; y++)
        for (int x = 0; x < spectralSize; x++)
        {
            var v = sum[x, y];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        float denom = (max - min) > 1e-12f ? (max - min) : 1f;

        var outImg = new Image<L8>(spectralSize, spectralSize);
        outImg.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < spectralSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < spectralSize; x++)
                {
                    float nv = (sum[x, y] - min) / denom;
                    byte b = (byte)Math.Clamp((int)MathF.Round(nv * 255f), 0, 255);
                    row[x] = new L8(b);
                }
            }
        });

        return outImg;
    }

    private static void AccumulateChannel(
        float[,] sum,
        Image<Rgba32> small,
        int n,
        int avgFilterSize,
        float gaussianSigma,
        Func<Rgba32, float> getChannel01)
    {
        // Build spatial signal for FFT
        var spatial = new Complex[n, n];
        small.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < n; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < n; x++)
                {
                    float v = getChannel01(row[x]);
                    spatial[x, y] = new Complex(v, 0);
                }
            }
        });

        // Forward FFT: F{I(x)}
        var freq = FFT2D(spatial, inverse: false);

        // A(f) and P(f)
        var logAmp = new float[n, n];
        var phase = new float[n, n];
        const double eps = 1e-12;

        for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            var z = freq[x, y];
            double a = z.Magnitude;
            logAmp[x, y] = (float)Math.Log(a + eps);
            phase[x, y] = (float)Math.Atan2(z.Imaginary, z.Real);
        }

        // A(f) ~= hn * L(f) (local average filter), then R(f) = L - A
        var avg = BoxFilter2D(logAmp, avgFilterSize);
        var residual = new float[n, n];
        for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            residual[x, y] = logAmp[x, y] - avg[x, y];

        // Reconstruct spectrum: exp(R(f) + i*P(f))
        var reconFreq = new Complex[n, n];
        for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            double mag = Math.Exp(residual[x, y]);
            double ph = phase[x, y];
            reconFreq[x, y] = new Complex(mag * Math.Cos(ph), mag * Math.Sin(ph));
        }

        // Inverse FFT to spatial, saliency = |.|^2, then Gaussian smooth
        var reconSpatial = FFT2D(reconFreq, inverse: true);

        var sal = new float[n, n];
        for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            double m = reconSpatial[x, y].Magnitude;
            sal[x, y] = (float)(m * m);
        }

        sal = GaussianBlur2D(sal, gaussianSigma);

        for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            sum[x, y] += sal[x, y];
    }

    private static Rectangle FindSalientRoi(Image<L8> saliency, float thresholdMul, float minRoiRelSize)
    {
        int w = saliency.Width;
        int h = saliency.Height;

        // Read saliency as floats
        float[,] s = new float[w, h];
        double mean = 0;

        saliency.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    float v = row[x].PackedValue; // 0..255
                    s[x, y] = v;
                    mean += v;
                }
            }
        });

        mean /= (w * (double)h);
        float thr = (float)(mean * thresholdMul);

        // Binary mask above threshold
        bool[,] mask = new bool[w, h];
        int countOn = 0;
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            bool on = s[x, y] > thr;
            mask[x, y] = on;
            if (on) countOn++;
        }

        // If nothing passes threshold, fallback to max pixel neighborhood
        if (countOn == 0)
        {
            int mx = 0, my = 0;
            float best = -1;
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (s[x, y] > best) { best = s[x, y]; mx = x; my = y; }
            }

            int side = Math.Max(16, (int)MathF.Round(MathF.Min(w, h) * MathF.Max(minRoiRelSize, 0.2f)));
            int half = side / 2;
            int x0 = Math.Clamp(mx - half, 0, w - 1);
            int y0 = Math.Clamp(my - half, 0, h - 1);
            int x1 = Math.Clamp(mx + half, 0, w - 1);
            int y1 = Math.Clamp(my + half, 0, h - 1);

            return Rectangle.FromLTRB(x0, y0, x1 + 1, y1 + 1);
        }

        // Largest connected component (4-neighborhood)
        bool[,] visited = new bool[w, h];
        Rectangle bestRect = Rectangle.Empty;
        int bestArea = -1;

        var q = new Queue<(int x, int y)>();
        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };

        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (!mask[x, y] || visited[x, y]) continue;

            int minX = x, maxX = x, minY = y, maxY = y;
            int area = 0;

            visited[x, y] = true;
            q.Enqueue((x, y));

            while (q.Count > 0)
            {
                var (cx, cy) = q.Dequeue();
                area++;

                if (cx < minX) minX = cx;
                if (cx > maxX) maxX = cx;
                if (cy < minY) minY = cy;
                if (cy > maxY) maxY = cy;

                for (int k = 0; k < 4; k++)
                {
                    int nx = cx + dx[k];
                    int ny = cy + dy[k];
                    if ((uint)nx >= (uint)w || (uint)ny >= (uint)h) continue;
                    if (!mask[nx, ny] || visited[nx, ny]) continue;
                    visited[nx, ny] = true;
                    q.Enqueue((nx, ny));
                }
            }

            if (area > bestArea)
            {
                bestArea = area;
                bestRect = Rectangle.FromLTRB(minX, minY, maxX + 1, maxY + 1);
            }
        }

        // Enforce a minimum ROI size (relative to full image)
        int minW = (int)MathF.Round(w * minRoiRelSize);
        int minH = (int)MathF.Round(h * minRoiRelSize);

        if (bestRect.Width < minW || bestRect.Height < minH)
        {
            int cx = bestRect.Left + bestRect.Width / 2;
            int cy = bestRect.Top + bestRect.Height / 2;

            int targetW = Math.Max(bestRect.Width, minW);
            int targetH = Math.Max(bestRect.Height, minH);

            int left = Math.Clamp(cx - targetW / 2, 0, w - targetW);
            int top = Math.Clamp(cy - targetH / 2, 0, h - targetH);

            bestRect = new Rectangle(left, top, targetW, targetH);
        }

        return bestRect;
    }

    private static string ImageToJpegDataUrl(Image img, int quality)
    {
        using var ms = new MemoryStream();
        img.Save(ms, new JpegEncoder { Quality = quality });
        var b64 = Convert.ToBase64String(ms.ToArray());
        return "data:image/jpeg;base64," + b64;
    }

    private static float[,] BoxFilter2D(float[,] src, int k)
    {
        int w = src.GetLength(0);
        int h = src.GetLength(1);
        int r = k / 2;

        float[,] dst = new float[w, h];
        float inv = 1f / (k * k);

        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float sum = 0;
            for (int j = -r; j <= r; j++)
            for (int i = -r; i <= r; i++)
            {
                int xx = Wrap(x + i, w);
                int yy = Wrap(y + j, h);
                sum += src[xx, yy];
            }
            dst[x, y] = sum * inv;
        }

        return dst;
    }

    private static float[,] GaussianBlur2D(float[,] src, float sigma)
    {
        int w = src.GetLength(0);
        int h = src.GetLength(1);

        int radius = Math.Max(1, (int)MathF.Ceiling(3f * sigma));
        int size = 2 * radius + 1;

        float[] kernel = new float[size];
        float sum = 0;
        float s2 = 2f * sigma * sigma;

        for (int i = -radius; i <= radius; i++)
        {
            float v = MathF.Exp(-(i * i) / s2);
            kernel[i + radius] = v;
            sum += v;
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;

        // Horizontal
        float[,] tmp = new float[w, h];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float acc = 0;
            for (int i = -radius; i <= radius; i++)
            {
                int xx = Wrap(x + i, w);
                acc += src[xx, y] * kernel[i + radius];
            }
            tmp[x, y] = acc;
        }

        // Vertical
        float[,] dst = new float[w, h];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float acc = 0;
            for (int i = -radius; i <= radius; i++)
            {
                int yy = Wrap(y + i, h);
                acc += tmp[x, yy] * kernel[i + radius];
            }
            dst[x, y] = acc;
        }

        return dst;
    }

    // -------- FFT (Cooley-Tukey radix-2) --------

    private static Complex[,] FFT2D(Complex[,] input, bool inverse)
    {
        int w = input.GetLength(0);
        int h = input.GetLength(1);
        if (w != h) throw new ArgumentException("FFT2D expects a square array.");
        int n = w;

        // Copy
        var data = new Complex[n, n];
        for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            data[x, y] = input[x, y];

        // Rows
        var row = new Complex[n];
        for (int y = 0; y < n; y++)
        {
            for (int x = 0; x < n; x++) row[x] = data[x, y];
            FFT1D(row, inverse);
            for (int x = 0; x < n; x++) data[x, y] = row[x];
        }

        // Cols
        var col = new Complex[n];
        for (int x = 0; x < n; x++)
        {
            for (int y = 0; y < n; y++) col[y] = data[x, y];
            FFT1D(col, inverse);
            for (int y = 0; y < n; y++) data[x, y] = col[y];
        }

        return data;
    }

    private static void FFT1D(Complex[] buffer, bool inverse)
    {
        int n = buffer.Length;
        if (!IsPowerOfTwo(n)) throw new ArgumentException("FFT length must be a power of two.");

        // Bit reversal
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;

            if (i < j)
            {
                (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
            }
        }

        // Cooley-Tukey
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2.0 * Math.PI / len * (inverse ? 1 : -1);
            Complex wlen = new Complex(Math.Cos(ang), Math.Sin(ang));

            for (int i = 0; i < n; i += len)
            {
                Complex w = Complex.One;
                int half = len >> 1;
                for (int j = 0; j < half; j++)
                {
                    Complex u = buffer[i + j];
                    Complex v = buffer[i + j + half] * w;
                    buffer[i + j] = u + v;
                    buffer[i + j + half] = u - v;
                    w *= wlen;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++) buffer[i] /= n;
        }
    }

    private static int Wrap(int x, int n)
    {
        int r = x % n;
        return r < 0 ? r + n : r;
    }

    private static bool IsPowerOfTwo(int x) => x > 0 && (x & (x - 1)) == 0;
}
