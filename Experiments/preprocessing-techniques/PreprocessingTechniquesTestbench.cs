using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.Metrics;
using SixLabors.ImageSharp;

class PreprocessingTechniquesTestbench
{
    public static void Run(
        int maxSamples = 1000,
        int warmupSamples = 10,
        string datasetDir = "data",
        string outDir = "outputs/preprocessing-techniques")
    {
        var techniques = new List<(string Name, IPreprocessor Preprocessor)>
        {
            ("Baseline_NoOp", new IdentityPreprocessor()),
            ("Downsampler", new DownsamplerPreprocessor(targetWidth: 100, targetHeight: 100)),
            ("Grayscale", new GrayscalePreprocessor()),

            // Format conversion preprocessors
            ("Jpeg_Q85", new JpegPreprocessor(quality: 85, useOriginalBytes: false)),
            ("WebP_Lossy_Q75", new WebPPreprocessor(quality: 75, lossless: false)),
            ("Bmp", new BmpPreprocessor()),

            // NEW: Gaze ROI + global thumbnail
            ("GazeRoi+GlobalThumb",
                new GazeRoiAndThumbnailPreprocessor(
                    rho: 0.5f,
                    minRoiRelSize: 0.2f,
                    gaussianSigma: 15f,
                    globalThumbW: 28,
                    globalThumbH: 28,
                    jpegQuality: 90)),

            // Saliency-based ROI + global thumbnail
            ("SalientRoi+GlobalThumb",
                new SpectralResidualSalientPreprocessor(
                    avgFilterSize: 3,
                    gaussianSigma: 8f,
                    roiThresholdMul: 3f,
                    minRoiRelSize: 0.2f,
                    globalThumbW: 28,
                    globalThumbH: 28,
                    jpegQuality: 90)),
        };

        Directory.CreateDirectory(outDir);

        var samples = VoilaDatasetLoader
            .LoadSamplesFromVoila(datasetDir, maxSamples: maxSamples)
            .ToList();

        if (samples.Count == 0)
            throw new InvalidOperationException("No samples loaded. Check datasetDir and prepared.jsonl.");

        DatasetSample vizSample = samples.First();
        SaveVisualizationImages(outDir, vizSample, techniques);

        var warmup = samples.Take(Math.Min(warmupSamples, samples.Count)).ToList();

        var timeRecords = new List<TimeRecord>();
        var sizeRecords = new List<SizeRecord>();

        foreach (var (name, preprocessor) in techniques)
        {
            foreach (var s in warmup)
                _ = preprocessor.Preprocess(s);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            foreach (var s in samples)
            {
                long originalBinaryBytes = GetOriginalBinaryBytes(s);
                var (origW, origH) = GetOriginalPixelDims(s);

                var sw = Stopwatch.StartNew();
                var processed = preprocessor.Preprocess(s);
                sw.Stop();

                timeRecords.Add(new TimeRecord
                {
                    Technique = name,
                    ImageId = SafeImageId(s),
                    TimeMs = sw.Elapsed.TotalMilliseconds
                });

                // Choose the "effective payload" data URLs for size accounting:
                // - If ROI exists: ROI + GlobalThumb
                // - Else if GlobalThumb exists: GlobalThumb only
                // - Else: fallback to ImageDataUrl (single-image techniques)
                var payloadImages = GetPayloadImageDataUrls(processed);

                if (payloadImages.Count > 0 && originalBinaryBytes > 0 && origW > 0 && origH > 0)
                {
                    var composite = GetCompositeDataUrlStats(payloadImages);

                    // Primary dims (for backwards compatibility with existing CSV columns)
                    // Use ROI if present, else GlobalThumb if present, else ImageDataUrl.
                    string primaryUrl = GetPrimaryDataUrl(processed);
                    var (procW, procH) = !string.IsNullOrWhiteSpace(primaryUrl)
                        ? GetPixelDimsFromDataUrl(primaryUrl)
                        : (0, 0);

                    // Total pixels across all payload images (ROI + thumbnail, or thumbnail only, etc.)
                    long totalPixels = 0;
                    foreach (var url in payloadImages)
                    {
                        var (w, h) = GetPixelDimsFromDataUrl(url);
                        if (w > 0 && h > 0)
                            totalPixels += (long)w * h;
                    }

                    double ratio = (double)composite.BinaryBytesTotal / (double)originalBinaryBytes;

                    int jsonUtf8Bytes = BuildRealtimeJsonUtf8ByteCount(payloadImages);

                    sizeRecords.Add(new SizeRecord
                    {
                        Technique = name,
                        ImageId = SafeImageId(s),

                        OriginalPixelsW = origW,
                        OriginalPixelsH = origH,

                        // Kept for compatibility: dims of primary payload image
                        ProcessedPixelsW = procW,
                        ProcessedPixelsH = procH,

                        // NEW: total pixels across all payload images
                        ProcessedPixelsTotal = totalPixels,

                        OriginalBinaryBytes = originalBinaryBytes,

                        // This is what you asked: ROI bytes + thumbnail bytes (or thumbnail only, etc.)
                        ProcessedBinaryBytes = composite.BinaryBytesTotal,

                        CompressionRatio = ratio,

                        // NEW: totals across all payload images
                        Base64CharsTotal = composite.Base64CharsTotal,
                        Base64BytesUtf8Total = composite.Base64Utf8BytesTotal,
                        DataUrlBytesUtf8Total = composite.DataUrlUtf8BytesTotal,

                        // Kept name, but now represents the realistic JSON for N input_image entries
                        JsonBytesUtf8 = jsonUtf8Bytes,

                        PayloadImageCount = payloadImages.Count
                    });
                }
            }
        }

        string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss");

        string rawCsvPath = Path.Combine(outDir, $"PreprocessingTimes_{ts}.csv");
        WriteTimeCsv(rawCsvPath, timeRecords);

        string sizeCsvPath = Path.Combine(outDir, $"PreprocessingSizes_{ts}.csv");
        WriteSizeCsv(sizeCsvPath, sizeRecords);

        var averages = timeRecords
            .GroupBy(r => r.Technique)
            .Select(g => new AvgRecord
            {
                Technique = g.Key,
                AvgTimeMs = g.Average(x => x.TimeMs),
                Samples = g.Count()
            })
            .OrderBy(x => x.AvgTimeMs)
            .ToList();

        string avgCsvPath = Path.Combine(outDir, $"PreprocessingAverages_{ts}.csv");
        WriteAvgCsv(avgCsvPath, averages);

        Console.WriteLine("Preprocessing Techniques: Average time per image (ms)");
        foreach (var a in averages)
            Console.WriteLine($"- {a.Technique}: {a.AvgTimeMs:F2} ms (n={a.Samples})");

        Console.WriteLine();
        Console.WriteLine($"Raw timings: {rawCsvPath}");
        Console.WriteLine($"Averages:    {avgCsvPath}");
        Console.WriteLine($"Sizes:       {sizeCsvPath}");
    }

    private static long GetOriginalBinaryBytes(DatasetSample s)
    {
        if (!string.IsNullOrWhiteSpace(s.ImagePath) && File.Exists(s.ImagePath))
            return new FileInfo(s.ImagePath).Length;
        return 0;
    }

    private static (int W, int H) GetOriginalPixelDims(DatasetSample s)
    {
        if (string.IsNullOrWhiteSpace(s.ImagePath) || !File.Exists(s.ImagePath))
            return (0, 0);

        var info = Image.Identify(s.ImagePath);
        if (info == null) return (0, 0);
        return (info.Width, info.Height);
    }

    // Returns (decoded binary bytes, base64 chars, base64 utf8 bytes, data url utf8 bytes)
    private static (long BinaryBytes, int Base64Chars, int Base64Utf8Bytes, int DataUrlUtf8Bytes)
        GetDataUrlSizeStats(string dataUrl)
    {
        int comma = dataUrl.IndexOf(',');
        if (comma < 0)
            throw new ArgumentException("Invalid data URL (missing comma).");

        string b64 = dataUrl[(comma + 1)..].Trim();

        byte[] bytes = Convert.FromBase64String(b64);

        int base64Chars = b64.Length;
        int base64Utf8Bytes = Encoding.UTF8.GetByteCount(b64);
        int dataUrlUtf8Bytes = Encoding.UTF8.GetByteCount(dataUrl);

        return (bytes.LongLength, base64Chars, base64Utf8Bytes, dataUrlUtf8Bytes);
    }

    private static (int W, int H) GetPixelDimsFromDataUrl(string dataUrl)
    {
        int comma = dataUrl.IndexOf(',');
        if (comma < 0) return (0, 0);

        string b64 = dataUrl[(comma + 1)..].Trim();
        byte[] bytes = Convert.FromBase64String(b64);

        var info = Image.Identify(bytes);
        if (info == null) return (0, 0);
        return (info.Width, info.Height);
    }

    // Counts bytes of a realistic Realtime JSON event with 1+ images embedded.
    private static int BuildRealtimeJsonUtf8ByteCount(IReadOnlyList<string> imageDataUrls)
    {
        var content = new List<object>
        {
            new { type = "input_text", text = "test" }
        };

        foreach (var url in imageDataUrls)
            content.Add(new { type = "input_image", image_url = url });

        var payload = new
        {
            type = "conversation.item.create",
            item = new
            {
                type = "message",
                role = "user",
                content = content.ToArray()
            }
        };

        return JsonSerializer.SerializeToUtf8Bytes(payload).Length;
    }

    private static List<string> GetPayloadImageDataUrls(PreprocessedSample processed)
    {
        // Your requested semantics: treat ROI + thumb as the processed payload.
        // Do not include processed.ImageDataUrl if it is just the original (optional field).
        if (!string.IsNullOrWhiteSpace(processed.RoiImageDataUrl))
        {
            var list = new List<string> { processed.RoiImageDataUrl };
            if (!string.IsNullOrWhiteSpace(processed.GlobalThumbnailDataUrl))
                list.Add(processed.GlobalThumbnailDataUrl);
            return list;
        }

        if (!string.IsNullOrWhiteSpace(processed.GlobalThumbnailDataUrl))
            return new List<string> { processed.GlobalThumbnailDataUrl };

        if (!string.IsNullOrWhiteSpace(processed.ImageDataUrl))
            return new List<string> { processed.ImageDataUrl };

        return new List<string>();
    }

    private static string GetPrimaryDataUrl(PreprocessedSample processed)
    {
        if (!string.IsNullOrWhiteSpace(processed.RoiImageDataUrl))
            return processed.RoiImageDataUrl;
        if (!string.IsNullOrWhiteSpace(processed.GlobalThumbnailDataUrl))
            return processed.GlobalThumbnailDataUrl;
        return processed.ImageDataUrl ?? "";
    }

    private static CompositeStats GetCompositeDataUrlStats(IReadOnlyList<string> dataUrls)
    {
        long binTotal = 0;
        int b64CharsTotal = 0;
        int b64Utf8Total = 0;
        int dataUrlUtf8Total = 0;

        foreach (var url in dataUrls)
        {
            var (bin, b64Chars, b64Utf8, dataUtf8) = GetDataUrlSizeStats(url);
            binTotal += bin;
            b64CharsTotal += b64Chars;
            b64Utf8Total += b64Utf8;
            dataUrlUtf8Total += dataUtf8;
        }

        return new CompositeStats
        {
            BinaryBytesTotal = binTotal,
            Base64CharsTotal = b64CharsTotal,
            Base64Utf8BytesTotal = b64Utf8Total,
            DataUrlUtf8BytesTotal = dataUrlUtf8Total
        };
    }

    private sealed class CompositeStats
    {
        public long BinaryBytesTotal { get; set; }
        public int Base64CharsTotal { get; set; }
        public int Base64Utf8BytesTotal { get; set; }
        public int DataUrlUtf8BytesTotal { get; set; }
    }

    private static void SaveVisualizationImages(
        string outDir,
        DatasetSample sample,
        List<(string Name, IPreprocessor Preprocessor)> techniques)
    {
        string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string visDir = Path.Combine(outDir, $"sample-visualization_{ts}");
        Directory.CreateDirectory(visDir);

        if (!string.IsNullOrWhiteSpace(sample.ImagePath) && File.Exists(sample.ImagePath))
        {
            string ext = Path.GetExtension(sample.ImagePath);
            if (string.IsNullOrWhiteSpace(ext)) ext = ".jpg";

            string originalOut = Path.Combine(visDir, $"original{ext}");
            File.Copy(sample.ImagePath, originalOut, overwrite: true);
        }

        foreach (var (name, pre) in techniques)
        {
            var processed = pre.Preprocess(sample);

            if (string.IsNullOrWhiteSpace(processed.RoiImageDataUrl) && !string.IsNullOrWhiteSpace(processed.ImageDataUrl))
            {
                string outPath = Path.Combine(visDir, $"{SanitizeFileName(name)}.jpg");
                ImageOutput.SaveDataUrl(processed.ImageDataUrl, outPath);
            }

            if (!string.IsNullOrWhiteSpace(processed.RoiImageDataUrl))
            {
                string outPath = Path.Combine(visDir, $"{SanitizeFileName(name)}_roi.jpg");
                ImageOutput.SaveDataUrl(processed.RoiImageDataUrl, outPath);
            }

            if (!string.IsNullOrWhiteSpace(processed.GlobalThumbnailDataUrl))
            {
                string outPath = Path.Combine(visDir, $"{SanitizeFileName(name)}_global.jpg");
                ImageOutput.SaveDataUrl(processed.GlobalThumbnailDataUrl, outPath);
            }
        }

        Console.WriteLine($"Saved visualization images to: {visDir}");
    }

    private static string SanitizeFileName(string s)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            s = s.Replace(c, '_');
        return s;
    }

    private static string SafeImageId(DatasetSample s)
    {
        if (!string.IsNullOrWhiteSpace(s.ImagePath))
            return Path.GetFileName(s.ImagePath);
        return s.Id ?? "unknown";
    }

    private static void WriteTimeCsv(string path, List<TimeRecord> records)
    {
        using var w = new StreamWriter(path);
        w.WriteLine("technique,image_id,time_ms");
        foreach (var r in records)
        {
            w.WriteLine(string.Join(",",
                EscapeCsv(r.Technique),
                EscapeCsv(r.ImageId),
                r.TimeMs.ToString("F3", CultureInfo.InvariantCulture)));
        }
    }

    private static void WriteSizeCsv(string path, List<SizeRecord> records)
    {
        using var w = new StreamWriter(path);
        w.WriteLine(string.Join(",",
            "technique",
            "image_id",
            "original_pixels_w",
            "original_pixels_h",
            "processed_pixels_w",
            "processed_pixels_h",
            "processed_pixels_total",
            "payload_image_count",
            "original_binary_bytes",
            "processed_binary_bytes",
            "compression_ratio",
            "base64_chars_total",
            "base64_bytes_utf8_total",
            "data_url_bytes_utf8_total",
            "json_bytes_utf8"
        ));

        foreach (var r in records)
        {
            w.WriteLine(string.Join(",",
                EscapeCsv(r.Technique),
                EscapeCsv(r.ImageId),
                r.OriginalPixelsW.ToString(CultureInfo.InvariantCulture),
                r.OriginalPixelsH.ToString(CultureInfo.InvariantCulture),
                r.ProcessedPixelsW.ToString(CultureInfo.InvariantCulture),
                r.ProcessedPixelsH.ToString(CultureInfo.InvariantCulture),
                r.ProcessedPixelsTotal.ToString(CultureInfo.InvariantCulture),
                r.PayloadImageCount.ToString(CultureInfo.InvariantCulture),
                r.OriginalBinaryBytes.ToString(CultureInfo.InvariantCulture),
                r.ProcessedBinaryBytes.ToString(CultureInfo.InvariantCulture),
                r.CompressionRatio.ToString("F6", CultureInfo.InvariantCulture),
                r.Base64CharsTotal.ToString(CultureInfo.InvariantCulture),
                r.Base64BytesUtf8Total.ToString(CultureInfo.InvariantCulture),
                r.DataUrlBytesUtf8Total.ToString(CultureInfo.InvariantCulture),
                r.JsonBytesUtf8.ToString(CultureInfo.InvariantCulture)
            ));
        }
    }

    private static void WriteAvgCsv(string path, List<AvgRecord> records)
    {
        using var w = new StreamWriter(path);
        w.WriteLine("technique,avg_time_ms,samples");
        foreach (var r in records)
        {
            w.WriteLine(string.Join(",",
                EscapeCsv(r.Technique),
                r.AvgTimeMs.ToString("F3", CultureInfo.InvariantCulture),
                r.Samples.ToString(CultureInfo.InvariantCulture)));
        }
    }

    private static string EscapeCsv(string s)
    {
        if (s.Contains(',') || s.Contains('"') || s.Contains('\n') || s.Contains('\r'))
            return $"\"{s.Replace("\"", "\"\"")}\"";
        return s;
    }

    private sealed class TimeRecord
    {
        public string Technique { get; set; } = "";
        public string ImageId { get; set; } = "";
        public double TimeMs { get; set; }
    }

    private sealed class SizeRecord
    {
        public string Technique { get; set; } = "";
        public string ImageId { get; set; } = "";

        public int OriginalPixelsW { get; set; }
        public int OriginalPixelsH { get; set; }

        // Backwards compatible: dims of the primary payload image (ROI or thumb or single processed image)
        public int ProcessedPixelsW { get; set; }
        public int ProcessedPixelsH { get; set; }

        // NEW: total pixels across all payload images (ROI + thumb, etc.)
        public long ProcessedPixelsTotal { get; set; }

        public int PayloadImageCount { get; set; }

        public long OriginalBinaryBytes { get; set; }

        // NEW behavior: total bytes across all payload images
        public long ProcessedBinaryBytes { get; set; }

        public double CompressionRatio { get; set; }

        // NEW: totals across all payload images
        public int Base64CharsTotal { get; set; }
        public int Base64BytesUtf8Total { get; set; }
        public int DataUrlBytesUtf8Total { get; set; }

        public int JsonBytesUtf8 { get; set; }
    }

    private sealed class AvgRecord
    {
        public string Technique { get; set; } = "";
        public double AvgTimeMs { get; set; }
        public int Samples { get; set; }
    }
}
