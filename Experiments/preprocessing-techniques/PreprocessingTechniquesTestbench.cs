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

                if (!string.IsNullOrWhiteSpace(processed.ImageDataUrl) && originalBinaryBytes > 0 && origW > 0 && origH > 0)
                {
                    var (processedBinaryBytes, base64Chars, base64Utf8Bytes, dataUrlUtf8Bytes) =
                        GetDataUrlSizeStats(processed.ImageDataUrl);

                    var (procW, procH) = GetPixelDimsFromDataUrl(processed.ImageDataUrl);

                    double ratio = (double)processedBinaryBytes / (double)originalBinaryBytes;

                    int jsonUtf8Bytes = BuildRealtimeJsonUtf8ByteCount(processed.ImageDataUrl);

                    sizeRecords.Add(new SizeRecord
                    {
                        Technique = name,
                        ImageId = SafeImageId(s),

                        OriginalPixelsW = origW,
                        OriginalPixelsH = origH,
                        ProcessedPixelsW = procW,
                        ProcessedPixelsH = procH,

                        OriginalBinaryBytes = originalBinaryBytes,
                        ProcessedBinaryBytes = processedBinaryBytes,
                        CompressionRatio = ratio,

                        Base64Chars = base64Chars,
                        Base64BytesUtf8 = base64Utf8Bytes,
                        DataUrlBytesUtf8 = dataUrlUtf8Bytes,
                        JsonBytesUtf8 = jsonUtf8Bytes
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

    // Counts bytes of a realistic Realtime JSON event with the data URL embedded.
    private static int BuildRealtimeJsonUtf8ByteCount(string imageDataUrl)
    {
        var payload = new
        {
            type = "conversation.item.create",
            item = new
            {
                type = "message",
                role = "user",
                content = new object[]
                {
                    new { type = "input_text", text = "test" },
                    new { type = "input_image", image_url = imageDataUrl }
                }
            }
        };

        return JsonSerializer.SerializeToUtf8Bytes(payload).Length;
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

            if (!string.IsNullOrWhiteSpace(processed.ImageDataUrl))
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
            "original_binary_bytes",
            "processed_binary_bytes",
            "compression_ratio",
            "base64_chars",
            "base64_bytes_utf8",
            "data_url_bytes_utf8",
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
                r.OriginalBinaryBytes.ToString(CultureInfo.InvariantCulture),
                r.ProcessedBinaryBytes.ToString(CultureInfo.InvariantCulture),
                r.CompressionRatio.ToString("F6", CultureInfo.InvariantCulture),
                r.Base64Chars.ToString(CultureInfo.InvariantCulture),
                r.Base64BytesUtf8.ToString(CultureInfo.InvariantCulture),
                r.DataUrlBytesUtf8.ToString(CultureInfo.InvariantCulture),
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
        public int ProcessedPixelsW { get; set; }
        public int ProcessedPixelsH { get; set; }

        public long OriginalBinaryBytes { get; set; }
        public long ProcessedBinaryBytes { get; set; }
        public double CompressionRatio { get; set; }

        public int Base64Chars { get; set; }
        public int Base64BytesUtf8 { get; set; }
        public int DataUrlBytesUtf8 { get; set; }
        public int JsonBytesUtf8 { get; set; }
    }

    private sealed class AvgRecord
    {
        public string Technique { get; set; } = "";
        public double AvgTimeMs { get; set; }
        public int Samples { get; set; }
    }
}