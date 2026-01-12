using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.Metrics;

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
            // Add more techniques as needed
        };

        Directory.CreateDirectory(outDir);

        var samples = VoilaDatasetLoader
            .LoadSamplesFromVoila(datasetDir, maxSamples: maxSamples)
            .ToList();

        if (samples.Count == 0)
            throw new InvalidOperationException("No samples loaded. Check datasetDir and prepared.jsonl.");

        // Pick one sample for visualization (first one)
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
                long originalBytes = GetOriginalBytes(s);

                var sw = Stopwatch.StartNew();
                var processed = preprocessor.Preprocess(s);
                sw.Stop();

                timeRecords.Add(new TimeRecord
                {
                    Technique = name,
                    ImageId = SafeImageId(s),
                    TimeMs = sw.Elapsed.TotalMilliseconds
                });

                // Size measurement (only if we have an image output)
                if (!string.IsNullOrWhiteSpace(processed.ImageDataUrl) && originalBytes > 0)
                {
                    long processedBytes = Base64PayloadByteCount(processed.ImageDataUrl);
                    double ratio = (double)processedBytes / (double)originalBytes;

                    sizeRecords.Add(new SizeRecord
                    {
                        Technique = name,
                        ImageId = SafeImageId(s),
                        OriginalBytes = originalBytes,
                        ProcessedBytes = processedBytes,
                        CompressionRatio = ratio
                    });
                }
            }
        }

        string ts = DateTime.Now.ToString("yyyyMMdd_HHmmss");

        // Write raw times CSV (for boxplot)
        string rawCsvPath = Path.Combine(outDir, $"PreprocessingTimes_{ts}.csv");
        WriteTimeCsv(rawCsvPath, timeRecords);

        // Write sizes CSV (for size scatter plot)
        string sizeCsvPath = Path.Combine(outDir, $"PreprocessingSizes_{ts}.csv");
        WriteSizeCsv(sizeCsvPath, sizeRecords);

        // Compute averages (for bar chart)
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

    private static long GetOriginalBytes(DatasetSample s)
    {
        if (!string.IsNullOrWhiteSpace(s.ImagePath) && File.Exists(s.ImagePath))
            return new FileInfo(s.ImagePath).Length;
        return 0;
    }

    // data:image/jpeg;base64,<payload>  -> bytes of decoded payload
    private static long Base64PayloadByteCount(string dataUrl)
    {
        int comma = dataUrl.IndexOf(',');
        if (comma < 0)
            throw new ArgumentException("Invalid data URL (missing comma).");

        string b64 = dataUrl[(comma + 1)..];

        // Some data URLs might contain whitespace/newlines
        b64 = b64.Trim();

        byte[] bytes = Convert.FromBase64String(b64);
        return bytes.LongLength;
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
        w.WriteLine("technique,image_id,original_bytes,processed_bytes,compression_ratio");
        foreach (var r in records)
        {
            w.WriteLine(string.Join(",",
                EscapeCsv(r.Technique),
                EscapeCsv(r.ImageId),
                r.OriginalBytes.ToString(CultureInfo.InvariantCulture),
                r.ProcessedBytes.ToString(CultureInfo.InvariantCulture),
                r.CompressionRatio.ToString("F6", CultureInfo.InvariantCulture)));
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
        public long OriginalBytes { get; set; }
        public long ProcessedBytes { get; set; }
        public double CompressionRatio { get; set; }
    }

    private sealed class AvgRecord
    {
        public string Technique { get; set; } = "";
        public double AvgTimeMs { get; set; }
        public int Samples { get; set; }
    }
}
