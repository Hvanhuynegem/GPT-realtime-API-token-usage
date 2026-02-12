using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.Metrics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using System.Runtime.InteropServices;


class GazeRoiExperiment
{
    public static void Run()
    {
        IPreprocessor preprocessor = new GazeRoiAndThumbnailPreprocessor();

        int maxSamples = 50;      // load enough so index i exists
        int i = 11;                // 0-based index: i=9 saves the 10th sample

        string dataDir = Path.Combine(Directory.GetCurrentDirectory(), "data", "VQA-HMUG-data");
        Console.WriteLine("Dataset dir: " + dataDir);

        var rawSamples = VoilaDatasetLoader
            .LoadSamplesFromVQAMHUG(dataDir, maxSamples)
            .ToList();

        if (rawSamples.Count <= i)
        {
            Console.WriteLine($"Not enough samples loaded. Loaded {rawSamples.Count}, need index {i}.");
            return;
        }

        var sample = rawSamples[i];

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string outDir = Path.Combine("outputs");
        Directory.CreateDirectory(outDir);

        Console.WriteLine($"Sample ID: {sample.Id}");

        // Save original image with gaze dots
        {
            string imagePath = ResolveImagePath(sample.ImagePath);
            if (!File.Exists(imagePath))
            {
                Console.WriteLine("Resolved image path does not exist: " + imagePath);
            }
            else
            {
                string outPath = Path.Combine(outDir, $"GazeRoiExperiment_{timestamp}_idx_{i}_gaze_dots.png");
                DrawGazeDotsAndSave(imagePath, sample.Trace ?? Array.Empty<(int x, int y)>(), outPath);
                Console.WriteLine("Saved gaze overlay: " + outPath);
            }
        }

        // Keep your existing preprocessing outputs (ROI crops + global thumbnail)
        PreprocessedSample processed = preprocessor.Preprocess(sample);

        // Save ROI crops (0..N)
        if (processed.Rois != null && processed.Rois.Count > 0)
        {
            foreach (var roi in processed.Rois)
            {
                if (string.IsNullOrWhiteSpace(roi.ImageDataUrl))
                    continue;

                string roiPath = Path.Combine(
                    outDir,
                    $"GazeRoiExperiment_{timestamp}_idx_{i}_roi_{roi.Index}.jpg"
                );

                ImageOutput.SaveDataUrl(roi.ImageDataUrl, roiPath);
            }
        }

        // Save global thumbnail
        if (!string.IsNullOrWhiteSpace(processed.GlobalThumbnailDataUrl))
        {
            string thumbPath = Path.Combine(
                outDir,
                $"GazeRoiExperiment_{timestamp}_idx_{i}_global.jpg"
            );

            ImageOutput.SaveDataUrl(processed.GlobalThumbnailDataUrl, thumbPath);
        }
    }

    private static string ResolveImagePath(string? rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
            return rawPath ?? "";

        if (Path.IsPathRooted(rawPath))
            return rawPath;

        string p = rawPath.Replace('/', Path.DirectorySeparatorChar);
        string cwd = Directory.GetCurrentDirectory();

        // Most common case: JSON stores "data/VQA-HMUG-data/..."
        string candidate1 = Path.Combine(cwd, p);
        if (File.Exists(candidate1))
            return candidate1;

        // If it starts with "data\<...>", also try stripping leading "data\"
        if (p.StartsWith("data" + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        {
            string stripped = p.Substring(("data" + Path.DirectorySeparatorChar).Length);
            string candidate2 = Path.Combine(cwd, "data", stripped);
            if (File.Exists(candidate2))
                return candidate2;
        }

        return candidate1;
    }

    private static void DrawGazeDotsAndSave(
    string imagePath,
    IReadOnlyList<(int x, int y)> points,
    string outPath)
    {
        using Image<Rgba32> image = Image.Load<Rgba32>(imagePath);

        Console.WriteLine($"Image size: {image.Width} x {image.Height}");
        Console.WriteLine($"Total gaze points: {points.Count}");

        if (points.Count == 0)
        {
            Console.WriteLine("No gaze points present.");
            image.Save(outPath);
            return;
        }

        int minX = points.Min(p => p.x);
        int maxX = points.Max(p => p.x);
        int minY = points.Min(p => p.y);
        int maxY = points.Max(p => p.y);

        Console.WriteLine($"Gaze X range: {minX} → {maxX}");
        Console.WriteLine($"Gaze Y range: {minY} → {maxY}");

        int inside = 0;
        int outside = 0;

        const int radius = 10;

        image.Mutate(ctx =>
        {
            foreach (var (x, y) in points)
            {
                if (x < 0 || y < 0 || x >= image.Width || y >= image.Height)
                {
                    outside++;
                    continue;
                }

                inside++;

                ctx.Fill(
                    Color.Red,
                    new Rectangle(x - radius, y - radius, radius * 2, radius * 2)
                );
            }
        });

        Console.WriteLine($"Points inside image: {inside}");
        Console.WriteLine($"Points outside image: {outside}");

        image.Save(outPath);
    }



    // Keeping this here in case you still use it elsewhere
    private static DatasetSample LoadSample()
    {
        string datasetDir = "data"; // root containing prepared.jsonl

        return VoilaDatasetLoader
            .LoadSamplesFromVoila(datasetDir, maxSamples: 1)
            .First();
    }
}
