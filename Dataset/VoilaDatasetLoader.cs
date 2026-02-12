using System.Text.Json;

namespace Thesis.Dataset;

public static class VoilaDatasetLoader
{
    public static IEnumerable<DatasetSample> LoadSamplesFromVoila(string datasetDir, int? maxSamples = null)
    {
        string jsonlPath = Path.Combine(datasetDir, "prepared.jsonl");
        if (!File.Exists(jsonlPath))
            throw new FileNotFoundException($"Cannot find {jsonlPath}");

        int count = 0;
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        foreach (var line in File.ReadLines(jsonlPath))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            PreparedRow? row;
            try
            {
                row = JsonSerializer.Deserialize<PreparedRow>(line, options);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: could not parse line: {ex.Message}");
                continue;
            }

            if (row == null)
                continue;

            yield return new DatasetSample
            {
                Id = row.id ?? "",
                Text = row.question ?? "",
                Answer = row.answer ?? "",
                ImagePath = row.image_path,
                Trace = ConvertGazePoints(row.gaze_points_px)
            };

            count++;
            if (maxSamples.HasValue && count >= maxSamples.Value)
                yield break;
        }
    }

    public static IEnumerable<DatasetSample> LoadSamplesFromVQAMHUG(
    string datasetDir,
    int? maxSamples = null,
    int seed = 12345,
    bool imagePlateOnly = true)
    {
        string jsonPath = Path.Combine(datasetDir, "prepared.json");
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException($"Cannot find {jsonPath}");

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };

        using var fs = File.OpenRead(jsonPath);

        var rows = JsonSerializer.Deserialize<List<VqaMhugRow>>(fs, options)
                ?? new List<VqaMhugRow>();

        int count = 0;

        foreach (var row in rows)
        {
            if (row == null)
                continue;

            var allPts = row.gaze_points_px ?? new List<GazePoint>();

            IEnumerable<GazePoint> pts = allPts;
            if (imagePlateOnly)
                pts = pts.Where(p => string.Equals(p.plate, "imgplate", StringComparison.OrdinalIgnoreCase));

            // Collect participants available for this sample
            var participantIds = pts
                .Select(p => p.participant_id)
                .Where(id => !string.IsNullOrWhiteSpace(id))
                .Distinct()
                .OrderBy(id => id, StringComparer.Ordinal) // stable ordering
                .ToList();

            // Pick one participant deterministically-random using (seed + sampleId)
            string? chosenPid = null;
            if (participantIds.Count > 0)
            {
                int sampleId = row.id ?? 0;
                var rng = new Random(HashCode.Combine(seed, sampleId));
                chosenPid = participantIds[rng.Next(participantIds.Count)];
                pts = pts.Where(p => p.participant_id == chosenPid);
            }

            string answerPacked = row.answers == null ? "" : string.Join("\n", row.answers);

            yield return new DatasetSample
            {
                Id = row.id?.ToString() ?? "",
                Text = row.question ?? "",
                Answer = answerPacked,
                ImagePath = row.image_path ?? "",
                Trace = ConvertGazeFixations(pts.ToList())
            };

            count++;
            if (maxSamples.HasValue && count >= maxSamples.Value)
                yield break;
        }
    }



    private static IReadOnlyList<(int x, int y)> ConvertGazePoints(List<List<int>>? gazePointsPx)
    {
        if (gazePointsPx == null || gazePointsPx.Count == 0)
            return Array.Empty<(int, int)>();

        var points = new List<(int x, int y)>(gazePointsPx.Count);

        foreach (var p in gazePointsPx)
        {
            // Expect [x, y]
            if (p == null || p.Count < 2)
                continue;

            points.Add((p[0], p[1]));
        }

        return points;
    }

    private static IReadOnlyList<(int x, int y)> ConvertGazeFixations(List<GazePoint>? fixations)
    {
        if (fixations == null || fixations.Count == 0)
            return Array.Empty<(int, int)>();

        var points = new List<(int x, int y)>(fixations.Count);

        foreach (var f in fixations)
        {
            // Prefer COCO pixel coords if present, otherwise fall back to legacy x/y.
            int x = f.x_px ?? f.x;
            int y = f.y_px ?? f.y;
            points.Add((x, y));
        }

        return points;
    }

    public sealed class GazePoint
    {
        public string? participant_id { get; set; }
        public int fix_idx { get; set; }

        // Legacy/stimulus coords (may still exist in older prepared.json files)
        public int x { get; set; }
        public int y { get; set; }

        // New COCO image pixel coords from your updated preprocessing script
        public int? x_px { get; set; }
        public int? y_px { get; set; }

        public double duration { get; set; }
        public string? plate { get; set; }   // "imgplate", "txtplate", "centerfix", ...
        public string? eye { get; set; }
        public double start { get; set; }
        public double end { get; set; }
    }



    public sealed class VqaMhugRow
    {
        public int? id { get; set; }
        public string? question { get; set; }
        public List<string>? answers { get; set; }
        public string? image_path { get; set; }
        public List<GazePoint>? gaze_points_px { get; set; }
    }

}
