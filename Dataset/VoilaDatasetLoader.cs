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

    public static IEnumerable<DatasetSample> LoadSamplesFromVQAMHUG(string datasetDir, int? maxSamples = null)
    {
        string jsonPath = Path.Combine(datasetDir, "prepared.json");
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException($"Cannot find {jsonPath}");

        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        using var fs = File.OpenRead(jsonPath);

        var rows = JsonSerializer.Deserialize<List<VqaMhugRow>>(fs, options)
                ?? new List<VqaMhugRow>();

        int count = 0;

        foreach (var row in rows)
        {
            if (row == null)
                continue;

            string answerPacked = row.answers == null
                ? ""
                : string.Join("\n", row.answers);

            yield return new DatasetSample
            {
                Id = row.id?.ToString() ?? "",
                Text = row.question ?? "",
                Answer = answerPacked,
                ImagePath = row.image_path ?? "",
                Trace = ConvertGazeFixations(row.gaze_points_px)
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

    private static IReadOnlyList<(int x, int y)> ConvertGazeFixations(List<VqaMhugFixation>? fixations)
    {
        if (fixations == null || fixations.Count == 0)
            return Array.Empty<(int, int)>();

        var points = new List<(int x, int y)>(fixations.Count);

        foreach (var f in fixations)
        {
            if (!f.x.HasValue || !f.y.HasValue)
                continue;

            points.Add((f.x.Value, f.y.Value));
        }

        return points;
    }

    private sealed class VqaMhugRow
    {
        public int? id { get; set; }
        public string? question { get; set; }
        public List<string>? answers { get; set; }
        public string? image_path { get; set; }
        public List<VqaMhugFixation>? gaze_points_px { get; set; }
    }

    private sealed class VqaMhugFixation
    {
        public int? x { get; set; }
        public int? y { get; set; }
    }
}
