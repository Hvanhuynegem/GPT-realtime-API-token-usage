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
                Id = row.id,
                Text = row.question,
                Answer = row.answer,
                ImagePath = row.image_path
            };

            count++;
            if (maxSamples.HasValue && count >= maxSamples.Value)
                yield break;
        }
    }
}