using System.Text.Json;
using System.Text;

namespace Thesis.Metrics;

public static class ExperimentLogger
{
    public static void SaveExperimentResults(
    string baseDir,
    string experimentId,
    string model,
    string datasetName,
    string preprocessorName,
    List<SampleMetrics> metrics)
    {
        string expDir = Path.Combine(baseDir, experimentId);
        Directory.CreateDirectory(expDir);

        // 1. Save config.json
        var config = new
        {
            experiment_id = experimentId,
            timestamp = DateTime.UtcNow.ToString("o"),
            model = model,
            dataset_name = datasetName,
            num_samples = metrics.Count,
            preprocessor = preprocessorName
        };

        string configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(Path.Combine(expDir, "config.json"), configJson);

        // 2. Save metrics.csv
        var sb = new StringBuilder();
        sb.AppendLine("sample_id,end_to_end_ms,send_to_response_created_ms,response_created_to_first_token_ms,first_token_to_done_ms,input_tokens,output_tokens,total_tokens,text_input_tokens,image_input_tokens,cached_tokens,total_cost_usd");

        foreach (var m in metrics)
        {
            sb.AppendLine(string.Join(",",
                EscapeCsv(m.SampleId),
                m.EndToEndMs,
                m.SendToResponseCreatedMs,
                m.ResponseCreatedToFirstTokenMs,
                m.FirstTokenToDoneMs,
                m.InputTokens,
                m.OutputTokens,
                m.TotalTokens,
                m.TextInputTokens,
                m.ImageInputTokens,
                m.CachedTokens,
                m.TotalCostUsd.ToString("F8", System.Globalization.CultureInfo.InvariantCulture)));
        }

        File.WriteAllText(Path.Combine(expDir, "metrics.csv"), sb.ToString());

        // 3. Optional: save raw responses as JSONL
        // One line per sample with sample id and assistant text etc.
        var lines = metrics.Select(m => JsonSerializer.Serialize(new
        {
            sample_id = m.SampleId,
            assistant_text = m.AssistantText
        }));
        File.WriteAllLines(Path.Combine(expDir, "responses.jsonl"), lines);
    }

    static string EscapeCsv(string value)
    {
        if (value == null) return "";
        bool mustQuote = value.Contains(",") || value.Contains("\"") || value.Contains("\n");
        if (!mustQuote) return value;
        string escaped = value.Replace("\"", "\"\"");
        return "\"" + escaped + "\"";
    }
}