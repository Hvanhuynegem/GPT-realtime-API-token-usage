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

        // 1) config.json: write once
        string configPath = Path.Combine(expDir, "config.json");
        if (!File.Exists(configPath))
        {
            var config = new
            {
                experiment_id = experimentId,
                timestamp = DateTime.UtcNow.ToString("o"),
                model = model,
                dataset_name = datasetName
            };

            string configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(configPath, configJson);
        }

        // 2) metrics.csv: append, write header only once
        string metricsPath = Path.Combine(expDir, "metrics.csv");
        bool writeHeader = !File.Exists(metricsPath);

        using (var sw = new StreamWriter(metricsPath, append: true))
        {
            if (writeHeader)
            {
                sw.WriteLine("preprocessor,sample_id,end_to_end_ms,send_to_response_created_ms,response_created_to_first_token_ms,first_token_to_done_ms,input_tokens,output_tokens,total_tokens,text_input_tokens,image_input_tokens,cached_tokens,total_cost_usd");
            }

            foreach (var m in metrics)
            {
                sw.WriteLine(string.Join(",",
                    EscapeCsv(preprocessorName),
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
        }

        // 3) responses.jsonl: append
        string responsesPath = Path.Combine(expDir, "responses.jsonl");
        using (var sw = new StreamWriter(responsesPath, append: true))
        {
            foreach (var m in metrics)
            {
                sw.WriteLine(JsonSerializer.Serialize(new
                {
                    preprocessor = preprocessorName,
                    sample_id = m.SampleId,
                    assistant_text = m.AssistantText
                }));
            }
        }
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