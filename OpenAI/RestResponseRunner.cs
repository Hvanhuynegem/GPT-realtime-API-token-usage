using System.Net.Http;
using System.Text.Json;
using System.Text;
using System.Diagnostics;
using Thesis.Metrics;
using Thesis.Preprocessing;

namespace Thesis.OpenAI;

public static class RestResponseRunner
{
    public static async Task<SampleMetrics> RunSingleSampleRestAsync(
    HttpClient http,
    PreprocessedSample pre,
    ModelConfig cfg)
    {
        const int MaxRetries = 5;
        const int RetryDelayMs = 3000; // 3 seconds

        for (int attempt = 1; attempt <= MaxRetries; attempt++)
        {
            Console.WriteLine($"RunSingleSampleRestAsync attempt {attempt}/{MaxRetries}");

            var metrics = new SampleMetrics();
            var endToEnd = Stopwatch.StartNew();
            string? lastErrorMessage = null;
            bool shouldRetry = false;

            // Build user content parts (same as realtime)
            var contentParts = new List<object>();

            if (!string.IsNullOrWhiteSpace(pre.Text))
            {
                contentParts.Add(new
                {
                    type = "input_text",
                    text = pre.Text
                });
            }

            if (!string.IsNullOrWhiteSpace(pre.ImageDataUrl))
            {
                contentParts.Add(new
                {
                    type = "input_image",
                    image_url = pre.ImageDataUrl,
                    detail = "low"
                });
            }

            // Messages: system (to mirror session.update) + user
            var messages = new List<object>
            {
                new
                {
                    role = "system",
                    content = new object[]
                    {
                        new
                        {
                            type = "input_text",
                            text = "You are a helpful assistant. Answer in one short, direct sentence with no extra details."
                        }
                    }
                },
                new
                {
                    role = "user",
                    content = contentParts.ToArray()
                }
            };

            // Build /v1/responses request body
            var body = new
            {
                model = cfg.ModelName,
                input = messages.ToArray(),
                // Optional: mirror short responses if you want
                // max_output_tokens = 64
            };

            string jsonBody = JsonSerializer.Serialize(body);
            using var requestContent = new StringContent(jsonBody, Encoding.UTF8, "application/json");

            var sendTime = DateTime.UtcNow;
            metrics.ClientSendTimeUtc = sendTime;

            HttpResponseMessage response;
            string responseText;

            try
            {
                response = await http.PostAsync("/v1/responses", requestContent);
                responseText = await response.Content.ReadAsStringAsync();
            }
            catch (Exception ex)
            {
                // Network level error - treat as fatal for now
                endToEnd.Stop();
                Console.WriteLine("REST exception while calling /v1/responses:");
                Console.WriteLine(ex);
                metrics.EndToEndMs = (long)endToEnd.Elapsed.TotalMilliseconds;
                throw;
            }

            var doneTime = DateTime.UtcNow;

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine("REST error from server:");
                Console.WriteLine("Status: " + (int)response.StatusCode + " " + response.StatusCode);
                Console.WriteLine(responseText);

                // Try to inspect error.code to decide about retry
                try
                {
                    using var errDoc = JsonDocument.Parse(responseText);
                    var root = errDoc.RootElement;

                    if (root.TryGetProperty("error", out var errorObj) &&
                        errorObj.ValueKind == JsonValueKind.Object)
                    {
                        string? code = errorObj.TryGetProperty("code", out var codeProp) &&
                                    codeProp.ValueKind == JsonValueKind.String
                            ? codeProp.GetString()
                            : null;

                        string? msg = errorObj.TryGetProperty("message", out var msgProp) &&
                                    msgProp.ValueKind == JsonValueKind.String
                            ? msgProp.GetString()
                            : null;

                        lastErrorMessage = msg ?? code ?? "unknown error";

                        if (code == "rate_limit_exceeded")
                        {
                            shouldRetry = true;
                        }
                    }
                }
                catch
                {
                    // If parsing fails, just do not retry
                }

                endToEnd.Stop();
                metrics.EndToEndMs = (long)endToEnd.Elapsed.TotalMilliseconds;

                if (shouldRetry)
                {
                    if (attempt == MaxRetries)
                    {
                        Console.WriteLine($"Giving up after {MaxRetries} attempts. Last error: {lastErrorMessage}");
                        throw new Exception($"RunSingleSampleRestAsync failed after {MaxRetries} retries. Last error: {lastErrorMessage}");
                    }

                    Console.WriteLine($"Rate limit hit on REST, retrying after {RetryDelayMs} ms. Error: {lastErrorMessage}");
                    await Task.Delay(RetryDelayMs);
                    continue;
                }

                // Non retryable error - return metrics with timing filled but no usage
                return metrics;
            }

            // Success case
            using (var doc = JsonDocument.Parse(responseText))
            {
                var root = doc.RootElement;

                // Collect assistant text from all message outputs
                var assistantBuilder = new StringBuilder();

                if (root.TryGetProperty("output", out var outputElem) &&
                    outputElem.ValueKind == JsonValueKind.Array)
                {
                    foreach (var outputItem in outputElem.EnumerateArray())
                    {
                        if (outputItem.TryGetProperty("type", out var outTypeProp) &&
                            outTypeProp.ValueKind == JsonValueKind.String &&
                            outTypeProp.GetString() == "message" &&
                            outputItem.TryGetProperty("content", out var contentElem) &&
                            contentElem.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var part in contentElem.EnumerateArray())
                            {
                                if (part.TryGetProperty("type", out var typeProp) &&
                                    typeProp.ValueKind == JsonValueKind.String &&
                                    typeProp.GetString() == "output_text" &&
                                    part.TryGetProperty("text", out var textProp) &&
                                    textProp.ValueKind == JsonValueKind.String)
                                {
                                    assistantBuilder.Append(textProp.GetString());
                                }
                            }
                        }
                    }
                }

                metrics.AssistantText = assistantBuilder.ToString();

                // Usage, same fields as realtime
                if (root.TryGetProperty("usage", out var usage))
                {
                    // Console.WriteLine("REST usage: " + usage.ToString());
                    if (usage.TryGetProperty("total_tokens", out var totalTokensProp) &&
                        totalTokensProp.ValueKind == JsonValueKind.Number)
                    {
                        metrics.TotalTokens = totalTokensProp.GetInt32();
                    }

                    if (usage.TryGetProperty("input_tokens", out var inputTokensProp) &&
                        inputTokensProp.ValueKind == JsonValueKind.Number)
                    {
                        metrics.InputTokens = inputTokensProp.GetInt32();
                        metrics.ImageInputTokens = 65; // REST has fixed 65 image tokens per image
                        metrics.TextInputTokens = metrics.InputTokens - metrics.ImageInputTokens; // REST has no BREAKDOWN
                    }

                    if (usage.TryGetProperty("output_tokens", out var outputTokensProp) &&
                        outputTokensProp.ValueKind == JsonValueKind.Number)
                    {
                        metrics.OutputTokens = outputTokensProp.GetInt32();
                    }
                }

                // Timing metrics - we do not get internal timestamps from REST, so keep it simple
                endToEnd.Stop();
                metrics.EndToEndMs = (long)endToEnd.Elapsed.TotalMilliseconds;

                metrics.SendToResponseCreatedMs = metrics.EndToEndMs;
                metrics.ResponseCreatedToFirstTokenMs = 0;
                metrics.FirstTokenToDoneMs = 0;

                // Cost estimate - same as realtime logic
                int textInput = metrics.InputTokens;
                int imageInput = metrics.ImageInputTokens;
                int cached = metrics.CachedTokens;
                int output = metrics.OutputTokens;

                double textInputCost = textInput * cfg.TextInputRate;
                double imageInputCost = imageInput * cfg.ImageInputRate;
                double cachedCost = cached * cfg.CachedInputRate;
                double outputCost = output * cfg.TextOutputRate;

                metrics.TotalCostUsd = textInputCost + imageInputCost + cachedCost + outputCost;

                return metrics;
            }
        }

        // Should never reach here
        throw new Exception("RunSingleSampleRestAsync fell through without returning.");
    }
}