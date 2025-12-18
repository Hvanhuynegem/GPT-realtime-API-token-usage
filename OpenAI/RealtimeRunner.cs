using System.Net.WebSockets;
using System.Text.Json;
using System.Text;
using System.Diagnostics;
using Thesis.Metrics;
using Thesis.Preprocessing;

namespace Thesis.OpenAI;

public static class RealtimeRunner
{
    public static async Task<SampleMetrics> RunSingleSampleRealtimeAsync(
        ClientWebSocket ws,
        PreprocessedSample pre,
        ModelConfig cfg)
    {
        const int MaxRetries = 5;
        const int RetryDelayMs = 3000; // 3 seconds

        for (int attempt = 1; attempt <= MaxRetries; attempt++)
        {
            await Task.Delay(RetryDelayMs);
            Console.WriteLine($"RunSingleSampleAsync attempt {attempt}/{MaxRetries}");

            var buffer = new byte[64 * 1024];

            // Build conversation.item.create content array
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
                    // Optional detail: "low" or "high"
                    detail = "low"
                });
            }

            var createItemEvent = new
            {
                type = "conversation.item.create",
                previous_item_id = "",
                item = new
                {
                    type = "message",
                    role = "user",
                    content = contentParts
                }
            };

            string createItemJson = JsonSerializer.Serialize(createItemEvent);
            string responseCreateJson = @"{ ""type"": ""response.create"" }";

            var metrics = new SampleMetrics();

            // Timers
            var endToEnd = Stopwatch.StartNew();
            DateTime sendTime = DateTime.UtcNow;
            metrics.ClientSendTimeUtc = sendTime;

            DateTime? tResponseCreated = null;
            DateTime? tFirstToken = null;
            DateTime? tDone = null;
            bool firstTokenSeen = false;
            string currentResponseId = "";

            bool shouldRetry = false;
            string? lastErrorMessage = null;

            // Send the item and then response.create
            await SendJson(ws, createItemJson);
            await SendJson(ws, responseCreateJson);

            // Console.WriteLine("Sent conversation.item.create and response.create");

            // Collect assistant text for debugging
            var assistantBuilder = new StringBuilder();

            while (ws.State == WebSocketState.Open)
            {
                var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    Console.WriteLine($"Server closed connection. Status: {result.CloseStatus}, Desc: {result.CloseStatusDescription}");
                    break;
                }

                var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                // Console.WriteLine("Raw event: " + json);

                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                if (!root.TryGetProperty("type", out var typeProp))
                    continue;

                var type = typeProp.GetString();

                if (type == "response.created")
                {
                    var response = root.GetProperty("response");
                    currentResponseId = response.GetProperty("id").GetString() ?? "";
                    tResponseCreated = DateTime.UtcNow;
                }
                else if (type == "response.output_text.delta" || type == "response.text.delta")
                {
                    // First token time stamp
                    if (!firstTokenSeen)
                    {
                        firstTokenSeen = true;
                        tFirstToken = DateTime.UtcNow;
                    }

                    if (root.TryGetProperty("delta", out var deltaProp) &&
                        deltaProp.ValueKind == JsonValueKind.String)
                    {
                        var chunk = deltaProp.GetString();
                        if (!string.IsNullOrEmpty(chunk))
                        {
                            assistantBuilder.Append(chunk);
                        }
                    }
                }
                else if (type == "response.output_item.done")
                {
                    // Some outputs also come as full messages in this event, but we already accumulate deltas above.
                    continue;
                }
                else if (type == "response.done")
                {
                    tDone = DateTime.UtcNow;
                    endToEnd.Stop();

                    var response = root.GetProperty("response");
                    var status = response.GetProperty("status").GetString();

                    if (status != "completed")
                    {
                        Console.WriteLine($"Response status is {status}, not completed.");

                        if (response.TryGetProperty("status_details", out var statusDetails) &&
                            statusDetails.ValueKind == JsonValueKind.Object &&
                            statusDetails.TryGetProperty("error", out var errorObj) &&
                            errorObj.ValueKind == JsonValueKind.Object)
                        {
                            string? code = errorObj.TryGetProperty("code", out var codeProp) && codeProp.ValueKind == JsonValueKind.String
                                ? codeProp.GetString()
                                : null;
                            string? msg = errorObj.TryGetProperty("message", out var msgProp) && msgProp.ValueKind == JsonValueKind.String
                                ? msgProp.GetString()
                                : null;

                            lastErrorMessage = msg ?? code ?? "unknown error";

                            if (code == "rate_limit_exceeded")
                            {
                                shouldRetry = true;
                            }
                        }
                    }

                    // Extract usage only if we are not going to retry
                    if (!shouldRetry && response.TryGetProperty("usage", out var usage))
                    {
                        metrics.TotalTokens = usage.GetProperty("total_tokens").GetInt32();
                        metrics.InputTokens = usage.GetProperty("input_tokens").GetInt32();
                        metrics.OutputTokens = usage.GetProperty("output_tokens").GetInt32();

                        if (usage.TryGetProperty("input_token_details", out var inputDetails))
                        {
                            if (inputDetails.TryGetProperty("cached_tokens", out var cachedProp) &&
                                cachedProp.ValueKind == JsonValueKind.Number)
                            {
                                metrics.CachedTokens = cachedProp.GetInt32();
                            }

                            if (inputDetails.TryGetProperty("text_tokens", out var textTokensProp)
                                && textTokensProp.ValueKind == JsonValueKind.Number)
                            {
                                metrics.TextInputTokens = textTokensProp.GetInt32();
                            }

                            if (inputDetails.TryGetProperty("image_tokens", out var imageTokensProp)
                                && imageTokensProp.ValueKind == JsonValueKind.Number)
                            {
                                metrics.ImageInputTokens = imageTokensProp.GetInt32();
                            }
                        }
                    }

                    // Fill assistant text
                    if (!shouldRetry)
                    {
                        metrics.AssistantText = assistantBuilder.ToString();
                    }

                    break;
                }
                else if (type == "error")
                {
                    Console.WriteLine("Error event from server:");
                    Console.WriteLine(json);

                    if (root.TryGetProperty("error", out var errorObj) &&
                        errorObj.ValueKind == JsonValueKind.Object)
                    {
                        string? code = errorObj.TryGetProperty("code", out var codeProp) && codeProp.ValueKind == JsonValueKind.String
                            ? codeProp.GetString()
                            : null;
                        string? msg = errorObj.TryGetProperty("message", out var msgProp) && msgProp.ValueKind == JsonValueKind.String
                            ? msgProp.GetString()
                            : null;

                        lastErrorMessage = msg ?? code ?? "unknown error";

                        if (code == "rate_limit_exceeded")
                        {
                            shouldRetry = true;
                        }
                    }

                    break;
                }
            }

            // If we should retry, wait and try again
            if (shouldRetry)
            {
                if (attempt == MaxRetries)
                {
                    Console.WriteLine($"Giving up after {MaxRetries} attempts. Last error: {lastErrorMessage}");
                    throw new Exception($"RunSingleSampleAsync failed after {MaxRetries} retries. Last error: {lastErrorMessage}");
                }

                Console.WriteLine($"Rate limit hit, retrying after {RetryDelayMs} ms. Error: {lastErrorMessage}");
                await Task.Delay(RetryDelayMs);
                continue;
            }

            // Compute times (only for non retry case)
            endToEnd.Stop();
            metrics.EndToEndMs = (long)endToEnd.Elapsed.TotalMilliseconds;

            if (tResponseCreated.HasValue)
            {
                metrics.SendToResponseCreatedMs = (long)(tResponseCreated.Value - metrics.ClientSendTimeUtc).TotalMilliseconds;
            }

            if (tResponseCreated.HasValue && tFirstToken.HasValue)
            {
                metrics.ResponseCreatedToFirstTokenMs = (long)(tFirstToken.Value - tResponseCreated.Value).TotalMilliseconds;
            }

            if (tFirstToken.HasValue && tDone.HasValue)
            {
                metrics.FirstTokenToDoneMs = (long)(tDone.Value - tFirstToken.Value).TotalMilliseconds;
            }

            // Cost estimate (simple interpretation)
            int textInput = metrics.TextInputTokens;
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

        // Should never reach here
        throw new Exception("RunSingleSampleWebsocketAsync fell through without returning.");
    }

    public static Task SendJson(ClientWebSocket ws, string json)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        return ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None);
    }

    public static async Task ReceiveAndPrintOneFrame(ClientWebSocket ws)
    {
        var buffer = new byte[16 * 1024];
        var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
        if (result.MessageType == WebSocketMessageType.Text)
        {
            var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
            Console.WriteLine("First event from server:");
            Console.WriteLine(json);

            try
            {
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                if (root.TryGetProperty("type", out var typeProp)
                    && typeProp.GetString() == "error"
                    && root.TryGetProperty("error", out var errorProp))
                {
                    var msg = errorProp.TryGetProperty("message", out var m) ? m.GetString() : null;
                    var code = errorProp.TryGetProperty("code", out var c) ? c.GetString() : null;

                    Console.WriteLine("Error detected");
                    Console.WriteLine("Message: " + msg);
                    Console.WriteLine("Code: " + code);

                    Environment.Exit(1);
                }
            }
            catch (JsonException)
            {
                Console.WriteLine("Invalid JSON received");
                Environment.Exit(1);
            }
        }
    }
}