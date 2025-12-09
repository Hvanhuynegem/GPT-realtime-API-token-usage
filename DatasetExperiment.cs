using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


public enum AIModel
{
    GPT_Realtime_Mini,
    GPT_Realtime,
    GPT5_1,
    GPT5_Mini,
    GPT5_Nano,
    GPT4_1,
    GPT4_1_Mini
}

public class ModelConfig
{
    public string ModelName { get; init; } = "";
    public double TextInputRate { get; init; }
    public double TextOutputRate { get; init; }
    public double ImageInputRate { get; init; }
    public double CachedInputRate { get; init; }

    public string WebSocketUrl {get; init; } = "";
    public bool IsRealtime { get; init; }

}

public static class ModelConfigs
{
    public static ModelConfig Get(AIModel model)
    {
        return model switch
        {
            AIModel.GPT_Realtime_Mini => new ModelConfig
            {
                ModelName = "gpt-realtime-mini",
                TextInputRate = 0.60 / 1_000_000.0,
                TextOutputRate = 2.40 / 1_000_000.0,
                ImageInputRate = 0.80 / 1_000_000.0,
                CachedInputRate = 0.06 / 1_000_000.0,
                WebSocketUrl = "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini",
                IsRealtime = true

            },

            AIModel.GPT_Realtime => new ModelConfig
            {
                ModelName = "gpt-realtime",
                TextInputRate = 4.00 / 1_000_000.0,
                TextOutputRate = 16.00 / 1_000_000.0,
                ImageInputRate = 5.00 / 1_000_000.0,
                CachedInputRate = 0.40 / 1_000_000.0,
                WebSocketUrl = "wss://api.openai.com/v1/realtime?model=gpt-realtime",
                IsRealtime = true
            },

            // Add other models here:
            AIModel.GPT5_1 => new ModelConfig
            {
                ModelName = "gpt-5.1-2025-11-13",
                TextInputRate = 1.25 / 1_000_000.0,
                TextOutputRate = 10 / 1_000_000.0,
                ImageInputRate =  0.00 / 1_000_000.0,
                CachedInputRate = 0.125 / 1_000_000.0,
                WebSocketUrl = "",   // no websocket
                IsRealtime = false
            },

            AIModel.GPT5_Mini => new ModelConfig
            {
                ModelName = "gpt-5-mini-2025-08-07",
                TextInputRate = 0.25 / 1_000_000.0,
                TextOutputRate = 2.00 / 1_000_000.0,
                ImageInputRate =  0.00 / 1_000_000.0,
                CachedInputRate = 0.025 / 1_000_000.0,
                WebSocketUrl = "",   // no websocket
                IsRealtime = false
            },

            AIModel.GPT5_Nano => new ModelConfig
            {
                ModelName = "gpt-5-nano-2025-08-07",
                TextInputRate = 0.05 / 1_000_000.0,
                TextOutputRate = 0.40 / 1_000_000.0,
                ImageInputRate =  0.00 / 1_000_000.0,
                CachedInputRate = 0.005 / 1_000_000.0,
                WebSocketUrl = "",   // no websocket
                IsRealtime = false
            },

            AIModel.GPT4_1 => new ModelConfig
            {
                ModelName = "gpt-4.1-2025-04-14",
                TextInputRate = 2.00 / 1_000_000.0,
                TextOutputRate = 8.00 / 1_000_000.0,
                ImageInputRate =  0.00 / 1_000_000.0,
                CachedInputRate = 0.50 / 1_000_000.0,
                WebSocketUrl = "",   // no websocket
                IsRealtime = false
            },

            AIModel.GPT4_1_Mini => new ModelConfig
            {
                ModelName = "gpt-4.1-mini-2025-04-14",
                TextInputRate = 0.40 / 1_000_000.0,
                TextOutputRate = 1.60 / 1_000_000.0,
                ImageInputRate =  0.00 / 1_000_000.0,
                CachedInputRate = 0.10 / 1_000_000.0,
                WebSocketUrl = "",   // no websocket
                IsRealtime = false
            },

            _ => throw new ArgumentOutOfRangeException(nameof(model), model, null)
        };
    }
}



class DatasetExperiment
{
    static async Task Main(string[] args)
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("OPENAI_API_KEY is not set.");
            return;
        }

        var visible = apiKey.Length > 10
            ? apiKey[..6] + "..." + apiKey[^4..]
            : apiKey;
        Console.WriteLine("Using API key (partial): " + visible);

        int maxSamples = args.Length > 0 && int.TryParse(args[0], out var n) ? n : 10;

        // Load the dataset.
        Console.WriteLine("Current working directory: " + Directory.GetCurrentDirectory());

        //string datasetDir = Path.Combine(Directory.GetCurrentDirectory(), "dataset");
        string datasetDir = "C:\\Users\\henri\\Desktop\\GPT-realtime-API-token-usage\\dataset";
        Console.WriteLine("Dataset dir: " + datasetDir);

        var rawSamples = LoadSamplesFromVoila(datasetDir, maxSamples).ToList();

        IPreprocessor preprocessor = new IdentityPreprocessor();


        // SELECT MODEL HERE ---------------------------------------------------------
        AIModel selected = AIModel.GPT_Realtime_Mini;
        // ---------------------------------------------------------------------------


        ModelConfig cfg = ModelConfigs.Get(selected);

        Console.WriteLine("Using model: " + cfg.ModelName);

        if (cfg.IsRealtime)
        {
            Console.WriteLine("Using Realtime WebSocket endpoint: " + cfg.WebSocketUrl);
            await RunRealtimeExperimentAsync(apiKey, cfg, rawSamples, preprocessor);
        }
        else
        {
            Console.WriteLine("Using REST /v1/responses endpoint");
            await RunRestExperimentAsync(apiKey, cfg, rawSamples, preprocessor);
        }
    }

    static async Task RunRestExperimentAsync(string apiKey, ModelConfig cfg, List<DatasetExperiment.DatasetSample> rawSamples, IPreprocessor preprocessor)
    {
        using var http = new HttpClient
        {
            BaseAddress = new Uri("https://api.openai.com"),
            Timeout = TimeSpan.FromMinutes(10)
        };
        http.DefaultRequestHeaders.Authorization =
            new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);

        var metricsList = new List<SampleMetrics>();
        int index = 0;

        foreach (var sample in rawSamples)
        {
            index++;
            var pre = preprocessor.Preprocess(sample);

            Console.WriteLine();
            Console.WriteLine($"=== Sample {index}/{rawSamples.Count} id={sample.Id} ===");

            Console.WriteLine("Question:");
            Console.WriteLine(sample.Text);
            Console.WriteLine();

            Console.WriteLine("Baseline answer (dataset):");
            Console.WriteLine(sample.Answer);
            Console.WriteLine();

            var metrics = await RunSingleSampleRestAsync(http, pre, cfg);
            metrics.SampleId = sample.Id;
            metricsList.Add(metrics);

            PrintMetrics(metrics);
        }

        string baseDir = "logs";
        string timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd_HH-mm-ss");
        string experimentId = $"{timestamp}_{cfg.ModelName}_baseline_rest";

        SaveExperimentResults(baseDir, experimentId,
            model: cfg.ModelName,
            datasetName: "VOIL-A",
            preprocessorName: preprocessor.GetType().Name,
            metrics: metricsList);
    }


    static async Task RunRealtimeExperimentAsync(string apiKey, ModelConfig cfg, List<DatasetExperiment.DatasetSample> rawSamples, IPreprocessor preprocessor)
    {
        var metricsList = new List<SampleMetrics>();

        int index = 0;
        foreach (var sample in rawSamples)
        {
            using var ws = new ClientWebSocket();
            ws.Options.SetRequestHeader("Authorization", $"Bearer {apiKey}");
            ws.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");
            Console.WriteLine("Connecting to OpenAI Realtime...");
            await ws.ConnectAsync(new Uri(cfg.WebSocketUrl), CancellationToken.None);
            Console.WriteLine("Connected.");

            // First event should be session.created
            await ReceiveAndPrintOneFrame(ws);

            // Limit output to text only
            //string sessionUpdateJson = @"
            //{
            //  ""type"": ""session.update"",
            //  ""session"": {
            //    ""modalities"": [""text""]
            //  }
            //}";
            string sessionUpdateJson = @"
            {
            ""type"": ""session.update"",
            ""session"": {
                ""modalities"": [""text""],
                ""instructions"": ""You are a helpful assistant. Answer in one short, direct sentence with no extra details.""
            }
            }";
            await SendJson(ws, sessionUpdateJson);

            await SendJson(ws, sessionUpdateJson);
            Console.WriteLine("Sent session.update");


            index++;
            var pre = preprocessor.Preprocess(sample);

            Console.WriteLine();
            Console.WriteLine($"=== Sample {index}/{rawSamples.Count} id={sample.Id} ===");

            Console.WriteLine("Question:");
            Console.WriteLine(sample.Text);
            Console.WriteLine();

            Console.WriteLine("Baseline answer (dataset):");
            Console.WriteLine(sample.Answer);
            Console.WriteLine();

            var metrics = await RunSingleSampleAsync(ws, pre, cfg);
            metrics.SampleId = sample.Id;
            metricsList.Add(metrics);

            PrintMetrics(metrics);

            if (ws.State == WebSocketState.Open)
            {
                await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
            }
        }

        string baseDir = "logs";
        string timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd_HH-mm-ss");
        string experimentId = $"{timestamp}_gpt-realtime-mini_baseline";

        SaveExperimentResults(baseDir, experimentId, model: cfg.ModelName,
            datasetName: "VOIL-A", preprocessorName: preprocessor.GetType().Name,
            metrics: metricsList);

        Console.WriteLine("Done.");
    }

    static async Task<SampleMetrics> RunSingleSampleRestAsync(HttpClient http, PreprocessedSample pre, ModelConfig cfg)
    {
        var metrics = new SampleMetrics();

        // Build the "content" array in the same way as for realtime
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

        // Build the /v1/responses request body
        var body = new
        {
            model = cfg.ModelName,   // for example "gpt-5.1"
            input = new[]
            {
                new
                {
                    role = "user",
                    content = contentParts
                }
            },
            // // Optional: keep responses short to mirror your realtime settings
            // max_output_tokens = 256
        };

        string jsonBody = JsonSerializer.Serialize(body);
        using var requestContent = new StringContent(jsonBody, Encoding.UTF8, "application/json");

        var sendTime = DateTime.UtcNow;
        metrics.ClientSendTimeUtc = sendTime;

        using var response = await http.PostAsync("/v1/responses", requestContent);
        var responseText = await response.Content.ReadAsStringAsync();
        var doneTime = DateTime.UtcNow;
        // Console.WriteLine("REST response received: " + responseText);

        if (!response.IsSuccessStatusCode)
        {
            Console.WriteLine("REST error from server:");
            Console.WriteLine("Status: " + (int)response.StatusCode + " " + response.StatusCode);
            Console.WriteLine(responseText);

            // Set timing and return with empty usage
            metrics.EndToEndMs = (long)(doneTime - sendTime).TotalMilliseconds;
            return metrics;
        }

        using var doc = JsonDocument.Parse(responseText);
        var root = doc.RootElement;

        // Extract assistant text from all output items of type "message"
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

        // Extract usage
        if (root.TryGetProperty("usage", out var usage))
        // Console.WriteLine("REST usage: " + usage.ToString());
        {
            if (usage.TryGetProperty("total_tokens", out var totalTokensProp) &&
                totalTokensProp.ValueKind == JsonValueKind.Number)
            {
                metrics.TotalTokens = totalTokensProp.GetInt32();
            }

            if (usage.TryGetProperty("input_tokens", out var inputTokensProp) &&
                inputTokensProp.ValueKind == JsonValueKind.Number)
            {
                metrics.InputTokens = inputTokensProp.GetInt32();
                metrics.TextInputTokens = metrics.InputTokens;
            }

            if (usage.TryGetProperty("output_tokens", out var outputTokensProp) &&
                outputTokensProp.ValueKind == JsonValueKind.Number)
            {
                metrics.OutputTokens = outputTokensProp.GetInt32();
            }

            if (usage.TryGetProperty("input_token_details", out var inputDetails) &&
                inputDetails.ValueKind == JsonValueKind.Object)
            {
                if (inputDetails.TryGetProperty("cached_tokens", out var cachedProp) &&
                    cachedProp.ValueKind == JsonValueKind.Number)
                {
                    metrics.CachedTokens = cachedProp.GetInt32();
                }

                if (inputDetails.TryGetProperty("text_tokens", out var textTokensProp) &&
                    textTokensProp.ValueKind == JsonValueKind.Number)
                {
                    metrics.TextInputTokens = textTokensProp.GetInt32();
                }

                if (inputDetails.TryGetProperty("image_tokens", out var imageTokensProp) &&
                    imageTokensProp.ValueKind == JsonValueKind.Number)
                {
                    metrics.ImageInputTokens = imageTokensProp.GetInt32();
                }
            }
        }

        // Timing metrics
        metrics.EndToEndMs = (long)(doneTime - sendTime).TotalMilliseconds;

        // For REST we do not have internal events, so keep the breakdown simple
        metrics.SendToResponseCreatedMs = metrics.EndToEndMs;
        metrics.ResponseCreatedToFirstTokenMs = 0;
        metrics.FirstTokenToDoneMs = 0;

        // Cost estimate, same logic as realtime
        int textInput = metrics.InputTokens;
        int imageInput = metrics.ImageInputTokens;
        int cached = metrics.CachedTokens;
        int outputTokens = metrics.OutputTokens;

        double textInputCost = textInput * cfg.TextInputRate;
        double imageInputCost = imageInput * cfg.ImageInputRate;
        double cachedCost = cached * cfg.CachedInputRate;
        double outputCost = outputTokens * cfg.TextOutputRate;

        metrics.TotalCostUsd = textInputCost + imageInputCost + cachedCost + outputCost;

        return metrics;
    }

    private static async Task<ClientWebSocket> CreateRealtimeWebSocketAsync(string apiKey, Uri uri)
    {
        var ws = new ClientWebSocket();
        ws.Options.SetRequestHeader("Authorization", $"Bearer {apiKey}");
        ws.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");
        await ws.ConnectAsync(uri, CancellationToken.None);
        return ws;
    }



    private static async Task<SampleMetrics> RunSingleSampleAsync(
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
        throw new Exception("RunSingleSampleAsync fell through without returning.");
    }


    private static void PrintMetrics(SampleMetrics m)
    {
        Console.WriteLine("Assistant output:");
        Console.WriteLine(m.AssistantText);
        Console.WriteLine();

        Console.WriteLine($"End to end latency (send -> response.done): {m.EndToEndMs} ms");
        Console.WriteLine($"Approx client -> server + initial queue (send -> response.created): {m.SendToResponseCreatedMs} ms");
        Console.WriteLine($"Approx server processing start (response.created -> first token): {m.ResponseCreatedToFirstTokenMs} ms");
        Console.WriteLine($"Approx generation + streaming (first token -> done): {m.FirstTokenToDoneMs} ms");
        Console.WriteLine();

        Console.WriteLine($"Tokens - total: {m.TotalTokens}, input: {m.InputTokens}, output: {m.OutputTokens}");
        Console.WriteLine($"  text input tokens:  {m.TextInputTokens}");
        Console.WriteLine($"  image input tokens: {m.ImageInputTokens}");
        Console.WriteLine($"  cached tokens:      {m.CachedTokens}");
        Console.WriteLine($"  text output tokens: {m.OutputTokens}");
        Console.WriteLine();

        Console.WriteLine("Cost estimate for this interaction (gpt-realtime-mini):");
        Console.WriteLine($"  Total cost: {m.TotalCostUsd:F8} USD");
    }

    private static Task SendJson(ClientWebSocket ws, string json)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        return ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None);
    }

    private static async Task ReceiveAndPrintOneFrame(ClientWebSocket ws)
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

    private static IEnumerable<DatasetSample> LoadSamplesFromVoila(string datasetDir, int? maxSamples = null)
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
    
    // Convert an image file to a data URL suitable for input_image.image_url
    public static string? ImageFileToDataUrl(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;

        var bytes = File.ReadAllBytes(path);
        string base64 = Convert.ToBase64String(bytes);

        string ext = Path.GetExtension(path).ToLowerInvariant();
        string format = ext switch
        {
            ".png" => "png",
            ".jpg" => "jpeg",
            ".jpeg" => "jpeg",
            ".webp" => "webp",
            _ => "png"
        };

        return $"data:image/{format};base64,{base64}";
    }

    // Data structures

    public class DatasetSample
    {
        public string Id { get; set; } = "";
        public string Text { get; set; } = "";
        public string Answer { get; set; } = "";
        public string? ImagePath { get; set; }
    }

    public class PreparedRow
    {
        public string id { get; set; } = "";
        public string question { get; set; } = "";
        public string answer { get; set; } = "";
        public string image_path { get; set; } = "";
    }


    public class VoilaAnnoItem
    {
        [JsonPropertyName("question")]
        public string Question { get; set; } = "";

        [JsonPropertyName("answer")]
        public string Answer { get; set; } = "";

        [JsonPropertyName("image_ids")]
        public List<String> ImageIds { get; set; } = new();
    }

    private static T LoadJsonFile<T>(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<T>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? throw new InvalidOperationException($"Failed to deserialize {path}");
    }

    // VOIL-A uses urlsafe base64, like Python's urlsafe_b64decode
    private static byte[] UrlSafeBase64ToBytes(string urlSafe)
    {
        string s = urlSafe.Replace('-', '+').Replace('_', '/');
        int pad = s.Length % 4;
        if (pad > 0)
        {
            s = s + new string('=', 4 - pad);
        }
        return Convert.FromBase64String(s);
    }


    public class PreprocessedSample
    {
        public string Text { get; set; } = "";
        public string? ImageDataUrl { get; set; }
    }

    public interface IPreprocessor
    {
        PreprocessedSample Preprocess(DatasetSample sample);
    }

    // Simple identity preprocessor: no text change, just convert image to data URL.
    public class IdentityPreprocessor : IPreprocessor
    {
        public PreprocessedSample Preprocess(DatasetSample sample)
        {
            return new PreprocessedSample
            {
                Text = sample.Text,
                ImageDataUrl = DatasetExperiment.ImageFileToDataUrl(sample.ImagePath ?? "")
            };
        }
    }

    public class SampleMetrics
    {
        public string SampleId { get; set; } = "";

        // Timing
        public DateTime ClientSendTimeUtc { get; set; }
        public long EndToEndMs { get; set; }
        public long SendToResponseCreatedMs { get; set; }
        public long ResponseCreatedToFirstTokenMs { get; set; }
        public long FirstTokenToDoneMs { get; set; }

        // Tokens
        public int TotalTokens { get; set; }
        public int InputTokens { get; set; }
        public int OutputTokens { get; set; }
        public int CachedTokens { get; set; }
        public int TextInputTokens { get; set; }
        public int ImageInputTokens { get; set; }

        // Cost
        public double TotalCostUsd { get; set; }

        // Result text
        public string AssistantText { get; set; } = "";
    }

    static void SaveExperimentResults(
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
