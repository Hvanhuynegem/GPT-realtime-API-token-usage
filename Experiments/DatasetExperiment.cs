using System.Net.WebSockets;
using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.Metrics;
using Thesis.OpenAI;
using System.Text.Json;


class DatasetExperiment
{
    public static async Task RunAsync(
    List<DatasetSample> rawSamples,
    IPreprocessor preprocessor,
    string preprocessorName,
    AIModel selectedModel,
    string planRunId
    )
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("OPENAI_API_KEY is not set.");
            return;
        }

        ModelConfig cfg = ModelConfigs.Get(selectedModel);

        Console.WriteLine($"Model: {cfg.ModelName}");
        Console.WriteLine($"Preprocessor: {preprocessorName}");

        if (cfg.IsRealtime)
        {
            await RunRealtimeExperimentAsync(
                apiKey,
                cfg,
                rawSamples,
                preprocessor,
                preprocessorName,
                planRunId
            );
        }
        else
        {
            await RunRestExperimentAsync(
                apiKey,
                cfg,
                rawSamples,
                preprocessor,
                preprocessorName,
                planRunId
            );
        }
    }

    static async Task RunRestExperimentAsync(
        string apiKey, 
        ModelConfig cfg, 
        List<DatasetSample> rawSamples, 
        IPreprocessor preprocessor, 
        string preprocessorName, 
        string planRunId)
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

            var metrics = await RestResponseRunner.RunSingleSampleRestAsync(http, pre, cfg);
            metrics.SampleId = sample.Id;
            metricsList.Add(metrics);

            PrintMetrics(metrics);
        }

        string baseDir = "logs";
        string experimentId = planRunId; // same for all preprocessors


        ExperimentLogger.SaveExperimentResults(
            baseDir, 
            experimentId,
            model: cfg.ModelName,
            datasetName: "VOIL-A",
            preprocessorName: preprocessorName,
            metrics: metricsList);
    }


    static async Task RunRealtimeExperimentAsync(
        string apiKey, 
        ModelConfig cfg, 
        List<DatasetSample> rawSamples, 
        IPreprocessor preprocessor, 
        string preprocessorName,
        string planRunId)
    {
        var metricsList = new List<SampleMetrics>();

        int index = 0;
        foreach (var sample in rawSamples)
        {
            Console.WriteLine("Connecting to OpenAI Realtime...");
            using var ws = await ConnectRealtimeWithRetryAsync(
                apiKey,
                cfg.WebSocketUrl,
                maxAttempts: 6,
                baseDelayMs: 250,
                maxDelayMs: 8000,
                ct: CancellationToken.None
            );
            Console.WriteLine("Connected.");


            // string sessionUpdateJson = @"
            // {
            // ""type"": ""session.update"",
            // ""session"": {
            //     ""modalities"": [""text""],
            //     ""instructions"": ""Answer like VQA: output only the final answer as a short phrase (1 to 3 words, max 4). No full sentence. No explanation. No extra details. No leading articles (a, an, the). No punctuation. Use lowercase. If the answer is a number, output digits. Examples: writing | holding bag | wood | yes | 2.""
            // }
            // }";

            string sessionUpdateJson = @"
            {
            ""type"": ""session.update"",
            ""session"": {
                ""modalities"": [""text""],
                ""instructions"": ""You are answering a VQA benchmark. Output only the canonical short answer that humans would write.\n\nRules:\n- Output 1 to 2 words (max 3).\n- No full sentence, no extra objects, no prepositions like 'on', 'in', 'with'.\n- For action questions ('What is X doing?'), output only the main verb (e.g., 'writing', 'eating', 'standing').\n- No articles (a/an/the), no punctuation.\n- Lowercase.\n- Numbers as digits.\n- If unsure, choose the simplest, most common answer.\n\nExamples:\nQ: what is the man doing? A: writing\nQ: what material is the counter made of? A: wood\nQ: does the laptop have a cable? A: yes\nQ: how many people have a camera? A: 1""
            }
            }";

            await RealtimeRunner.SendJson(ws, sessionUpdateJson);
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

            var metrics = await RealtimeRunner.RunSingleSampleRealtimeAsync(ws, pre, cfg);
            metrics.SampleId = sample.Id;
            metricsList.Add(metrics);

            PrintMetrics(metrics);

            if (ws.State == WebSocketState.Open)
            {
                await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
            }
        }

        string baseDir = "logs";
        string experimentId = planRunId; // same for all preprocessors


        ExperimentLogger.SaveExperimentResults(baseDir, experimentId, model: cfg.ModelName,
            datasetName: "VOIL-A", 
            preprocessorName: preprocessorName,
            metrics: metricsList);

        Console.WriteLine("Done.");
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


    static async Task<ClientWebSocket> ConnectRealtimeWithRetryAsync(
    string apiKey,
    string webSocketUrl,
    int maxAttempts = 6,
    int baseDelayMs = 250,
    int maxDelayMs = 8000,
    CancellationToken ct = default)
    {
        var rng = new Random();

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            ClientWebSocket? ws = null;
            try
            {
                ws = new ClientWebSocket();
                ws.Options.SetRequestHeader("Authorization", $"Bearer {apiKey}");
                ws.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");

                await ws.ConnectAsync(new Uri(webSocketUrl), ct);

                // Optional: validate first frame is not an error
                // If your ReceiveAndPrintOneFrame already prints and returns the raw JSON,
                // use that. Otherwise use a "peek" receive method.
                string first = await RealtimeRunner.ReceiveOneFrameAsString(ws, ct);

                if (IsServerErrorEvent(first))
                {
                    // Clean close and retry
                    try
                    {
                        if (ws.State == WebSocketState.Open)
                            await ws.CloseAsync(WebSocketCloseStatus.InternalServerError, "Retry after server_error", ct);
                    }
                    catch { /* ignore */ }

                    ws.Dispose();

                    throw new Exception("Server returned error on first frame.");
                }

                // If you still want to print it:
                Console.WriteLine("First event from server:");
                Console.WriteLine(first);

                return ws;
            }
            catch (Exception ex) when (attempt < maxAttempts && IsRetryableRealtimeException(ex))
            {
                try { ws?.Dispose(); } catch { /* ignore */ }

                int delay = ComputeBackoffWithJitterMs(attempt, baseDelayMs, maxDelayMs, rng);
                Console.WriteLine($"Realtime connect attempt {attempt} failed: {ex.Message}");
                Console.WriteLine($"Retrying in {delay} ms...");
                await Task.Delay(delay, ct);
            }
        }

        throw new Exception($"Failed to connect to Realtime after {maxAttempts} attempts.");
    }

    static bool IsRetryableRealtimeException(Exception ex)
    {
        // Conservative: retry on websocket/network/transient issues.
        if (ex is WebSocketException) return true;
        if (ex is OperationCanceledException) return false; // user cancelled
        if (ex is TimeoutException) return true;

        // Your first-frame "server_error" throws generic Exception above
        if (ex.Message.Contains("Server returned error", StringComparison.OrdinalIgnoreCase)) return true;

        return false;
    }

    static int ComputeBackoffWithJitterMs(int attempt, int baseDelayMs, int maxDelayMs, Random rng)
    {
        // Exponential backoff: base * 2^(attempt-1), capped, plus jitter [0..base)
        double exp = baseDelayMs * Math.Pow(2, attempt - 1);
        int capped = (int)Math.Min(exp, maxDelayMs);
        int jitter = rng.Next(0, baseDelayMs);
        return capped + jitter;
    }

    static bool IsServerErrorEvent(string json)
    {
        // Detect: {"type":"error","error":{"type":"server_error", ...}}
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (!root.TryGetProperty("type", out var typeProp)) return false;
            if (!string.Equals(typeProp.GetString(), "error", StringComparison.OrdinalIgnoreCase)) return false;

            if (!root.TryGetProperty("error", out var errObj)) return false;
            if (!errObj.TryGetProperty("type", out var errType)) return false;

            var t = errType.GetString() ?? "";
            return string.Equals(t, "server_error", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }
}
