using System.Net.WebSockets;
using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.Metrics;
using Thesis.OpenAI;

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

    // public static async Task RunAsync()
    // {
    //     var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
    //     if (string.IsNullOrWhiteSpace(apiKey))
    //     {
    //         Console.WriteLine("OPENAI_API_KEY is not set.");
    //         return;
    //     }

    //     var visible = apiKey.Length > 10
    //         ? apiKey[..6] + "..." + apiKey[^4..]
    //         : apiKey;
    //     Console.WriteLine("Using API key (partial): " + visible);

    //     int maxSamples = 1;

    //     // Load the dataset.
    //     Console.WriteLine("Current working directory: " + Directory.GetCurrentDirectory());

    //     string dataDir = Path.Combine(Directory.GetCurrentDirectory(), "data");
    //     // string datasetDir = "C:\\Users\\henri\\Desktop\\GPT-realtime-API-token-usage\\dataset";
    //     Console.WriteLine("Dataset dir: " + dataDir);

    //     var rawSamples = VoilaDatasetLoader.LoadSamplesFromVoila(dataDir, maxSamples).ToList();
    //     IPreprocessor preprocessor = new IdentityPreprocessor();


    //     // SELECT MODEL HERE ---------------------------------------------------------
    //     AIModel selected = AIModel.GPT_Realtime_Mini;
    //     // ---------------------------------------------------------------------------


    //     ModelConfig cfg = ModelConfigs.Get(selected);

    //     Console.WriteLine("Using model: " + cfg.ModelName);

    //     if (cfg.IsRealtime)
    //     {
    //         Console.WriteLine("Using Realtime WebSocket endpoint: " + cfg.WebSocketUrl);
    //         await RunRealtimeExperimentAsync(apiKey, cfg, rawSamples, preprocessor);
    //     }
    //     else
    //     {
    //         Console.WriteLine("Using REST /v1/responses endpoint");
    //         await RunRestExperimentAsync(apiKey, cfg, rawSamples, preprocessor);
    //     }
    // }

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
            using var ws = new ClientWebSocket();
            ws.Options.SetRequestHeader("Authorization", $"Bearer {apiKey}");
            ws.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");
            Console.WriteLine("Connecting to OpenAI Realtime...");
            await ws.ConnectAsync(new Uri(cfg.WebSocketUrl), CancellationToken.None);
            Console.WriteLine("Connected.");

            // First event should be session.created
            await RealtimeRunner.ReceiveAndPrintOneFrame(ws);

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

}
