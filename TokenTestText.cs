//using System.Net.WebSockets;
//using System.Text;
//using System.Text.Json;
//using System.Diagnostics;


//class TokenTestTextOnly
//{
//    static async Task Main()
//    {
//        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

//        if (string.IsNullOrWhiteSpace(apiKey))
//        {
//            Console.WriteLine("OPENAI_API_KEY is not set.");
//            return;
//        }

//        // Debug: show only the first 6 and last 4 characters
//        string visible = apiKey.Length > 10
//            ? apiKey[..6] + "..." + apiKey[^4..]
//            : apiKey;

//        Console.WriteLine("Using API key (partial): " + visible);


//        var uri = new Uri("wss://api.openai.com/v1/realtime?model=gpt-realtime-mini");

//        using var ws = new ClientWebSocket();
//        ws.Options.SetRequestHeader("Authorization", $"Bearer {apiKey}");
//        ws.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");

//        Console.WriteLine("Connecting to OpenAI Realtime...");
//        await ws.ConnectAsync(uri, CancellationToken.None);
//        Console.WriteLine("Connected.");

//        // 1. Wait for session.created
//        await ReceiveOneFrame(ws);

//        // 1b. Force text-only output for this session
//        string sessionUpdateJson = @"
//        {
//          ""type"": ""session.update"",
//          ""session"": {
//            ""modalities"": [""text""]
//          }
//        }";
//        await SendJson(ws, sessionUpdateJson);
//        Console.WriteLine("Sent session.update");


//        // 2. Add a user message
//        string createItemJson = @"
//        {
//          ""type"": ""conversation.item.create"",
//          ""item"": {
//            ""type"": ""message"",
//            ""role"": ""user"",
//            ""content"": [
//              { ""type"": ""input_text"", ""text"": ""Say 'hello from C#' and tell me one fun fact about Delft."" }
//            ]
//          }
//        }";
//        var fullLatency = new Stopwatch();
//        var firstTokenLatency = new Stopwatch();
//        bool firstTokenMeasured = false;

//        fullLatency.Start();
//        firstTokenLatency.Start();
//        await SendJson(ws, createItemJson);
//        Console.WriteLine("Sent conversation.item.create");

//        // 3. Ask the model to respond
//        string responseCreateJson = @"{ ""type"": ""response.create"" }";
//        await SendJson(ws, responseCreateJson);
//        Console.WriteLine("Sent response.create");

//        // 4. Read events
//        var buffer = new byte[16 * 1024];
//        var sb = new StringBuilder();
//        Console.WriteLine("Waiting for text response events...");

//        while (ws.State == WebSocketState.Open)
//        {
//            var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

//            if (result.MessageType == WebSocketMessageType.Close)
//            {
//                Console.WriteLine($"Server closed connection. Status: {result.CloseStatus}, Description: {result.CloseStatusDescription}");
//                break;
//            }

//            var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
//            //Console.WriteLine("Raw event: " + json); // debug

//            using var doc = JsonDocument.Parse(json);
//            var root = doc.RootElement;

//            if (!root.TryGetProperty("type", out var typeProp))
//                continue;

//            var type = typeProp.GetString();

//            if (type == "response.text.delta")
//            {
//                // stop first token latency at the first delta
//                if (!firstTokenMeasured)
//                {
//                    firstTokenLatency.Stop();
//                    firstTokenMeasured = true;
//                    Console.WriteLine($"\nFirst token latency: {firstTokenLatency.ElapsedMilliseconds} ms\n");
//                }
//            }
//            else if (type == "response.done")
//            {
//                fullLatency.Stop();
//                Console.WriteLine($"\nFull latency until response.done: {fullLatency.ElapsedMilliseconds} ms\n");

//                var response = root.GetProperty("response");

//                // 1. Extract and print the assistant text
//                var outputArray = response.GetProperty("output").EnumerateArray();
//                foreach (var item in outputArray)
//                {
//                    var itemType = item.GetProperty("type").GetString();
//                    if (itemType != "message")
//                        continue;

//                    var contentArray = item.GetProperty("content").EnumerateArray();
//                    foreach (var content in contentArray)
//                    {
//                        var contentType = content.GetProperty("type").GetString();
//                        if (contentType == "text")
//                        {
//                            var text = content.GetProperty("text").GetString();
//                            Console.WriteLine("Assistant:");
//                            Console.WriteLine(text);
//                        }
//                    }
//                }

//                // 2. Extract and print token usage
//                var usage = response.GetProperty("usage");
//                int totalTokens = usage.GetProperty("total_tokens").GetInt32();
//                int inputTokens = usage.GetProperty("input_tokens").GetInt32();
//                int outputTokens = usage.GetProperty("output_tokens").GetInt32();

//                // cached tokens (usually zero if no reuse)
//                int cachedTokens = 0;
//                if (usage.TryGetProperty("input_token_details", out var inputDetails) &&
//                    inputDetails.TryGetProperty("cached_tokens", out var cachedProp) &&
//                    cachedProp.ValueKind == JsonValueKind.Number)
//                {
//                    cachedTokens = cachedProp.GetInt32();
//                }

//                Console.WriteLine();
//                Console.WriteLine($"Tokens - total: {totalTokens}, input: {inputTokens}, output: {outputTokens}, cached: {cachedTokens}");

//                // 3. Cost calculation for gpt-realtime-mini
//                const double inputRatePerToken = 0.60 / 1_000_000.0;     // USD per input token
//                const double cachedRatePerToken = 0.06 / 1_000_000.0;    // USD per cached input token
//                const double outputRatePerToken = 2.40 / 1_000_000.0;    // USD per output token

//                double inputCost = inputTokens * inputRatePerToken;
//                double cachedCost = cachedTokens * cachedRatePerToken;
//                double outputCost = outputTokens * outputRatePerToken;
//                double totalCostUsd = inputCost + cachedCost + outputCost;

//                Console.WriteLine($"Cost estimate for this interaction (gpt-realtime-mini):");
//                Console.WriteLine($"  Input cost:  {inputCost:F8} USD");
//                Console.WriteLine($"  Cached cost: {cachedCost:F8} USD");
//                Console.WriteLine($"  Output cost: {outputCost:F8} USD");
//                Console.WriteLine($"  Total cost:  {totalCostUsd:F8} USD");

//                break;
//            }
//            else if (type == "error")
//            {
//                Console.WriteLine("Error event from server:");
//                Console.WriteLine(json);
//                break;
//            }
//        }

//        if (ws.State == WebSocketState.Open)
//        {
//            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
//        }

//        Console.WriteLine("Done.");
//    }

//    private static Task SendJson(ClientWebSocket ws, string json)
//    {
//        var bytes = Encoding.UTF8.GetBytes(json);
//        return ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None);
//    }

//    private static async Task ReceiveOneFrame(ClientWebSocket ws)
//    {
//        var buffer = new byte[16 * 1024];
//        var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
//        if (result.MessageType == WebSocketMessageType.Text)
//        {
//            var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
//            Console.WriteLine("First event from server:");
//            Console.WriteLine(json);
//        }
//    }
//}
