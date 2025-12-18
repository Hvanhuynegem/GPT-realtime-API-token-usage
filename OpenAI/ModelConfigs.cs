namespace Thesis.OpenAI;

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