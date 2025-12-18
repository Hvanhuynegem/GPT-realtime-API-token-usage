namespace Thesis.Metrics;

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