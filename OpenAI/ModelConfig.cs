namespace Thesis.OpenAI;

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