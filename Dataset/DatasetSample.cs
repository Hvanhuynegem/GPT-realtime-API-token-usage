namespace Thesis.Dataset;

public class DatasetSample
    {
        public string Id { get; set; } = "";
        public string Text { get; set; } = "";
        public string Answer { get; set; } = "";
        public string? ImagePath { get; set; }

        // gaze points in pixel coordinates (image space)
        public IReadOnlyList<(int x, int y)> Trace { get; set; } = Array.Empty<(int, int)>();
    }