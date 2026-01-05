namespace Thesis.Dataset;

public class PreparedRow
    {
        public string id { get; set; } = "";
        public string question { get; set; } = "";
        public string answer { get; set; } = "";
        public string image_path { get; set; } = "";
        public List<List<int>>? gaze_points_px { get; set; }
    }