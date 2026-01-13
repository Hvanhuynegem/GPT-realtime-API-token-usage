using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.Metrics;

class GazeRoiExperiment
{
    public static void Run()
    {
        IPreprocessor preprocessor = new GazeRoiAndThumbnailPreprocessor();

        DatasetSample sample = LoadSample();
        PreprocessedSample processed = preprocessor.Preprocess(sample);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string outDir = Path.Combine("outputs");
        Directory.CreateDirectory(outDir);

        // Save ROI crops (0..N)
        if (processed.Rois != null && processed.Rois.Count > 0)
        {
            foreach (var roi in processed.Rois)
            {
                if (string.IsNullOrWhiteSpace(roi.ImageDataUrl))
                    continue;

                string roiPath = Path.Combine(
                    outDir,
                    $"GazeRoiExperiment_{timestamp}_roi_{roi.Index}.jpg"
                );

                ImageOutput.SaveDataUrl(roi.ImageDataUrl, roiPath);
            }
        }

        // Save global thumbnail
        if (!string.IsNullOrWhiteSpace(processed.GlobalThumbnailDataUrl))
        {
            string thumbPath = Path.Combine(
                outDir,
                $"GazeRoiExperiment_{timestamp}_global.jpg"
            );

            ImageOutput.SaveDataUrl(processed.GlobalThumbnailDataUrl, thumbPath);
        }
    }

    private static DatasetSample LoadSample()
    {
        string datasetDir = "data"; // root containing prepared.jsonl

        return VoilaDatasetLoader
            .LoadSamplesFromVoila(datasetDir, maxSamples: 1)
            .First();
    }
}
