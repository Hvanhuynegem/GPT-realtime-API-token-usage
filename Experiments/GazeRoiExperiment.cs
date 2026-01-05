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

        if (processed.RoiImageDataUrl != null)
        {
            string roiPath = Path.Combine(
                outDir,
                $"GazeRoiExperiment_{timestamp}_roi.jpg"
            );

            ImageOutput.SaveDataUrl(processed.RoiImageDataUrl, roiPath);
        }

        if (processed.GlobalThumbnailDataUrl != null)
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

        DatasetSample sample = VoilaDatasetLoader
            .LoadSamplesFromVoila(datasetDir, maxSamples: 1)
            .First();

        return sample;
    }

}
