using Thesis.Dataset;
using Thesis.Preprocessing;
using Thesis.OpenAI;

namespace Thesis.Experiments
{
    /// <summary>
    /// High-level experiment orchestrator.
    /// Loads the dataset once and runs DatasetExperiment
    /// for each preprocessing technique on the same samples.
    /// </summary>
    public static class ExperimentPlan
    {
        public static async Task RunAsync()
        {
            // -------------------------
            // Global experiment config
            // -------------------------
            int maxSamples = 5; // Set to -1 to use all samples
            AIModel model = AIModel.GPT_Realtime_Mini;

            // -------------------------
            // Dataset loading (ONCE)
            // -------------------------
            // string dataDir = Path.Combine(Directory.GetCurrentDirectory(), "data", "voila-data");
            string dataDir = Path.Combine(Directory.GetCurrentDirectory(), "data", "VQA-HMUG-data");
            Console.WriteLine("Dataset dir: " + dataDir);

            var rawSamples = VoilaDatasetLoader
                .LoadSamplesFromVQAMHUG(dataDir, maxSamples)
                .ToList();

            Console.WriteLine($"Loaded {rawSamples.Count} samples.");

            // -------------------------
            // Preprocessor plan
            // -------------------------
            var preprocessors = new List<PreprocessorEntry>
            {
                new("Baseline_NoOp", () => new IdentityPreprocessor()),
                new("Downsampler", () => new DownsamplerPreprocessor(targetWidth: 100, targetHeight: 100)),
                new("Grayscale", () => new GrayscalePreprocessor()),

                // Format conversion preprocessors
                new("Jpeg_Q85", () => new JpegPreprocessor(quality: 85, useOriginalBytes: false)),
                new("WebP_Lossy_Q85", () => new WebPPreprocessor(quality: 85, lossless: false)),
                // new("Bmp", () => new BmpPreprocessor()),

                // Gaze ROI + global thumbnail
                new("GazeRoi+GlobalThumb", () =>
                    new GazeRoiAndThumbnailPreprocessor(
                        rho: 0.5f,
                        minRoiRelSize: 0.2f,
                        gaussianSigma: 15f,
                        globalThumbW: 28,
                        globalThumbH: 28,
                        jpegQuality: 85)),

                // Saliency-based ROI + global thumbnail
                new("SalientRoi+GlobalThumb", () =>
                    new SpectralResidualSalientPreprocessor(
                        avgFilterSize: 3,
                        gaussianSigma: 8f,
                        roiThresholdMul: 3f,
                        minRoiRelSize: 0.2f,
                        globalThumbW: 28,
                        globalThumbH: 28,
                        jpegQuality: 85)),

                // YOLOv12 ROI + global thumbnail
                new("YoloV12SalientRoi+GlobalThumb", () =>
                    new YoloPreprocessor(
                        onnxRelativePath: @"Preprocessing\Yolov12\yolo12n.onnx",
                        inputSize: 640,
                        confThreshold: 0.25f,
                        iouThreshold: 0.45f,
                        thumbW: 28,
                        thumbH: 28,
                        jpegQuality: 85)),
            };

            // -------------------------
            // Run experiments
            // -------------------------
            string planRunId = DateTime.UtcNow.ToString("yyyy-MM-dd_HH-mm-ss") + "_" + model;

            foreach (var entry in preprocessors)
            {
                Console.WriteLine();
                Console.WriteLine("========================================");
                Console.WriteLine($"Running preprocessor: {entry.Name}");
                Console.WriteLine("========================================");

                var preprocessor = entry.Factory();

                await DatasetExperiment.RunAsync(
                    rawSamples,
                    preprocessor,
                    entry.Name,
                    model,
                    planRunId
                );
            }

            Console.WriteLine("All experiments completed.");
        }

        // Small helper record to keep plan readable
        private sealed record PreprocessorEntry(
            string Name,
            Func<IPreprocessor> Factory
        );
    }
}
