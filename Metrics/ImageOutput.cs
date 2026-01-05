using System;
using System.IO;

namespace Thesis.Metrics;

public static class ImageOutput
{
    public static void SaveDataUrl(string dataUrl, string outputPath)
    {
        int comma = dataUrl.IndexOf(',');
        if (comma < 0)
            throw new ArgumentException("Invalid data URL");

        string base64 = dataUrl[(comma + 1)..];
        byte[] bytes = Convert.FromBase64String(base64);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, bytes);
    }
}
