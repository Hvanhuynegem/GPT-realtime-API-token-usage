using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;

namespace Thesis.Preprocessing;

public static class ImageEncoding
{
    public static string EncodeFileAsDataUrl(string imagePath, string mime, IImageEncoder encoder)
    {
        if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            return "";

        using Image image = Image.Load(imagePath);

        using var ms = new MemoryStream();
        image.Save(ms, encoder);

        string b64 = Convert.ToBase64String(ms.ToArray());
        return $"data:{mime};base64,{b64}";
    }

    public static string ReadFileAsDataUrl(string imagePath, string mime)
    {
        if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            return "";

        byte[] bytes = File.ReadAllBytes(imagePath);
        string b64 = Convert.ToBase64String(bytes);
        return $"data:{mime};base64,{b64}";
    }

    public static string EncodeImageAsDataUrl(Image image, string mime, IImageEncoder encoder)
    {
        using var ms = new MemoryStream();
        image.Save(ms, encoder);

        string b64 = Convert.ToBase64String(ms.ToArray());
        return $"data:{mime};base64,{b64}";
    }
}
