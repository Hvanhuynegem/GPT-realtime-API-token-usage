namespace Thesis.Dataset;

public static class ImageDataUrl
{
    // Convert an image file to a data URL suitable for input_image.image_url
    public static string? ImageFileToDataUrl(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;

        var bytes = File.ReadAllBytes(path);
        string base64 = Convert.ToBase64String(bytes);

        string ext = Path.GetExtension(path).ToLowerInvariant();
        string format = ext switch
        {
            ".png" => "png",
            ".jpg" => "jpeg",
            ".jpeg" => "jpeg",
            ".webp" => "webp",
            _ => "png"
        };

        return $"data:image/{format};base64,{base64}";
    }
}
