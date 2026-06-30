#include "IO/BitmapWriter.h"

#include "IO/Bitmap.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <stb_image_write.h>
#include <tinyexr.h>

namespace
{
BitmapFormat formatFromPath(const std::string& path)
{
    std::string extension = std::filesystem::path(path).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (extension == ".exr") return BitmapFormat::OpenExr;
    if (extension == ".hdr") return BitmapFormat::RadianceHdr;
    if (extension == ".png") return BitmapFormat::Png;
    return BitmapFormat::Automatic;
}

float linearToSrgb(const float value)
{
    const float clamped = std::max(value, 0.0f);
    return clamped <= 0.0031308f
        ? 12.92f * clamped
        : 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
}

void setError(std::string* destination, std::string message)
{
    if (destination != nullptr)
        *destination = std::move(message);
}
}

bool BitmapWriter::write(
    const std::string& path,
    const Bitmap& bitmap,
    const BitmapWriteOptions& options,
    std::string* errorMessage)
{
    const BitmapFormat format = options.format == BitmapFormat::Automatic
        ? formatFromPath(path) : options.format;

    if (format == BitmapFormat::OpenExr) {
        const char* error = nullptr;
        const int result = SaveEXR(
            bitmap.rgba(), static_cast<int>(bitmap.width()), static_cast<int>(bitmap.height()),
            4, options.exrHalfFloat ? 1 : 0, path.c_str(), &error);
        if (result == TINYEXR_SUCCESS)
            return true;
        setError(errorMessage, error != nullptr ? error : "Unknown TinyEXR error");
        if (error != nullptr)
            FreeEXRErrorMessage(error);
        return false;
    }

    if (format == BitmapFormat::RadianceHdr) {
        if (stbi_write_hdr(path.c_str(), static_cast<int>(bitmap.width()),
                static_cast<int>(bitmap.height()), 4, bitmap.rgba()) != 0)
            return true;
        setError(errorMessage, "Failed to write Radiance HDR image");
        return false;
    }

    if (format == BitmapFormat::Png) {
        const float exposureScale = std::exp2(options.pngExposure);
        std::vector<uint8_t> pixels(bitmap.pixels().size() * 4);
        for (size_t index = 0; index < bitmap.pixels().size(); ++index) {
            const glm::vec4 source = bitmap.pixels()[index];
            const auto toByte = [](const float value) {
                return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
            };
            pixels[index * 4 + 0] = toByte(linearToSrgb(source.r * exposureScale));
            pixels[index * 4 + 1] = toByte(linearToSrgb(source.g * exposureScale));
            pixels[index * 4 + 2] = toByte(linearToSrgb(source.b * exposureScale));
            pixels[index * 4 + 3] = toByte(source.a);
        }
        if (stbi_write_png(path.c_str(), static_cast<int>(bitmap.width()),
                static_cast<int>(bitmap.height()), 4, pixels.data(),
                static_cast<int>(bitmap.width() * 4)) != 0)
            return true;
        setError(errorMessage, "Failed to write PNG image");
        return false;
    }

    setError(errorMessage, "Cannot infer bitmap format from file extension");
    return false;
}
