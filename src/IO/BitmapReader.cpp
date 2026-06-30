#include "IO/BitmapReader.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <stb_image.h>
#include <tinyexr.h>

namespace
{
std::string lowercaseExtension(const std::string& path)
{
    std::string extension = std::filesystem::path(path).extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension;
}
}

Bitmap BitmapReader::read(const std::string& path)
{
    float* rawPixels = nullptr;
    int width = 0;
    int height = 0;

    if (lowercaseExtension(path) == ".exr") {
        const char* error = nullptr;
        const int result = LoadEXR(&rawPixels, &width, &height, path.c_str(), &error);
        if (result != TINYEXR_SUCCESS) {
            const std::string message = error != nullptr ? error : "Unknown TinyEXR error";
            if (error != nullptr)
                FreeEXRErrorMessage(error);
            throw std::runtime_error("Failed to read bitmap '" + path + "': " + message);
        }
    } else {
        int channels = 0;
        rawPixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
        if (rawPixels == nullptr)
            throw std::runtime_error("Failed to read bitmap '" + path + "': " + stbi_failure_reason());
    }

    if (width <= 0 || height <= 0) {
        if (lowercaseExtension(path) == ".exr")
            std::free(rawPixels);
        else
            stbi_image_free(rawPixels);
        throw std::runtime_error("Bitmap has invalid dimensions: " + path);
    }

    std::vector<glm::vec4> pixels(static_cast<size_t>(width) * height);
    std::memcpy(pixels.data(), rawPixels, pixels.size() * sizeof(glm::vec4));
    if (lowercaseExtension(path) == ".exr")
        std::free(rawPixels);
    else
        stbi_image_free(rawPixels);
    return Bitmap(static_cast<uint32_t>(width), static_cast<uint32_t>(height), std::move(pixels));
}
