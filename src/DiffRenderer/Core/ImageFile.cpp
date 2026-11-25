#include "DiffRenderer/Core/ImageFile.h"

#include <cstdint>
#include <format>
#include <memory>

//! Static, so a program that links another copy of stb, as NoorRay does, has no duplicate symbols.
#define STB_IMAGE_STATIC
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#include "stb_image_write.h"

namespace DiffRenderer::Core {

namespace {

struct StbFree {
    void operator()(stbi_uc* data) const { stbi_image_free(data); }
};

}

std::expected<ImageFile, std::string> loadImageFile(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    const std::unique_ptr<stbi_uc, StbFree> data(stbi_load(path.c_str(), &width, &height, &channels, 0));
    if (!data)
        return std::unexpected(std::format("{} does not load: {}", path.string(), stbi_failure_reason()));
    ImageFile file{Image(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)), {}};
    file.image.setEncoding(Image::Encoding::Srgb);
    const bool hasAlpha = channels == 2 || channels == 4;
    const int colorChannels = hasAlpha ? channels - 1 : channels;
    const std::size_t pixelCount = std::size_t{file.image.getWidth()} * file.image.getHeight();
    if (hasAlpha)
        file.alpha.resize(pixelCount);
    const std::span<float> pixels = file.image.getPixels();
    constexpr float scale = 1.0f / 255.0f;
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        const stbi_uc* source = data.get() + pixel * channels;
        for (int channel = 0; channel < 3; ++channel)
            pixels[pixel * 3 + channel] = source[colorChannels == 3 ? channel : 0] * scale;
        if (hasAlpha)
            file.alpha[pixel] = source[colorChannels] * scale;
    }
    return file;
}

std::expected<void, std::string> savePng(const Image& image, const std::filesystem::path& path) {
    std::vector<std::uint32_t> rgba(std::size_t{image.getWidth()} * image.getHeight());
    toRgba8(image, rgba);
    const int width = static_cast<int>(image.getWidth());
    if (!stbi_write_png(path.c_str(), width, static_cast<int>(image.getHeight()), 4, rgba.data(), width * 4))
        return std::unexpected(std::format("{} does not save", path.string()));
    return {};
}

}
