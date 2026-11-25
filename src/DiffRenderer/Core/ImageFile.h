#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "DiffRenderer/Core/Image.h"

namespace DiffRenderer::Core {

struct ImageFile {
    //! The colour as stored in the file, with the Srgb encoding.
    Image image;
    //! The alpha in [0, 1] for each pixel, or empty when the file has no alpha.
    std::vector<float> alpha;
};

//! Reads an 8-bit PNG or JPG.
[[nodiscard]] std::expected<ImageFile, std::string> loadImageFile(const std::filesystem::path& path);
[[nodiscard]] std::expected<void, std::string> savePng(const Image& image, const std::filesystem::path& path);

}
