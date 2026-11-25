#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>

#include "NoorRay/Core/Error.h"

namespace NoorRay::Reconstruction {

struct ColmapProgress {
    std::string stage;
    float fraction = 0.0f;
};

//! The image names in sparse/images.bin are relative to images; the image files can be smaller than the camera size.
struct ColmapDataset {
    std::filesystem::path images;
    std::filesystem::path sparse;
};

//! Reuses photos/sparse/0, then photos/colmap/sparse/0; else runs COLMAP from libNoorRayColmap.so into photos/colmap with a PINHOLE result.
[[nodiscard]] std::expected<ColmapDataset, Core::Error> runColmap(const std::filesystem::path& photos, std::stop_token stop,
                                                                  const std::function<void(const ColmapProgress&)>& progress);

}
