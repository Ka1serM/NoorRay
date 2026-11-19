#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "NoorRHI/Core/Format.h"

namespace NoorRay::Scene {

struct TextureLevel {
    std::size_t offset = 0;
    std::size_t size = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

//! The levels describe the base level or a full mip chain in data, in the layout that NoorRHI::Image::upload takes; the GPU generates missing levels.
struct Texture {
    std::string name;
    NoorRHI::ImageFormat format = NoorRHI::ImageFormat::Rgba8Unorm;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<TextureLevel> levels;
    std::vector<std::byte> data;
};

}
