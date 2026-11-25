#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace DiffRenderer::Scenes {

struct NiftiVolume {
    std::array<std::uint32_t, 3> size{};
    //! x fastest, with the scale and offset of the header applied.
    std::vector<float> values;
};

//! Reads the first 3D volume of a gzip-compressed NIfTI-1 file.
[[nodiscard]] std::expected<NiftiVolume, std::string> readNifti(const std::filesystem::path& path);

}
