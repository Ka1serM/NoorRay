#pragma once

#include <expected>
#include <filesystem>
#include <span>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Reads 3D Gaussian Splatting PLY files (binary little endian, float properties) into one Gaussian cloud.
class PlyGaussianImporter {
public:
    [[nodiscard]] static std::expected<Scene::GaussianCloudHandle, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
    [[nodiscard]] static std::expected<Scene::GaussianCloud, Core::Error> read(std::span<const std::byte> bytes, std::string name);
    static bool isGaussianPly(std::span<const std::byte> bytes);
};

}
