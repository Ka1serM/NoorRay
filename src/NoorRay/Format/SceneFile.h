#pragma once

#include <expected>
#include <filesystem>
#include <span>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

//! A NoorRay scene file (.nrs): a NoorRayScene flatbuffer that is verified once and then read in place.
class SceneFile {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(const std::filesystem::path& path);
    [[nodiscard]] static std::expected<void, Core::Error> verify(std::span<const std::byte> bytes);
    [[nodiscard]] static std::expected<void, Core::Error> load(std::span<const std::byte> bytes, Scene::Scene& scene);
    [[nodiscard]] std::expected<void, Core::Error> load(Scene::Scene& scene) const { return load(file.getBytes(), scene); }

private:
    Core::MappedFile file;
};

}
