#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Format/Animation.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Chooses the importer by file extension, adds the file's contents to the scene and gives the file's animation.
class SceneImporter {
public:
    [[nodiscard]] static std::expected<Format::Animation, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
    static bool isSupported(const std::filesystem::path& path);
};

}
