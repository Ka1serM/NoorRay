#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

class PbrtImporter {
public:
    [[nodiscard]] static std::expected<void, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
};

}
