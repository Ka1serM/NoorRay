#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

class SceneWriter {
public:
    static std::vector<std::byte> write(const Scene::Scene& scene);
    [[nodiscard]] static std::expected<void, Core::Error> write(const Scene::Scene& scene, const std::filesystem::path& path);
};

}
