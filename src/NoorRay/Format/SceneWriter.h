#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Format/AnimationData.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

class SceneWriter {
public:
    [[nodiscard]] static std::expected<void, Core::Error> write(const Scene::Scene& scene, const std::filesystem::path& path);
    [[nodiscard]] static std::expected<void, Core::Error> write(const Scene::Scene& scene, const AnimationData& animation, const std::filesystem::path& path);
};

}
