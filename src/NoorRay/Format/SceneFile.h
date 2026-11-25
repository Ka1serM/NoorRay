#pragma once

#include <expected>
#include <filesystem>
#include <memory>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Format/Animation.h"
#include "NoorRay/Format/SceneCodec.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Format {

//! A NoorRay scene file (.nrs), memory-mapped; SceneCodec.h gives its layout.
class SceneFile {
public:
    [[nodiscard]] std::expected<void, Core::Error> open(const std::filesystem::path& path);
    //! Adds the file contents to the scene and gives the animation of the file, which is empty for a still scene.
    [[nodiscard]] std::expected<Animation, Core::Error> load(Scene::Scene& scene) const;

private:
    //! Shared with the animation, which reads key frames from it.
    std::shared_ptr<Core::MappedFile> file;
    FileHeader header;
};

}
