#pragma once

#include <functional>
#include <optional>

#include "NoorRay/Materials/TextureCache.h"
#include "NoorRay/Scene/Scene.h"

namespace tinygltf {
struct Material;
}

namespace NoorRay::Import {

//! Resolves a glTF texture index to a scene texture in the given colour space; nullopt skips the texture.
using GltfTextureResolver = std::function<std::optional<Scene::MaterialTexture>(int texture, Materials::ColorSpace colorSpace)>;

class GltfMaterial {
public:
    static Scene::Material convert(const tinygltf::Material& material, const GltfTextureResolver& resolver);
};

}
