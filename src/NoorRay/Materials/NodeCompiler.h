#pragma once

#include <expected>
#include <string>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

//! Compiles the node tree of a scene object into its graph, program and fixed inputs, and stores them in the object through the scene. Returns the warnings.
class NodeCompiler {
public:
    [[nodiscard]] static std::expected<std::vector<std::string>, Core::Error> compileMaterial(Scene::Scene& scene, Scene::MaterialHandle handle,
                                                                                              bool approximate);
    [[nodiscard]] static std::expected<std::vector<std::string>, Core::Error> compileLight(Scene::Scene& scene, Scene::LightHandle handle, bool approximate);
    [[nodiscard]] static std::expected<std::vector<std::string>, Core::Error> compileEnvironment(Scene::Scene& scene, bool approximate);
};

}
