#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Materials/Material.h"
#include "Materials/BasicMaterial.h"
#include "Scene/Scene.h"

class SceneImporter {
public:
    static void ImportGltfScene(Scene& scene, const std::string& filepath);
    static void ImportObjScene(Scene& scene, const std::string& filepath, const BasicMaterial* materialOverride = nullptr);
    static void ImportPbrtScene(Scene& scene, const std::string& filepath);
    static void ImportJsonScene(Scene& scene, const std::string& filepath);
    static void ImportFile(Scene& scene, const std::string& filepath);
    // Replaces the scene's contents: a NoorRay scene file is read, any other
    // supported file imported into the cleared scene.
    static void Load(Scene& scene, const std::string& filepath);
    static void ImportGaussianScene(Scene& scene, const std::string& filepath);
    static bool IsGaussianFile(const std::string& filepath);
    static bool IsSceneFile(const std::string& filepath);

    static std::string nameFromPath(const std::string& path);
    static std::vector<char> readFile(const std::string& filename);
};
