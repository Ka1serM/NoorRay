#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Materials/Shading/Material.h"
#include "Scene/Scene.h"

class SceneImporter {
public:
    static void ImportGltfScene(Scene& scene, const std::string& filepath);
    static void ImportObjScene(Scene& scene, const std::string& filepath, const MaterialAuthoring* materialOverride = nullptr);
    static void ImportPbrtScene(Scene& scene, const std::string& filepath);
    static void ImportJsonScene(Scene& scene, const std::string& filepath);
    static void ImportFile(Scene& scene, const std::string& filepath);
    static void ImportGaussianScene(Scene& scene, const std::string& filepath);
    static bool IsGaussianFile(const std::string& filepath);
    static bool IsSceneFile(const std::string& filepath);

    static std::string nameFromPath(const std::string& path);
    static std::vector<char> readFile(const std::string& filename);
};
