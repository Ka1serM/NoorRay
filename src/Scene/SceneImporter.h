#pragma once

#include <vector>
#include <string>

#include "Scene/Scene.h"
class SceneImporter {
public:
    static void ImportGltfScene(Scene& scene, const std::string& filepath);
    static void ImportObjScene(Scene& scene, const std::string& filepath);
    static std::string nameFromPath(const std::string& path);
    static std::vector<char> readFile(const std::string& filename);
};