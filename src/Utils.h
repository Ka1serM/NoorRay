#pragma once

#include <vector>
#include <string>

#include "Scene/Scene.h"
#include "Shaders/PathTracing/SharedStructs.h"
#include "Vulkan/Renderer.h"

class Utils {
public:
    static void loadCrtScene(Scene& scene, const std::string& filepath, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces, std::vector<Material>& materials);
    static void loadObj(Scene& scene, const std::string& filepath, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces, std::vector<Material>& materials);

    static std::string nameFromPath(const std::string& path);
    static std::vector<char> readFile(const std::string& filename);
};
