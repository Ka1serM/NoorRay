#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "Image.h"
#include "Texture.h"
#include "shaders/SharedStructs.h"

class Utils {
public:
    static void loadGltf(const std::string &filename, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices,
                  std::vector<Face> &faces, std::vector<Material> &materials, std::vector<PointLight> &pointLights);

    static void loadObj(const Context &context, const std::string &filepath, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, std
                        ::vector<Face> &faces, std::vector<Material> &materials, std::vector<PointLight> &pointLights, std::vector<Texture> &
                        textures);

    static std::vector<char> readFile(const std::string& filename);
};