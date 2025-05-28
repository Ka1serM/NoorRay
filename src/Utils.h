#pragma once

#include <memory>
#include <vector>
#include <string>

#include "Vulkan/Texture.h"
#include "Shaders/SharedStructs.h"

class Utils {
public:
    static void loadObj(Context& context, const std::string &filepath, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, std
                        ::vector<Face> &faces, std::vector<Material> &materials, std::vector<Texture> &
                        textures);

    static std::vector<char> readFile(const std::string& filename);
};