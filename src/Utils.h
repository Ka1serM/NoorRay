#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#include "Image.h"
#include "Texture.h"
#include "shaders/SharedStructs.h"

class Utils {
public:
    static void loadObj(const std::shared_ptr<Context> &context, const std::string &filepath, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, std
                        ::vector<Face> &faces, std::vector<Material> &materials, std::vector<Texture> &
                        textures);

    static std::vector<char> readFile(const std::string& filename);
};